import math
import random
import time


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = state.get_possible_actions() if not state.is_terminal() else []

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=4):
        choices_weights = [
            child.value + c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        idx = random.randrange(len(self.untried_actions))
        action = self.untried_actions.pop(idx)
        next_state = self.state.take_action(action)
        child_node = MCTSNode(next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def rollout(self):
        current_state = self.state.clone()
        while not current_state.is_terminal():
            possible_actions = current_state.get_possible_actions()
            if not possible_actions:
                break
            action = random.choice(possible_actions)
            current_state = current_state.take_action(action)
        return current_state.get_reward()

    def backpropagate(self, reward, two_players: int = True):
        self.visits += 1
        self.value += (reward - self.value) / self.visits
        if self.parent:
            if two_players:
                self.parent.backpropagate(-reward, two_players=two_players)
            else:
                self.parent.backpropagate(reward, two_players=two_players)


def print_tree(node, indent=0):
    prefix = " " * (indent * 4)
    print(f"{prefix} First Player: {node.state.env.first_player}")
    print(f"{prefix}  Visits: {node.visits}, Value: {node.value:.2f}, Avg: {(node.value / node.visits if node.visits > 0 else 0):.2f}")
    print(f"{prefix}  Untried actions: {len(node.untried_actions)}")
    print(f"{prefix}  Children: {len(node.children)}")
    print()
    for child in node.children:
        print_tree(child, indent + 1)



def mcts(root_state, n_iter=2000, n_decks=50, c=5, two_players=True, time_limit=False):
    poss_actions = root_state.get_possible_actions()
    hash_actions = dict((hash(act), act.streets()) for act in poss_actions)
    nodes_mean = dict((hash(act), 0) for act in poss_actions)
    deck = root_state.env.visible_deck()
    start = time.time()
    for _ in range(n_decks):
        nodes_visits = dict((hash(act), 0) for act in poss_actions)
        nodes_values = dict((hash(act), 0) for act in poss_actions)

        t_root_state = root_state.clone()
        deck = random.sample(deck, len(deck))
        t_root_state.env.deck = deck
        root_node = MCTSNode(t_root_state)
        for _ in range(n_iter):
            node = root_node
            while node.is_fully_expanded() and node.children:
                node = node.best_child(c_param=c)
            if not node.is_fully_expanded():
                node = node.expand()
            reward = node.rollout()

            if node.state.env.first_player and two_players:
                reward *= -1

            node.backpropagate(reward, two_players)
        for node in root_node.children:
            action = node.action
            nodes_visits[hash(action)] += node.visits
            nodes_values[hash(action)] += node.value

        nodes_mean = dict([(x, nodes_values[x] + nodes_mean[x]) for x in nodes_visits])

        if time_limit and time.time() - start > time_limit:
            break

    key = max(nodes_mean, key=nodes_mean.get)
    return hash_actions[key]


def mcts_new(root_state, n_iter=2000, c=5, two_players=True, time_limit=False):
    poss_actions = root_state.get_possible_actions()
    hash_actions = {hash(act): act.streets() for act in poss_actions}
    nodes_visits = {hash(act): 0 for act in poss_actions}
    nodes_values = {hash(act): 0.0 for act in poss_actions}

    root_node = MCTSNode(root_state)
    start = time.time()
    for _ in range(n_iter):
        node = root_node

        # 1. SELECTION
        while node.is_fully_expanded() and node.children:
            node = node.best_child(c_param=c)

        # 2. EXPANSION
        if not node.is_fully_expanded():
            node = node.expand()

        # 3. ROLLOUT (с рандомизацией колоды)
        current_state = node.state.clone()
        # перемешиваем оставшуюся колоду
        current_state.env.deck = random.sample(current_state.env.deck, len(current_state.env.deck))
        reward = node.rollout()

        if node.state.env.first_player and two_players:
            reward *= -1

        # 4. BACKPROP
        node.backpropagate(reward, two_players)

        if time_limit and time.time() - start > time_limit:
            print(time.time() - start)
            break


    # собираем статистику по детям root
    for node in root_node.children:
        action = node.action
        nodes_visits[hash(action)] += node.visits
        nodes_values[hash(action)] += node.value * node.visits  # аккуратно суммируем
        print(action, nodes_visits[hash(action)], nodes_values[hash(action)]/nodes_visits[hash(action)])

    # усредняем значения
    nodes_mean = {k: (nodes_values[k] / nodes_visits[k] if nodes_visits[k] > 0 else -999)
                  for k in nodes_visits}

    key = max(nodes_mean, key=nodes_mean.get)
    return hash_actions[key]


if __name__ == '__main__':
    pass