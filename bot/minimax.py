from copy import deepcopy
import random

class MiniMaxNode():
    def __init__(self, state, two_players=True):
        self.state = state
        self.two_players = two_players

    def evaluate_action(self):
        if self.state.is_terminal():
            try:
                return self.state.get_reward()
            except:
                print(self.state.env)

        possible_actions =  self.state.get_possible_actions()
        scores = []
        for action in possible_actions:
            temp_state = self.state.take_action(action)
            temp_node = MiniMaxNode(temp_state, two_players=self.two_players)
            scores.append(temp_node.evaluate_action())
        if self.two_players and not self.state.env.first_player:
            return min(scores)
        else:
            return max(scores)


def minimax(root_state, n_decks=100, two_players=True):
    poss_actions = root_state.get_possible_actions()
    rewards = dict((hash(act), 0) for act in poss_actions)
    hash_actions = dict((hash(act), act.streets()) for act in poss_actions)

    deck = root_state.env.visible_deck()
    for _ in range(n_decks):
        deck = random.sample(deck, len(deck))
        for action in poss_actions:
            t_root_state = root_state.take_action(action)
            t_root_state.env.deck = deepcopy(deck)
            root_node = MiniMaxNode(t_root_state, two_players=two_players)
            score = root_node.evaluate_action()
            rewards[hash(action)] += score
    if two_players and not root_state.env.first_player:
        key = min(rewards, key=rewards.get)
    else:
        key = max(rewards, key=rewards.get)
    return hash_actions[key]




    