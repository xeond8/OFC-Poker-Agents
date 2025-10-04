import random
import math
import time

def f_info_key(state):
    env = state.env
    return (
        env.first_player,
        env.n_move,
        frozenset(env.hand),
        frozenset(env.board1.upper), frozenset(env.board1.middle), frozenset(env.board1.bottom),
        frozenset(env.board2.upper), frozenset(env.board2.middle), frozenset(env.board2.bottom)
    )

class ISMCTSNode:
    def __init__(self, info_key, possible_actions):
        self.info_key = info_key
        self.actions = {}
        for a in possible_actions:
            self.actions[hash(a)] = {'action': a, 'visits': 0, 'value': 0.0}
        self.visits = 0

    def ensure_action(self, action):
        h = hash(action)
        if h not in self.actions:
            self.actions[h] = {'action': action, 'visits': 0, 'value': 0.0}
        return h

class ISMCTS:
    def __init__(self, c=1.4,rollout_policy=None, two_players=True):
        self.c = c
        self.rollout_policy = rollout_policy
        self.two_players = two_players
        self.tree = {}

    def ensure_node(self, state):
        key = f_info_key(state)
        if key not in self.tree:
            poss = state.get_possible_actions()
            self.tree[key] = ISMCTSNode(key, poss)
        return self.tree[key]

    def select_action_uct(self, node):
        parent_n = max(1, node.visits)
        best_score = -float("inf")
        best_ah = None
        for ah, ainfo in node.actions.items():
            n_a = ainfo['visits']
            if n_a == 0:
                score = float('inf')
            else:
                exploit = ainfo['value']
                explore = self.c * math.sqrt(math.log(parent_n) / n_a)
                score = exploit + explore
            if score > best_score:
                best_score = score
                best_ah = ah
        return best_ah

    def rollout(self, state):
        cur = state
        while not cur.is_terminal():
            poss = cur.get_possible_actions()
            if not poss:
                break
            if self.rollout_policy is None:
                act = random.choice(poss)
            else:
                act = self.rollout_policy.act(cur.env, poss=poss)
            cur = cur.take_action(act)
        return cur.get_reward()

    def run_iteration(self, root_state):

        det_state = root_state.clone()
        det_state.env.deck = random.sample(det_state.env.deck, len(det_state.env.deck))

        state = det_state.clone()
        visited = []

        while not state.is_terminal():
            key = f_info_key(state)
            node = self.ensure_node(state)

            untried = [ah for ah, a in node.actions.items() if a['visits'] == 0]
            if untried:
                ah = random.choice(untried)
                action = node.actions[ah]['action']
            else:
                ah = self.select_action_uct(node)
                action = node.actions[ah]['action']

            visited.append((key, ah))

            state = state.take_action(action)

        if not state.is_terminal():
            reward = self.rollout(state)
        else:
            reward = state.get_reward()


        for (info_k, action_h) in visited:
            node = self.tree.get(info_k)
            if node is None:
                self.tree[info_k] = ISMCTSNode(info_k, [])
                node = self.tree[info_k]

            node.visits += 1
            ainfo = node.actions.get(action_h)

            if not ainfo['action'].first_player and self.two_players:
                real_reward = reward * (-1)
            else:
                real_reward = reward
            prev_vis = ainfo['visits']
            prev_mean = ainfo['value']
            new_vis = prev_vis + 1
            new_mean = prev_mean + (real_reward - prev_mean) / new_vis
            ainfo['visits'] = new_vis
            ainfo['value'] = new_mean

    def search(self, root_state, n_iter=2000, time_limit=False):
        root_key = f_info_key(root_state)
        self.ensure_node(root_state)
        start = time.time()
        for _ in range(n_iter):
            self.run_iteration(root_state)
            if time_limit and time.time() - start > time_limit:
                break 

        root_node = self.tree[root_key]

        best_action = None
        best_visits = -1
        best_mean = -float('inf')
        for ah, ainfo in root_node.actions.items():
            if ainfo['visits'] > best_visits:
                best_visits = ainfo['visits']
                best_mean = ainfo['value']
                best_action = ainfo['action']
            elif ainfo['visits'] == best_visits and ainfo['value'] > best_mean:
                best_mean = ainfo['value']
                best_action = ainfo['action']

        return best_action.streets()
    


def ismcts(root_state, n_iter=20000, c=5, rollout_policy=None, two_players=True, time_limit=False):
    engine = ISMCTS(c=c, rollout_policy=rollout_policy, two_players=two_players)
    
    return engine.search(root_state, n_iter=n_iter, time_limit=time_limit)
