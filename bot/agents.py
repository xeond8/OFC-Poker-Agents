from deuces import Card, Evaluator
from bot.environment import Environment, Board, full_deck, assignments, valid_starters, is_valid, streets_to_str
from bot.mcts import mcts, mcts_new
from bot.ismcts import ismcts
from bot.minimax import minimax
from bot.states import State, StateOneBoard
import numpy as np
import random
import time
import copy
import itertools

Evaluator = Evaluator()


class Agent():
    def __init__(self):
        pass

    def act(self, env:Environment):
        env.update_situation()


class RandomAgent(Agent):
    def act(self, env: Environment):
        if env.n_move == 0:
            res = random.choice(valid_starters)

            turn = [[], [], []]

            for it in zip(env.hand, res):
                turn[it[1]].append(it[0])

            return turn, None, 0
        else:
            i = np.random.randint(0, 2)
            dump_card = env.hand[i]
            cur_hand = env.hand[:i] + env.hand[i + 1:]

            if env.first_player:
                capacities = [3 - len(env.board1.upper), 5 - len(env.board1.middle), 5 - len(env.board1.bottom)]
            else:
                capacities = [3 - len(env.board2.upper), 5 - len(env.board2.middle), 5 - len(env.board2.bottom)]

            valid_pos = [assignment for assignment in assignments if is_valid(assignment, capacities)]
            res = random.choice(valid_pos)

            turn = [[], [], []]

            for it in zip(cur_hand, res):
                turn[it[1]].append(it[0])

            return turn, dump_card, 0


class MonteCarloOneBoardAgent(Agent):
    def __init__(self, n_mc=None):
        super().__init__()
        self.n_mc = n_mc if n_mc is not None else [1, 2, 15, 50]

    def act(self, env: Environment):
        if env.first_player:
            return self.monte_carlo_ev(env.board1, env.hand, env.deck + env.dump2, env.n_move, env.n_move)
        else:
            return self.monte_carlo_ev(env.board2, env.hand, env.deck + env.dump1, env.n_move, env.n_move)

    def monte_carlo_ev(self, board: Board, hand: list, deck: list, n_move: int, base_n_move: int):
        def count_q(turn):
            t_board = board.copy()
            for cards, street in zip(turn, t_board.streets()):
                street.extend(cards)

            if n_move == 4:
                return t_board.count_royalties()

            eq = 0
            n = self.n_mc[base_n_move]
            for _ in range(n):
                p = random.randint(0, len(deck) - 1)
                if p == 0:
                    new_hand = deck[-1:] + deck[:2]
                    new_deck = deck[2:-1]
                elif p == len(deck) - 1:
                    new_hand = deck[:1] + deck[-2:]
                    new_deck = deck[1:-2]
                else:
                    new_hand = deck[p - 1:p + 2]
                    new_deck = deck[:p - 1] + deck[p + 2:]

                eq += self.monte_carlo_ev(t_board, new_hand, new_deck, n_move + 1, base_n_move)[2]
            return eq / n

        best_score = -1
        best_move = [[], [], []]
        if n_move == 0:
            pass
        else:
            capacities = [3 - len(board.upper), 5 - len(board.middle), 5 - len(board.bottom)]
            valid_pos = [assignment for assignment in assignments if is_valid(assignment, capacities)]
            for i in range(3):
                cur_hand = hand[:i] + hand[i + 1:]
                dump_card = hand[i]
                for pos in valid_pos:
                    cur_move = [[], [], []]
                    for it in zip(cur_hand, pos):
                        cur_move[it[1]].append(it[0])
                    score = count_q(cur_move)
                    if score > best_score:
                        best_score = score
                        best_move = cur_move

        return best_move, dump_card, best_score


class MCTSAgent(Agent):
    def __init__(self, c=5, n_iter=2000, n_decks=100, two_players=True):
        super().__init__()
        self.c = c
        self.n_iter = n_iter
        self.n_decks = n_decks
        self.two_players = two_players

    def act(self, env: Environment,  time_limit=12):
        if self.two_players:
            #action =  mcts(State(env), n_iter=self.n_iter, n_decks=self.n_decks, c=self.c, two_players=self.two_players, time_limit=time_limit)
            action =  mcts_new(State(env), n_iter=self.n_iter, c=self.c, two_players=self.two_players, time_limit=time_limit)
        else:
            #action = mcts(StateOneBoard(env), n_iter=self.n_iter, n_decks=self.n_decks, c=self.c, two_players=self.two_players, time_limit=time_limit)
            action = mcts_new(StateOneBoard(env), n_iter=self.n_iter, c=self.c, two_players=self.two_players, time_limit=time_limit)
            
        dump = [x for x in env.hand if all(x not in sublist for sublist in action)]
        if env.n_move == 0:
            if len(dump) != 0:
                raise ValueError
            else:
                return action, None, 0

        if env.n_move > 0:
            if len(dump) != 1:
                raise ValueError
            else:
                return action, dump[0], 0
            
class ISMCTSAgent(Agent):
    def __init__(self, c=5, n_iter=2000,two_players=True, rollout_policy=None):
        super().__init__()
        self.c = c
        self.n_iter = n_iter
        self.rollout_policy = rollout_policy
        self.two_players = two_players

    def act(self, env: Environment,  time_limit=12):
        if self.two_players:
            action =  ismcts(State(env), n_iter=self.n_iter, c=self.c, rollout_policy=self.rollout_policy, two_players=self.two_players, time_limit=time_limit)
        else:
            action = ismcts(StateOneBoard(env), n_iter=self.n_iter, c=self.c, rollout_policy=self.rollout_policy, two_players=self.two_players, time_limit=time_limit)
            
        dump = [x for x in env.hand if all(x not in sublist for sublist in action)]
        if env.n_move == 0:
            if len(dump) != 0:
                raise ValueError
            else:
                return action, None, 0

        if env.n_move > 0:
            if len(dump) != 1:
                raise ValueError
            else:
                return action, dump[0], 0


class MiniMaxAgent(Agent):
    def __init__(self, n_decks=100, two_players=True):
        super().__init__()
        self.n_decks = n_decks
        self.two_players = two_players


    def act(self, env: Environment):
        if self.two_players:
            action = minimax(State(env), n_decks=self.n_decks, two_players=self.two_players)
        else:
            action = minimax(StateOneBoard(env), n_decks=self.n_decks, two_players=self.two_players)
        dump = [x for x in env.hand if all(x not in sublist for sublist in action)]
        if env.n_move == 0:
            if len(dump) != 0:
                raise ValueError
            else:
                return action, None, 0

        if env.n_move > 0:
            if len(dump) != 1:
                raise ValueError
            else:
                return action, dump[0], 0



class MoveDependentAgent(Agent):
    def __init__(self, agents):
        super().__init__()
        self.agents = agents
    def act(self, env: Environment, time_limit=12):
        return self.agents[env.n_move].act(env, time_limit=time_limit)

class HumanAgent(Agent):
    def __init__(self):
        super().__init__()

    def act(self, env: Environment):
        print(f"Your hand:\n {''.join([Card.int_to_pretty_str(card) for card in env.hand])} \n")
        print("Place the cards on the streets:")
        upper = [Card.new(card) for card in input().split()]
        middle = [Card.new(card) for card in input().split()]
        bottom = [Card.new(card) for card in input().split()]
        move = [upper, middle, bottom]
        dump = [card for card in env.hand if card not in upper + middle + bottom]

        if env.n_move == 0:
            if len(dump) != 0:
                raise ValueError
            else:
                return move, None, 0

        if env.n_move > 0:
            if len(dump) != 1:
                raise ValueError
            else:
                return move, dump[0], 0

class SimpleFantasyLikeOneBoard(Agent):
    def __init__(self, n_decks=100):
        super().__init__()
        self.n_decks = n_decks
    def act(self, env: Environment, time_limit=False):
        root_state = StateOneBoard(env)
        deck = env.visible_deck()
        actions_scores = {}
        actions_hash = {}
        for act in root_state.get_possible_actions():
            actions_hash[hash(act)] = act

        start = time.time()
        for _ in range(self.n_decks // 8):
            cards = random.sample(deck, 8 - 2*env.n_move + (env.n_move == 3))
            for act in root_state.get_possible_actions():
                child_state = root_state.take_action(act)
                ref_board = child_state.env.board1 if child_state.env.first_player else child_state.env.board2
                best_score = -1
                for bottom in itertools.combinations(cards, 5 - len(ref_board.bottom)):
                    m_board = ref_board.copy()
                    m_board.bottom.extend(bottom)
                    m_board.evaluate_bottom()
                    if env.n_move <= 1:
                        br = Evaluator.get_rank_class(m_board.bottom_eval)
                        if br > 7:
                            continue
                    for middle in itertools.combinations(set(cards) - set(bottom), 5 - len(ref_board.middle)):
                        for upper in itertools.combinations(set(cards) - set(middle) - set(bottom), 3 - len(ref_board.upper)):
                            board = m_board.copy()
                            board.upper.extend(upper)
                            board.middle.extend(middle)
                            s = board.count_royalties()
                            if s > best_score:
                                best_score = s
                actions_scores[hash(act)] = actions_scores.get(hash(act), 0) + best_score
            
            if time_limit and time.time() - start > time_limit:
                break

        best_scores = sorted(actions_scores.items(), key=lambda x: x[1], reverse=True)
        best_actions = []
        for h, score in best_scores[:(5 if len(root_state.get_possible_actions()) < 30 else 6)]:
            best_actions.append(actions_hash[h])


        for _ in range( 7 * self.n_decks // 8):
            cards = random.sample(deck, 8 - 2*env.n_move)
            for act in best_actions:
                child_state = root_state.take_action(act)
                ref_board = child_state.env.board1 if child_state.env.first_player else child_state.env.board2
                best_score = -1
                for bottom in itertools.combinations(cards, 5 - len(ref_board.bottom)):
                    m_board = ref_board.copy()
                    m_board.bottom.extend(bottom)
                    m_board.evaluate_bottom()
                    if env.n_move <= 1:
                        br = Evaluator.get_rank_class(m_board.bottom_eval)
                        if br > 7:
                            continue
                    for middle in itertools.combinations(set(cards) - set(bottom), 5 - len(ref_board.middle)):
                        for upper in itertools.combinations(set(cards) - set(middle) - set(bottom), 3 - len(ref_board.upper)):
                            board = m_board.copy()
                            board.upper.extend(upper)
                            board.middle.extend(middle)
                            s = board.count_royalties()
                            if s > best_score:
                                best_score = s
                actions_scores[hash(act)] = actions_scores.get(hash(act), 0) + best_score
            
            if time_limit and time.time() - start > time_limit:
                break
        
        sorted_scores = sorted(actions_scores.items(), key=lambda x:x[1], reverse=True)
        for act, score in sorted_scores[:5]:
            print(actions_hash[act], np.round(score / self.n_decks, 3))
        best_hash = max(actions_scores.items(), key=lambda x: x[1])[0]
        best_score = actions_scores[best_hash] / self.n_decks
        best_action = actions_hash[best_hash]
        dump_card = [x for x in env.hand if x not in best_action.upper + best_action.middle + best_action.bottom]


        if dump_card:
            return best_action.streets(), dump_card[0], best_score
        else:
            return best_action.streets(), None, best_score




class SimpleFantasyLikeTwoBoards(Agent):
    def __init__(self, n_decks=100, no_random:bool = False, ret_action: bool = False):
        super().__init__()
        self.n_decks = n_decks
        self.no_random = no_random
        self.ret_action = ret_action

    def act(self, env: Environment, poss = False, broke_pen=-1, time_limit=False):
        if env.n_move == 0:
            pass
        else:
            deck = env.visible_deck()
            b1_n_cards = 13 - sum(list(map(len, env.board1.streets()))) + (env.n_move == 3 and env.first_player == True)
            b2_n_cards = 13 - sum(list(map(len, env.board2.streets()))) + (env.n_move == 3 and env.first_player == False)
            root_state = StateOneBoard(env)
            actions_scores = {}
            actions_hash = {}
            if not poss:
                poss = root_state.get_possible_actions()
            for act in poss:
                actions_hash[hash(act)] = act

            start = time.time()

            for _ in range(self.n_decks // 6):
                if self.no_random:
                    cards = deck[:b1_n_cards + b2_n_cards - 2]
                else:
                    cards = random.sample(deck, b1_n_cards + b2_n_cards - 2)
                if env.first_player:
                    enemy_cards = cards[:b2_n_cards]
                    player_cards = cards[b2_n_cards:]
                    enemy_board = env.board2
                else:
                    enemy_cards = cards[:b1_n_cards]
                    player_cards = cards[b1_n_cards:]
                    enemy_board = env.board1

                best_enemy_board = None
                best_enemy_score = broke_pen - 1
                for bottom in itertools.combinations(enemy_cards, 5 - len(enemy_board.bottom)):
                    for middle in itertools.combinations(set(enemy_cards) - set(bottom), 5 - len(enemy_board.middle)):
                        for upper in itertools.combinations(set(enemy_cards) - set(middle) - set(bottom), 3 - len(enemy_board.upper)):
                            board = enemy_board.copy()
                            board.upper.extend(upper)
                            board.middle.extend(middle)
                            board.bottom.extend(bottom)
                            s = board.count_royalties()
                            if board.dead:
                                s = -1
                            if s > best_enemy_score:
                                best_enemy_score = s
                                best_enemy_board = board.copy()


                for act in root_state.get_possible_actions():
                    child_state = root_state.take_action(act)
                    ref_board = child_state.env.board1 if env.first_player else child_state.env.board2
                    best_res = -10000 * (1 if env.first_player else -1)
                    for bottom in itertools.combinations(player_cards, 5 - len(ref_board.bottom)):
                        for middle in itertools.combinations(set(player_cards) - set(bottom), 5 - len(ref_board.middle)):
                            for upper in itertools.combinations(set(player_cards) - set(middle) - set(bottom),
                                                                 3 - len(ref_board.upper)):
                                tt_env = child_state.env.copy()
                                if tt_env.first_player:
                                    tt_env.board1.upper.extend(upper)
                                    tt_env.board1.middle.extend(middle)
                                    tt_env.board1.bottom.extend(bottom)
                                    tt_env.board2 = best_enemy_board
                                else:
                                    tt_env.board2.upper.extend(upper)
                                    tt_env.board2.middle.extend(middle)
                                    tt_env.board2.bottom.extend(bottom)
                                    tt_env.board1 = best_enemy_board
                                s = tt_env.score_calc()
                                if tt_env.first_player:
                                    best_res = max(best_res, s)
                                else:
                                    best_res = min(best_res, s)
                    actions_scores[hash(act)] = actions_scores.get(hash(act), 0) + best_res

            if actions_scores:
                best_scores = sorted(actions_scores.items(), key=lambda x: x[1], reverse=env.first_player)
                best_actions = []
                for h, score in best_scores[:(5 if len(root_state.get_possible_actions()) < 30 else 6)]:
                    best_actions.append(actions_hash[h])
            else:
                best_actions = list(actions_hash.values())


            actions_scores = {}

            for _ in range(max(5 * self.n_decks // 6, 1)):
                if self.no_random:
                    cards = deck[:b1_n_cards + b2_n_cards - 2]
                else:
                    cards = random.sample(deck, b1_n_cards + b2_n_cards - 2)
                if env.first_player:
                    enemy_cards = cards[:b2_n_cards]
                    player_cards = cards[b2_n_cards:]
                    enemy_board = env.board2
                else:
                    enemy_cards = cards[:b1_n_cards]
                    player_cards = cards[b1_n_cards:]
                    enemy_board = env.board1

                best_enemy_board = None
                best_enemy_score = broke_pen - 1
                for bottom in itertools.combinations(enemy_cards, 5 - len(enemy_board.bottom)):
                    for middle in itertools.combinations(set(enemy_cards) - set(bottom), 5 - len(enemy_board.middle)):
                        for upper in itertools.combinations(set(enemy_cards) - set(middle) - set(bottom),
                                                            3 - len(enemy_board.upper)):
                            board = enemy_board.copy()
                            board.upper.extend(upper)
                            board.middle.extend(middle)
                            board.bottom.extend(bottom)
                            s = board.count_royalties()
                            if board.dead:
                                s = -1
                            if s > best_enemy_score:
                                best_enemy_score = s
                                best_enemy_board = board.copy()


                best_enemy_board.count_royalties()

                for act in best_actions:
                    child_state = root_state.take_action(act)
                    ref_board = child_state.env.board1 if env.first_player else child_state.env.board2
                    best_res = -10000 * (1 if env.first_player else -1)
                    for bottom in itertools.combinations(player_cards, 5 - len(ref_board.bottom)):
                        for middle in itertools.combinations(set(player_cards) - set(bottom), 5 - len(ref_board.middle)):
                            for upper in itertools.combinations(set(player_cards) - set(middle) - set(bottom),
                                                                 3 - len(ref_board.upper)):
                                t_env = child_state.env.copy()
                                if t_env.first_player:
                                    t_env.board1.upper.extend(upper)
                                    t_env.board1.middle.extend(middle)
                                    t_env.board1.bottom.extend(bottom)
                                    t_env.board2 = best_enemy_board
                                else:
                                    t_env.board2.upper.extend(upper)
                                    t_env.board2.middle.extend(middle)
                                    t_env.board2.bottom.extend(bottom)
                                    t_env.board1 = best_enemy_board
                                s = t_env.score_calc()
                                if t_env.first_player:
                                    best_res = max(best_res, s)
                                else:
                                    best_res = min(best_res, s)
                    actions_scores[hash(act)] = actions_scores.get(hash(act), 0) + best_res
                
                if time_limit and time.time() - start > time_limit:
                    break

            if env.first_player:
                best_hash = max(actions_scores.items(), key=lambda x: x[1])[0]
            else:
                best_hash = min(actions_scores.items(), key=lambda x: x[1])[0]


            best_score = actions_scores[best_hash] / self.n_decks
            best_action = actions_hash[best_hash]
            dump_card = [x for x in env.hand if
                         x not in best_action.upper + best_action.middle + best_action.bottom]

            if self.ret_action:
                return best_action
            return best_action.streets(), dump_card[0], best_score





k = 1.

sfltb_3000 = SimpleFantasyLikeTwoBoards(n_decks=3000)
sfltb_1800 = SimpleFantasyLikeTwoBoards(n_decks=1800)

sflob_2500 = SimpleFantasyLikeOneBoard(n_decks=int(2500*k))
sflob_3500 = SimpleFantasyLikeOneBoard(n_decks=int(3500*k))
sflob_350 = SimpleFantasyLikeOneBoard(n_decks=int(350*k))

BestAgentFP = MoveDependentAgent([sflob_350, sflob_2500, sfltb_1800, sfltb_3000, sfltb_3000])
BestAgentSP = MoveDependentAgent([sflob_350, sflob_2500, sfltb_1800, sfltb_3000, sfltb_3000])
AntiFantasyAgent = MoveDependentAgent([sflob_350, sflob_2500, sflob_3500, sflob_3500, sflob_350])



class FantasyAgent(Agent):
    def __init__(self):
        super().__init__()

    def act(self, cards, fantasy_e=11, count_fantasy=False):
        best_score = 0
        best_board = None
        for upper in itertools.combinations(cards, 3):
            for middle in itertools.combinations(set(cards) - set(upper), 5):
                for bottom in itertools.combinations(set(cards) - set(middle) - set(upper), 5):
                    board = Board(list(upper), list(middle), list(bottom))
                    s = board.count_royalties(fantasy_e=0)

                    if (not board.dead) and (board.upper_eval <= 2467 or board.bottom_eval <= 166):
                        s += fantasy_e


                    if not best_board or s > best_score:
                        best_score = s
                        best_board = board

                    elif s == best_score:
                        if best_board.upper_eval > board.upper_eval:
                            best_score = s
                            best_board = board

                        elif best_board.upper_eval == board.upper_eval:
                            if best_board.middle_eval > board.middle_eval:
                                best_score = s
                                best_board = board

                            elif best_board.middle_eval == board.middle_eval and best_board.bottom_eval > board.bottom_eval:
                                best_score = s
                                best_board = board

        if count_fantasy == True:
            return best_board.streets(), None, best_score

        return best_board.streets(), None, best_score
