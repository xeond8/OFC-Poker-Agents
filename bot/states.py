from bot.environment import Environment, assignments, is_valid, Card, Board, valid_starters
from copy import deepcopy
import numpy as np

class Action():
    def __init__(self, first_player, upper, middle, bottom):
        self.first_player = first_player
        self.upper = upper
        self.middle = middle
        self.bottom = bottom

    def streets(self):
        return [self.upper, self.middle, self.bottom]

    def __str__(self):
        s = ""
        for street in self.streets():
            s += ''.join([Card.int_to_pretty_str(card) for card in street]) + "/"
        return s[:-1]

    def __hash__(self):
        return hash((*self.upper, self.first_player, *self.middle, self.first_player, *self.bottom))


class State():
    def __init__(self, env: Environment):
        self.env = env
    
    def clone(self):
        return State(self.env.copy())

    def get_possible_actions(self):
        if self.env.n_move == 0:
            pass
        else:
            possible_actions = []
            if self.env.first_player:
                capacities = [3 - len(self.env.board1.upper), 5 - len(self.env.board1.middle),
                              5 - len(self.env.board1.bottom)]
            else:
                capacities = [3 - len(self.env.board2.upper), 5 - len(self.env.board2.middle),
                              5 - len(self.env.board2.bottom)]
            valid_pos = [assignment for assignment in assignments if is_valid(assignment, capacities)]
            for i in range(3):
                cur_hand = self.env.hand[:i] + self.env.hand[i + 1:]
                for pos in valid_pos:
                    cur_move = [[], [], []]
                    for it in zip(cur_hand, pos):
                        cur_move[it[1]].append(it[0])
                    action = Action(self.env.first_player, cur_move[0], cur_move[1], cur_move[2])
                    possible_actions.append(action)

            return possible_actions

    def take_action(self, action):
        t_env = deepcopy(self.env)
        if action.first_player:
            for cards, street in zip(action.streets(), t_env.board1.streets()):
                street.extend(cards)
            t_env.first_player = False

        else:
            for cards, street in zip(action.streets(), t_env.board2.streets()):
                street.extend(cards)
            t_env.first_player = True
            t_env.n_move += 1
        t_env.hand = t_env.deck[:3]
        t_env.deck = t_env.deck[3:]

        return State(t_env)

    def is_terminal(self):
        return self.env.n_move == 5

    def get_reward(self):
        return self.env.score_calc()
        '''if (self.env.first_player and self.env.board1.dead) or (not self.env.first_player and self.env.board2.dead):
            return -1
        else:
            return self.env.score_calc()'''


class StateOneBoard():
    def __init__(self, env: Environment):
        self.env = env

    def clone(self):
        return StateOneBoard(self.env.copy()) 

    def get_possible_actions(self):
        if self.env.n_move == 0:
            poss_actions = []
            for starter in valid_starters:
                cur_move = [[], [], []]
                for card, place in zip(self.env.hand, starter):
                    cur_move[place].append(card)
                action = Action(self.env.first_player, cur_move[0], cur_move[1], cur_move[2])
                poss_actions.append(action)
            return poss_actions
        else:
            possible_actions = []
            if self.env.first_player:
                capacities = [3 - len(self.env.board1.upper), 5 - len(self.env.board1.middle),
                              5 - len(self.env.board1.bottom)]
            else:
                capacities = [3 - len(self.env.board2.upper), 5 - len(self.env.board2.middle),
                              5 - len(self.env.board2.bottom)]
            valid_pos = [assignment for assignment in assignments if is_valid(assignment, capacities)]
            for i in range(3):
                cur_hand = self.env.hand[:i] + self.env.hand[i + 1:]
                for pos in valid_pos:
                    cur_move = [[], [], []]
                    for it in zip(cur_hand, pos):
                        cur_move[it[1]].append(it[0])
                    action = Action(self.env.first_player, cur_move[0], cur_move[1], cur_move[2])
                    possible_actions.append(action)

            return possible_actions

    def take_action(self, action):
        t_env = deepcopy(self.env)
        if action.first_player:
            for cards, street in zip(action.streets(), t_env.board1.streets()):
                street.extend(cards)

        else:
            for cards, street in zip(action.streets(), t_env.board2.streets()):
                street.extend(cards)

        t_env.n_move += 1
        t_env.hand = t_env.deck[:3]
        t_env.deck = t_env.deck[3:]

        return StateOneBoard(t_env)

    def is_terminal(self):
        return self.env.n_move == 5

    def is_correct(self):
        if self.env.first_player:
            counts = list(map(len, self.env.board1.streets()))
        else:
            counts = list(map(len, self.env.board2.streets()))

        capacities = [3, 5, 5]
        return all(counts[i] <= capacities[i] for i in range(3))

    def get_reward(self):
        if self.env.first_player:
            return self.env.board1.count_royalties()
        else:
            return self.env.board2.count_royalties()

    def flatten(self):
        vec = np.zeros(84)
        if self.env.first_player:
            board = self.env.board1
        else:
            board = self.env.board2

        for card_n in range(3):
            if len(board.upper) > card_n:
                card = board.upper[card_n]
                suit = int(np.log2(Card.get_suit_int(card)))
                rank = Card.get_rank_int(card)
                vec[(0 + card_n)*5] = rank
                vec[(0 + card_n)*5 + 1 + suit] = 1

        for card_n in range(5):
            if len(board.middle) > card_n:
                card = board.middle[card_n]
                suit = int(np.log2(Card.get_suit_int(card)))
                rank = Card.get_rank_int(card)
                vec[(3 + card_n)*5] = rank
                vec[(3 + card_n)*5 + 1 + suit] = 1

        for card_n in range(5):
            if len(board.bottom) > card_n:
                card = board.bottom[card_n]
                suit = int(np.log2(Card.get_suit_int(card)))
                rank = Card.get_rank_int(card)
                vec[(8 + card_n)*5] = rank
                vec[(8 + card_n)*5 + 1 + suit] = 1

        capacities = [3 - len(board.upper), 5 - len(board.middle), 5 - len(board.bottom)]
        vec[13*5:13*5+3] = capacities
        vec[13*5+3] = self.env.n_move

        for card_n in range(3):
            card = self.env.hand[card_n]
            suit = int(np.log2(Card.get_suit_int(card)))
            rank = Card.get_rank_int(card)
            vec[(13 + card_n) * 5 + 4] = rank
            vec[(13 + card_n) * 5 + 4 + 1 + suit] = 1

        return vec

if __name__ == '__main__':
    card = Card.new("5c")
    print(np.log2(Card.get_suit_int(card), dtype=int))
