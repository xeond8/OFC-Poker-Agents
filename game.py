from bot.environment import Environment, full_deck
from bot.agents import Agent, RandomAgent, MonteCarloOneBoardAgent, MoveDependentAgent, HumanAgent, MCTSAgent, MiniMaxAgent, FantasyAgent, BestAgentFP, BestAgentSP
from bot.deuces import Card, Evaluator

from tqdm import trange
import random
import copy

Card = Card()

class Game():
    def __init__(self, env: Environment = None):
        self.env = env if env is not None else Environment()

    def play_game(self, agent1: Agent = RandomAgent(), agent2: Agent = RandomAgent(), print_env: int = 0, start_n_move: int = 0):
        # print env: -1 - None, 0 - all, 1 - first, 2 - second
        if isinstance(agent2, FantasyAgent):
            cards = self.env.deck[-14:]
            f_streets, _, _, f = agent2.act(cards, count_fantasy=True)


        for n_move in range(start_n_move, 5):
            self.env.n_move = n_move
            if n_move == 0:
                if print_env in [0, 1]:
                    self.env.print_dump = print_env
                    print(self.env)
                self.env.first_player = True
                self.env.hand = self.env.deck[:5]
                self.env.deck = self.env.deck[5:]

                move, _, _ = agent1.act(self.env)
                for card, street in zip(move, self.env.board1.streets()):
                    street.extend(card)

                if not isinstance(agent2, FantasyAgent):
                    if print_env in [0, 2]:
                        self.env.print_dump = print_env
                        print(self.env)
                    self.env.first_player = False
                    self.env.hand = self.env.deck[:5]
                    self.env.deck = self.env.deck[5:]
                    move, _, _ = agent2.act(self.env)
                    for card, street in zip(move, self.env.board2.streets()):
                        street.extend(card)

            else:
                if print_env in [0, 1]:
                    self.env.print_dump = print_env
                    print(self.env)
                self.env.first_player = True
                self.env.hand = self.env.deck[:3]
                self.env.deck = self.env.deck[3:]
                move, dump, _ = agent1.act(copy.deepcopy(self.env))
                self.env.dump1.append(dump)
                for card, street in zip(move, self.env.board1.streets()):
                    street.extend(card)

                if not isinstance(agent2, FantasyAgent):
                    if print_env in [0, 2]:
                        self.env.print_dump = print_env
                        print(self.env)
                    self.env.first_player = False
                    self.env.hand = self.env.deck[:3]
                    self.env.deck = self.env.deck[3:]
                    move, dump, _ = agent2.act(copy.deepcopy(self.env))
                    self.env.dump2.append(dump)
                    for card, street in zip(move, self.env.board2.streets()):
                        street.extend(card)

        if isinstance(agent2, FantasyAgent):
            self.env.board2.upper = f_streets[0]
            self.env.board2.middle = f_streets[1]
            self.env.board2.bottom = f_streets[2]

            return f



if __name__ == '__main__':
    scores = []
    env = Environment()
    deck = full_deck
    fantasies = 0
    best_env = None
    best_score = 0
    for i in trange(100):

        t_env = copy.deepcopy(env)
        deck = random.sample(full_deck, len(full_deck))
        t_env.deck = deck
        t_env.print_score = False
        game = Game(t_env)
        fantasies += game.play_game(HumanAgent(), BestAgentSP, print_env=-1, start_n_move=0)
        score = t_env.score_calc()
        scores.append(score)
        t_env.print_score = True
        print(t_env)





