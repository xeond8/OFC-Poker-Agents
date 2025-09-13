from deuces import Card, Evaluator
import itertools

Card = Card()
Evaluator = Evaluator()

n = 5
a, b, c = 3, 5, 5

assignments_starters = list(itertools.product(range(3), repeat=n))

def is_valid(assignment, capacities):
    counts = [assignment.count(i) for i in range(3)]
    return all(counts[i] <= capacities[i] for i in range(3))


def is_valid_starters(assignment):
    counts = [assignment.count(i) for i in range(3)]
    if counts[2] <= 1:
        return False
    if counts[1] >= 4:
        return False
    if counts[0] >= 3:
        return False

    if counts[0] == 2 and counts[1] == 0:
        return False

    return True


capacities = [a, b, c]
valid_starters = [assignment for assignment in assignments_starters if is_valid_starters(assignment)]
assignments = list(itertools.product(range(3), repeat=2))

full_deck = []
for k in ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]:
    for m in ["h", "s", "d", "c"]:
        full_deck.append(Card.new(k + m))

middle_royalties = [50, 30, 20, 12, 8, 4, 2, 0, 0, 0]
bottom_royalties = [25, 15, 10, 6, 4, 2, 0, 0, 0, 0]


class Board:
    def __init__(self, upper: list = None, middle: list = None, bottom: list = None, n_move: int = 0):
        self.upper = upper if upper is not None else []
        self.middle = middle if middle is not None else []
        self.bottom = bottom if bottom is not None else []

        self.upper_eval = None
        self.middle_eval = None
        self.bottom_eval = None

        self.roaylties = None
        self.dead = False

        self.n_move = n_move

    def evaluate_upper(self):
        if len(self.upper) < 3:
            return 0
        if self.upper_eval:
            return self.upper_eval

        r1, r2, r3 = [Card.get_rank_int(x) for x in self.upper]
        if r1 == r2 and r2 == r3:
            self.upper_eval = 2467 - 66 * r1

        elif r1 == r2 or r2 == r3 or r1 == r3:
            r = r1 if r1 == r2 else r2 if r2 == r3 else r3
            t = r1 if r1 != r else r2 if r2 != r else r3

            n = t - 2 - (1 if t > r else 0)
            self.upper_eval = int(6185 - 220 * r - n * (n + 1) * (n + 2) / 6)

        else:
            r4 = 0
            while r4 in [r1, r2, r3]:
                r4 += 1
            M, m = max([r1, r2, r3, r4]), min([r1, r2, r3, r4])

            if m >= 2:
                r5 = 0
            elif m == 1:
                if M - m > 3:
                    r5 = 0
                else:
                    r5 = M + 2
            else:
                if M - m == 3:
                    r5 = M + 2
                elif M - m == 4:
                    r5 = M + 1
                else:
                    r5 = m + 1
                    while r5 in [r1, r2, r3, r4]:
                        r5 += 1

            card4 = 1 << r4 << 16 | 1 << 12 | r4 << 8 | Card.PRIMES[r4]
            card5 = 1 << r5 << 16 | 2 << 12 | r5 << 8 | Card.PRIMES[r5]
            self.upper_eval = Evaluator.evaluate(self.upper, [card4, card5])
            if self.upper_eval == 1609:
                if 3 not in [r1, r2, r3]:
                    self.upper_eval = 6678
                elif 2 in [r1, r2, r3]:
                    self.upper_eval = 6676
                else:
                    self.upper_eval = 6677
        return self.upper_eval

    def evaluate_middle(self):
        if len(self.middle) < 5:
            return 0
        if self.middle_eval:
            return self.middle_eval
        self.middle_eval = Evaluator.evaluate(self.middle, [])
        return self.middle_eval

    def evaluate_bottom(self):
        if len(self.bottom) < 5:
            return 0
        if self.bottom_eval:
            return self.bottom_eval
        self.bottom_eval = Evaluator.evaluate(self.bottom, [])
        return self.bottom_eval

    def evaluate(self):
        up_sc = self.evaluate_upper()
        mid_sc = self.evaluate_middle()
        bot_sc = self.evaluate_bottom()

        scores = [x for x in [up_sc, mid_sc, bot_sc] if x > 0]

        self.dead = any(scores[i] < scores[i + 1] for i in range(len(scores) - 1))

    def evals(self):
        return [self.upper_eval, self.middle_eval, self.bottom_eval]

    def count_royalties(self, fantasy_e=11):
        if self.roaylties:
            return self.roaylties
        self.evaluate()
        if self.dead:
            return 0

        s = 0

        r1, r2, r3 = [Card.get_rank_int(x) for x in self.upper]
        if r1 == r2 and r2 == r3:
            s += 10 + r1
        else:
            r = max(r1 if r1 == r2 else r2 if r2 == r3 else r3 if r3 == r1 else 3, 3)
            s += (r - 3)

            if fantasy_e and r >= 10:
                s += fantasy_e

        mr = min(self.middle_eval - 1, Evaluator.get_rank_class(self.middle_eval))
        s += middle_royalties[mr]

        br = min(self.bottom_eval - 1, Evaluator.get_rank_class(self.bottom_eval))
        s += bottom_royalties[br]
        self.roaylties = s
        return s

    def streets(self):
        return [self.upper, self.middle, self.bottom]

    def update_streets(self, streets):
        self.upper, self.middle, self.bottom = streets


    def __str__(self):
        s = ""
        for street in self.streets():
            s += ''.join([Card.int_to_pretty_str(card) for card in street]) + "\n"
        return s


class Environment:
    def __init__(self, board1: Board = None, board2: Board = None, dump1: list = None, dump2: list = None,
                 hand: list = None, deck: list = None, n_move: int = 0, first_player: bool = True, text: bool = True,
                 print_score: bool = False, print_dump: int = 0):
        self.board1 = board1 if board1 is not None else Board()
        self.board2 = board2 if board2 is not None else Board()
        self.dump1 = dump1 if dump1 is not None else []
        self.dump2 = dump2 if dump2 is not None else []
        self.hand = hand if hand is not None else []
        self.deck = deck if deck is not None else full_deck
        self.n_move = n_move
        self.first_player = first_player
        self.text = text
        self.print_score = print_score
        self.print_dump = print_dump  # 0 - all, 1 - first, 2 - second
        self.antifantasy = False

    def visible_deck(self):
        return [x for x in self.deck if x not in self.board1.upper + self.board1.middle + self.board1.bottom + self.board2.upper + self.board2.middle + self.board2.bottom + (
            self.dump2 if self.first_player else self.dump1) + self.hand]

    def __str__(self):
        s = "-" * 40 + "\n"
        if self.text:
            s += "First player: "
            if self.board1.dead:
                s += "(Dead)"
            s += "\n"
        s += self.board1.__str__()

        if self.print_dump in [0, 1]:
            if self.text:
                s += "Dump: \n"
            s += ''.join([Card.int_to_pretty_str(card) for card in self.dump1]) + "\n"

        s += "\n"

        if self.text:
            s += "Second player: "
            if self.board2.dead:
                s += "(Dead)"
            s += "\n"
        s += self.board2.__str__()

        if self.print_dump in [0, 2]:
            if self.text:
                s += "Dump: \n"
            s += ''.join([Card.int_to_pretty_str(card) for card in self.dump2]) + "\n"

        if self.print_score:
            n = self.score_calc()
            if self.text:
                s += "Score: \n"
            s += f"{n} : {-n}"
        return s

    def score_calc(self):
        self.board1.evaluate()
        self.board2.evaluate()

        royals = self.board1.count_royalties() - self.board2.count_royalties()

        if self.board1.dead or self.board2.dead:
            pairs = 6 * self.board2.dead - 6 * self.board1.dead
        else:
            street_compare = []
            for n, m in zip(self.board1.evals(), self.board2.evals()):
                if n < m:
                    street_compare.append(1)
                elif n > m:
                    street_compare.append(-1)
                else:
                    street_compare.append(0)

            if street_compare[0] == 0:
                for r1, r2 in zip(sorted(self.board1.upper, reverse=True), sorted(self.board2.upper, reverse=True)):
                    if Card.get_rank_int(r1) != Card.get_rank_int(r2):
                        street_compare[0] = 1 if r1 > r2 else -1
                        break

            pairs = sum(street_compare)

        if pairs == 3 or pairs == -3:
            pairs *= 2

        return royals + pairs


    def update_situation(self):
        b1_n_cards = sum(list(map(len, self.board1.streets())))
        b2_n_cards = sum(list(map(len, self.board2.streets())))

        if b1_n_cards == b2_n_cards:
            self.first_player = True
        else:
            self.first_player = False

        self.n_move = 0 if min(b1_n_cards, b2_n_cards) == 0 else (min(b1_n_cards, b2_n_cards) - 3) // 2

        if b2_n_cards == 0:
            self.antifantasy = True
            self.first_player = True
            self.n_move = 0 if b1_n_cards == 0 else (b1_n_cards - 3) // 2

def streets_to_str(streets: list, end="\n"):
    s = ""
    for street in streets:
        s += ''.join([Card.int_to_pretty_str(card) for card in street]) + end
    return s[:-1]


def str_streets_to_str(streets: list):
    s = ""
    for street in streets:
        s += ''.join([" " if len(card) < 2 else Card.int_to_pretty_str(Card.new(card)) for card in street]) + "\n"
    return s


def str_to_streets(str_streets:list):
    return [[Card.new(x) for x in street if x != ""] for street in str_streets]

if __name__ == '__main__':
    board1 = Board()
    board1.middle = [Card.new("2c"), Card.new("4d"), Card.new("5d"), Card.new("6c"), Card.new("Ad")]
    board1.evaluate_middle()
    print(board1.middle_eval)
