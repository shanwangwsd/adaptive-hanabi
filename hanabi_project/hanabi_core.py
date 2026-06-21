import random
import copy
from dataclasses import dataclass

GREEN = 0


YELLOW = 1


WHITE = 2


BLUE = 3


RED = 4


ALL_COLORS = [GREEN, YELLOW, WHITE, BLUE, RED]


COLORNAMES = ["green", "yellow", "white", "blue", "red"]


COUNTS = [3, 2, 2, 2, 1]   # 每种颜色 1~5 各几张


def f(something):
    """semi-intelligently format cards"""
    if type(something) == list:
        return list(map(f, something))
    elif type(something) == dict:
        return {k: f(v) for k, v in something.items()}
    elif type(something) == tuple and len(something) == 2:
        return (COLORNAMES[something[0]], something[1])
    return something


def make_deck():
    deck = []
    for col in ALL_COLORS:
        for num, cnt in enumerate(COUNTS):
            for _ in range(cnt):
                deck.append((col, num + 1))
    random.shuffle(deck)
    return deck


def initial_knowledge():
    return [COUNTS[:] for _ in ALL_COLORS]


def hint_color(knowledge, color, truth):
    result = []
    for col in ALL_COLORS:
        if truth == (col == color):
            result.append(knowledge[col][:])
        else:
            result.append([0] * len(knowledge[col]))
    return result


def hint_rank(knowledge, rank, truth):
    result = []
    for col in ALL_COLORS:
        colknow = []
        for i, k in enumerate(knowledge[col]):
            colknow.append(k if truth == (i + 1 == rank) else 0)
        result.append(colknow)
    return result


def iscard(c, n):
    knowledge = []
    for col in ALL_COLORS:
        row = []
        for i in range(len(COUNTS)):
            row.append(1 if col == c and i + 1 == n else 0)
        knowledge.append(row)
    return knowledge


HINT_COLOR  = 0


HINT_NUMBER = 1


PLAY        = 2


DISCARD     = 3


CANDISCARD  = 128


class Action:
    def __init__(self, type, pnr=None, col=None, num=None, cnr=None):
        self.type = type
        self.pnr  = pnr
        self.col  = col
        self.num  = num
        self.cnr  = cnr

    def __str__(self):
        if self.type == HINT_COLOR:
            return f"hints {self.pnr} about all their {COLORNAMES[self.col]} cards"
        if self.type == HINT_NUMBER:
            return f"hints {self.pnr} about all their {self.num}"
        if self.type == PLAY:
            return f"plays their {self.cnr}"
        if self.type == DISCARD:
            return f"discards their {self.cnr}"

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return (self.type, self.pnr, self.col, self.num, self.cnr) == \
               (other.type, other.pnr, other.col, other.num, other.cnr)


@dataclass
class PublicState:
    """
    Public snapshot passed to Player.inform().

    The observer's own real hand is hidden as [] in `hands`. This prevents
    learning code from accidentally reading hidden self-cards or the deck through
    the full Game object.
    """
    hands: list
    knowledge: list
    board: list
    trash: list
    played: list
    hints: int
    current_player: int


def get_possible(knowledge):
    result = []
    for col in ALL_COLORS:
        for i, cnt in enumerate(knowledge[col]):
            if cnt > 0:
                result.append((col, i + 1))
    return result


def playable(possible, board):
    if not possible:
        return False
    return all(board[col][1] + 1 == nr for col, nr in possible)


def potentially_playable(possible, board):
    return any(board[col][1] + 1 == nr for col, nr in possible)


def discardable(possible, board):
    if not possible:
        return False
    return all(board[col][1] >= nr for col, nr in possible)


def potentially_discardable(possible, board):
    return any(board[col][1] >= nr for col, nr in possible)


def update_knowledge(knowledge, used):
    """
    Remove globally visible used cards from each possible-card knowledge table.
    This is the standard Hanabi public-card-count update: every unknown card
    position should rule out cards already played or discarded elsewhere.
    """
    result = copy.deepcopy(knowledge)
    for r in result:
        for (c, nr), cnt in used.items():
            r[c][nr - 1] = max(r[c][nr - 1] - cnt, 0)
    return result


def format_card(col_num):
    return COLORNAMES[col_num[0]] + " " + str(col_num[1])


def format_hand(hand):
    return ", ".join(map(format_card, hand))


def format_knowledge(k):
    result = ""
    for col in ALL_COLORS:
        for i, cnt in enumerate(k[col]):
            if cnt > 0:
                result += COLORNAMES[col] + " " + str(i + 1) + ": " + str(cnt) + "\n"
    return result


def format_intention(i):
    if isinstance(i, str): return i
    if i == PLAY:        return "Play"
    if i == DISCARD:     return "Discard"
    if i == CANDISCARD:  return "Can Discard"
    return "Keep"


def whattodo(knowledge, pointed, board):
    possible = get_possible(knowledge)
    play    = potentially_playable(possible, board)
    discard = potentially_discardable(possible, board)
    if play    and pointed: return PLAY
    if discard and pointed: return DISCARD
    return None


HINT_VALUE = 0.5


def pretend(action, knowledge, intentions, hand, board):
    (type_, value) = action
    positive = []
    haspositive = False
    change = False
    if type_ == HINT_COLOR:
        newknowledge = []
        for i, (col, num) in enumerate(hand):
            positive.append(value == col)
            newknowledge.append(hint_color(knowledge[i], value, value == col))
            if value == col:
                haspositive = True
                if newknowledge[-1] != knowledge[i]:
                    change = True
    else:
        newknowledge = []
        for i, (col, num) in enumerate(hand):
            positive.append(value == num)
            newknowledge.append(hint_rank(knowledge[i], value, value == num))
            if value == num:
                haspositive = True
                if newknowledge[-1] != knowledge[i]:
                    change = True
    if not haspositive: return False, 0, ["Invalid hint"]
    if not change:      return False, 0, ["No new information"]

    score = 0
    predictions = []
    pos = False
    for i, c, k, p in zip(intentions, hand, newknowledge, positive):
        action_result = whattodo(k, p, board)
        if action_result == PLAY and i != PLAY:
            return False, 0, predictions + [PLAY]
        if action_result == DISCARD and i not in [DISCARD, CANDISCARD]:
            return False, 0, predictions + [DISCARD]
        if action_result == PLAY and i == PLAY:
            pos = True; predictions.append(PLAY); score += 3
        elif action_result == DISCARD and i in [DISCARD, CANDISCARD]:
            pos = True; predictions.append(DISCARD)
            score += 2 if i == DISCARD else 1
        else:
            predictions.append(None)
    if not pos: return False, score, predictions
    return True, score, predictions


def pretend_discard(act, knowledge, board, trash):
    which = copy.deepcopy(knowledge[act.cnr])
    for (col, num) in trash:
        if which[col][num - 1]: which[col][num - 1] -= 1
    for col in ALL_COLORS:
        for i in range(board[col][1]):
            if which[col][i]: which[col][i] -= 1
    possibilities = sum(map(sum, which))
    if possibilities == 0: return (act, 0, [])
    expected = 0
    terms = []
    for col in ALL_COLORS:
        for i, cnt in enumerate(which[col]):
            rank = i + 1
            if cnt > 0:
                prob = cnt / possibilities
                if board[col][1] >= rank:
                    expected += prob * HINT_VALUE
                    terms.append((col, rank, cnt, prob, prob * HINT_VALUE))
                else:
                    dist = rank - board[col][1]
                    value = prob * (6 - rank) / (dist * dist)
                    if rank == 5: value += prob * HINT_VALUE
                    expected -= value
                    terms.append((col, rank, cnt, prob, -value))
    return (act, expected, terms)


def _decrement_card_from_knowledge(knowledge, card):
    result = copy.deepcopy(knowledge)
    col, num = card
    for k in result:
        if k[col][num - 1] > 0:
            k[col][num - 1] -= 1
    return result


def do_sample(knowledge):
    if not knowledge:
        return []
    possible = []
    for col in ALL_COLORS:
        for i, c in enumerate(knowledge[0][col]):
            for _ in range(c):
                possible.append((col, i + 1))
    if not possible:
        return None
    random.shuffle(possible)
    for chosen in possible:
        remaining_knowledge = _decrement_card_from_knowledge(knowledge[1:], chosen)
        other = do_sample(remaining_knowledge)
        if other is not None:
            return [chosen] + other
    return None


def sample_hand(knowledge, max_tries=2000):
    """
    Sample a consistent hand from the given knowledge.
    Returns an empty list if no valid sample is found within max_tries,
    instead of looping forever when the knowledge state is infeasible.
    """
    for _ in range(max_tries):
        result = do_sample(knowledge)
        if result is not None:
            return result
    return []
