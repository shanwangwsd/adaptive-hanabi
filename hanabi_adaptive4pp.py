import random
import sys
import copy
import os
import csv
import numpy as np
from scipy.stats import t as student_t
from dataclasses import dataclass

# ─────────────────────────────────────────────
#  基础常量
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
#  牌堆 / 知识结构
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
#  Action 类型
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
#  知识推断辅助函数
# ─────────────────────────────────────────────
def get_possible(knowledge):
    result = []
    for col in ALL_COLORS:
        for i, cnt in enumerate(knowledge[col]):
            if cnt > 0:
                result.append((col, i + 1))
    return result

def playable(possible, board):
    return all(board[col][1] + 1 == nr for col, nr in possible)

def potentially_playable(possible, board):
    return any(board[col][1] + 1 == nr for col, nr in possible)

def discardable(possible, board):
    return all(board[col][1] >= nr for col, nr in possible)

def potentially_discardable(possible, board):
    return any(board[col][1] >= nr for col, nr in possible)

def update_knowledge(knowledge, used):
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

# ─────────────────────────────────────────────
#  whattodo / pretend（原版保留，IntentionalPlayer 依赖）
# ─────────────────────────────────────────────
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

# ═══════════════════════════════════════════════════════════════════════
#
#  PART 1: Continuous teammate styles and realistic teammate policy
#
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GivingStyle:
    """Continuous giving style: how this player tends to give hints."""
    play_bias: float
    color_bias: float
    conservatism: float
    coverage_preference: float

@dataclass
class ReceivingStyle:
    trust_level: float
    playable_interpretation: float
    safe_interpretation: float
    uncertainty_tolerance: float



@dataclass
class TeammateStylePair:
    giving: GivingStyle
    receiving: ReceivingStyle


def sample_teammate_style(seed=None):
    rng = np.random.default_rng(seed)

    giving = GivingStyle(
        play_bias=float(rng.beta(2.0, 2.0)),
        color_bias=float(rng.beta(3.0, 3.0)),
        conservatism=float(1.0 + 7.0 * rng.beta(2.0, 2.0)),
        coverage_preference=float(rng.beta(1.5, 1.5)),
    )
    receiving = ReceivingStyle(
        trust_level=float(rng.beta(2.0, 1.5)),
        playable_interpretation=float(rng.beta(2.0, 2.0)),
        safe_interpretation=float(rng.beta(1.5, 2.0)),
        uncertainty_tolerance=float(rng.beta(2.0, 2.0)),
    )

    return TeammateStylePair(giving=giving, receiving=receiving)


# ─────────────────────────────────────────────
#  Hand-crafted teammate style sets for experiments
# ─────────────────────────────────────────────

STYLE_PLAY_DIRECTOR = TeammateStylePair(
    giving=GivingStyle(
        play_bias=0.95,
        color_bias=0.50,
        conservatism=4.0,
        coverage_preference=0.20,
    ),
    receiving=ReceivingStyle(
        trust_level=0.80,
        playable_interpretation=0.80,
        safe_interpretation=0.20,
        uncertainty_tolerance=0.40,
    ),
)

STYLE_SAFETY_MANAGER = TeammateStylePair(
    giving=GivingStyle(
        play_bias=0.10,
        color_bias=0.50,
        conservatism=6.5,
        coverage_preference=0.60,
    ),
    receiving=ReceivingStyle(
        trust_level=0.65,
        playable_interpretation=0.75,
        safe_interpretation=0.35,
        uncertainty_tolerance=0.30,
    ),
)

STYLE_COLOR_SPEAKER = TeammateStylePair(
    giving=GivingStyle(
        play_bias=0.60,
        color_bias=0.95,
        conservatism=4.5,
        coverage_preference=0.50,
    ),
    receiving=ReceivingStyle(
        trust_level=0.70,
        playable_interpretation=0.60,
        safe_interpretation=0.40,
        uncertainty_tolerance=0.50,
    ),
)

STYLE_RANK_SPEAKER = TeammateStylePair(
    giving=GivingStyle(
        play_bias=0.60,
        color_bias=0.05,
        conservatism=4.5,
        coverage_preference=0.50,
    ),
    receiving=ReceivingStyle(
        trust_level=0.70,
        playable_interpretation=0.60,
        safe_interpretation=0.40,
        uncertainty_tolerance=0.50,
    ),
)

STYLE_BROADCASTER = TeammateStylePair(
    giving=GivingStyle(
        play_bias=0.55,
        color_bias=0.50,
        conservatism=4.5,
        coverage_preference=0.95,
    ),
    receiving=ReceivingStyle(
        trust_level=0.60,
        playable_interpretation=0.55,
        safe_interpretation=0.45,
        uncertainty_tolerance=0.80,
    ),
)

STYLE_PRECISION_TEACHER = TeammateStylePair(
    giving=GivingStyle(
        play_bias=0.55,
        color_bias=0.50,
        conservatism=4.5,
        coverage_preference=0.05,
    ),
    receiving=ReceivingStyle(
        trust_level=0.80,
        playable_interpretation=0.65,
        safe_interpretation=0.35,
        uncertainty_tolerance=0.20,
    ),
)


EXPERIMENT_STYLE_SETS = {
    "SET_A_PLAY_DIRECTOR": STYLE_PLAY_DIRECTOR,
    "SET_B_SAFETY_MANAGER": STYLE_SAFETY_MANAGER,
    "SET_C_COLOR_SPEAKER": STYLE_COLOR_SPEAKER,
    "SET_D_RANK_SPEAKER": STYLE_RANK_SPEAKER,
    "SET_E_BROADCASTER": STYLE_BROADCASTER,
    "SET_F_PRECISION_TEACHER": STYLE_PRECISION_TEACHER,
}


def get_experiment_style(style_name):
    if style_name not in EXPERIMENT_STYLE_SETS:
        valid = ", ".join(EXPERIMENT_STYLE_SETS.keys())
        raise ValueError(f"Unknown style_name={style_name}. Valid styles: {valid}")
    return copy.deepcopy(EXPERIMENT_STYLE_SETS[style_name])


def make_teammate_style_assignment(n_players, mode="realistic", style_name="SET_A_PLAY_DIRECTOR", seed=None, adaptive_pnr=0, manual_ratio=0.5):
    """
    mode="manual":
        Every realistic teammate is randomly selected from the six predefined SETs.

    mode="realistic":
        Every realistic teammate is generated by sample_teammate_style(seed).
        This is the old continuous realistic-style family, but seed-controllable.

    mode="hybrid":
        Some teammates are randomly selected from the predefined SETs, and the
        others are generated by sample_teammate_style(seed).

    mode="none":
        Return None and let RealisticStylePlayer sample by itself exactly as before.

    Backward-compatible aliases:
        mode="random" -> "realistic"
        mode="fixed"  -> all teammates use style_name
        mode="mixed"  -> "manual"
    """
    if mode is None:
        mode = "realistic"
    mode = str(mode).lower()

    if mode == "random":
        mode = "realistic"
    if mode == "mixed":
        mode = "manual"

    if mode == "none":
        return None

    rng = np.random.default_rng(seed)
    assignment = {}
    style_names = list(EXPERIMENT_STYLE_SETS.keys())

    if mode == "realistic":
        for pnr in range(n_players):
            if pnr == adaptive_pnr:
                continue
            style_seed = None if seed is None else int(seed + 1009 * pnr)
            assignment[pnr] = sample_teammate_style(style_seed)
        return assignment

    if mode == "manual":
        for pnr in range(n_players):
            if pnr == adaptive_pnr:
                continue
            chosen_name = str(rng.choice(style_names))
            assignment[pnr] = get_experiment_style(chosen_name)
        return assignment

    if mode == "hybrid":
        manual_ratio = max(0.0, min(1.0, float(manual_ratio)))
        for pnr in range(n_players):
            if pnr == adaptive_pnr:
                continue
            use_manual = bool(rng.random() < manual_ratio)
            if use_manual:
                chosen_name = str(rng.choice(style_names))
                assignment[pnr] = get_experiment_style(chosen_name)
            else:
                style_seed = None if seed is None else int(seed + 1009 * pnr)
                assignment[pnr] = sample_teammate_style(style_seed)
        return assignment

    if mode == "fixed":
        style = get_experiment_style(style_name)
        for pnr in range(n_players):
            if pnr == adaptive_pnr:
                continue
            assignment[pnr] = copy.deepcopy(style)
        return assignment

    raise ValueError("Unknown teammate style mode. Use: manual, realistic, hybrid, none, or fixed")


def make_realistic_teammate(name, pnr, style_assignment=None):
    style_pair = None
    if style_assignment is not None and pnr in style_assignment:
        style_pair = style_assignment[pnr]
    return RealisticStylePlayer(name, pnr, style_pair=style_pair)


def print_style_assignment(style_assignment):
    if style_assignment is None:
        print("Teammate style assignment: none / old default")
        return
    print("Teammate style assignment:")
    for pnr in sorted(style_assignment.keys()):
        print(f"  P{pnr}: {format_style_pair(style_assignment[pnr])}")


def format_style_pair(pair):
    g = pair.giving
    r = pair.receiving
    return (
        f"G(play={g.play_bias:.2f}, color={g.color_bias:.2f}, "
        f"cons={g.conservatism:.1f}, cov={g.coverage_preference:.2f}); "
        f"R(trust={r.trust_level:.2f}, play={r.playable_interpretation:.2f}, "
        f"safe={r.safe_interpretation:.2f}, tol={r.uncertainty_tolerance:.2f})"
    )




# Global teammate-style configuration.
# Default is "none", which preserves the old behavior: RealisticStylePlayer samples
# its own style when no style_pair is explicitly passed.
TEAMMATE_STYLE_MODE = "none"
TEAMMATE_STYLE_NAME = "SET_A_PLAY_DIRECTOR"
TEAMMATE_STYLE_SEED = None
TEAMMATE_ADAPTIVE_PNR = 0
TEAMMATE_MANUAL_RATIO = 0.5
TEAMMATE_STYLE_ASSIGNMENT = None


def configure_teammate_styles(mode="none", style_name="SET_A_PLAY_DIRECTOR", seed=None, adaptive_pnr=0, manual_ratio=0.5):
    """
    Configure how RealisticStylePlayer chooses its style when style_pair is not
    explicitly provided.

    mode="none":      old default behavior; every RealisticStylePlayer samples itself.
    mode="realistic": all teammates are generated by sample_teammate_style(seed).
    mode="manual":    all teammates are randomly selected from SET_A ... SET_F.
    mode="hybrid":    each teammate is manual with probability manual_ratio,
                       otherwise realistic.
    mode="fixed":     all teammates use one selected predefined style set.

    Backward-compatible aliases:
        mode="random" -> "realistic"
        mode="mixed"  -> "manual"
    """
    global TEAMMATE_STYLE_MODE
    global TEAMMATE_STYLE_NAME
    global TEAMMATE_STYLE_SEED
    global TEAMMATE_ADAPTIVE_PNR
    global TEAMMATE_MANUAL_RATIO
    global TEAMMATE_STYLE_ASSIGNMENT

    mode = str(mode).lower()
    if mode == "random":
        mode = "realistic"
    if mode == "mixed":
        mode = "manual"

    if mode not in ["none", "realistic", "manual", "hybrid", "fixed"]:
        raise ValueError("Unknown teammate style mode. Use: none, realistic, manual, hybrid, or fixed")
    if style_name not in EXPERIMENT_STYLE_SETS:
        valid = ", ".join(EXPERIMENT_STYLE_SETS.keys())
        raise ValueError(f"Unknown style_name={style_name}. Valid styles: {valid}")

    TEAMMATE_STYLE_MODE = mode
    TEAMMATE_STYLE_NAME = style_name
    TEAMMATE_STYLE_SEED = seed
    TEAMMATE_ADAPTIVE_PNR = adaptive_pnr
    TEAMMATE_MANUAL_RATIO = max(0.0, min(1.0, float(manual_ratio)))
    TEAMMATE_STYLE_ASSIGNMENT = None
    if mode != "none":
        TEAMMATE_STYLE_ASSIGNMENT = make_teammate_style_assignment(
            n_players=4,
            mode=mode,
            style_name=style_name,
            seed=seed,
            adaptive_pnr=adaptive_pnr,
            manual_ratio=TEAMMATE_MANUAL_RATIO,
        )

    print(
        f"Configured teammate styles: mode={mode}, style_name={style_name}, "
        f"seed={seed}, adaptive_pnr={adaptive_pnr}, manual_ratio={TEAMMATE_MANUAL_RATIO}"
    )


def style_pair_from_global_config(pnr):
    """Return a style_pair for RealisticStylePlayer according to global config."""
    if TEAMMATE_STYLE_MODE == "none":
        return None

    if pnr == TEAMMATE_ADAPTIVE_PNR:
        return None

    if TEAMMATE_STYLE_ASSIGNMENT is not None and pnr in TEAMMATE_STYLE_ASSIGNMENT:
        return copy.deepcopy(TEAMMATE_STYLE_ASSIGNMENT[pnr])

    # Fallback for experiments with more than four players. This keeps behavior
    # defined even if configure_teammate_styles() created the default 4-player
    # assignment but a larger game is later constructed.
    assignment = make_teammate_style_assignment(
        n_players=pnr + 1,
        mode=TEAMMATE_STYLE_MODE,
        style_name=TEAMMATE_STYLE_NAME,
        seed=TEAMMATE_STYLE_SEED,
        adaptive_pnr=TEAMMATE_ADAPTIVE_PNR,
        manual_ratio=TEAMMATE_MANUAL_RATIO,
    )
    if assignment is not None and pnr in assignment:
        return copy.deepcopy(assignment[pnr])
    return None


def style_pair_from_mode_seed(pnr, seed=None):
    """
    Build one RealisticStylePlayer style directly from the current style-mode
    and an explicit player/team seed.

    This is used by configured(...) teammates in experiment teamsets, so
    --style-mode manual/realistic/hybrid actually affects compare, baselines,
    ablation, curve, and adhoc experiments.
    """
    mode = TEAMMATE_STYLE_MODE
    if mode == "none":
        return None

    base_seed = 0 if TEAMMATE_STYLE_SEED is None else int(TEAMMATE_STYLE_SEED)
    local_seed = 0 if seed is None else int(seed)
    rng_seed = base_seed + local_seed + 1009 * int(pnr)
    rng = np.random.default_rng(rng_seed)
    style_names = list(EXPERIMENT_STYLE_SETS.keys())

    if mode == "realistic":
        return sample_teammate_style(rng_seed)

    if mode == "manual":
        chosen_name = str(rng.choice(style_names))
        return get_experiment_style(chosen_name)

    if mode == "hybrid":
        use_manual = bool(rng.random() < TEAMMATE_MANUAL_RATIO)
        if use_manual:
            chosen_name = str(rng.choice(style_names))
            return get_experiment_style(chosen_name)
        return sample_teammate_style(rng_seed)

    if mode == "fixed":
        return get_experiment_style(TEAMMATE_STYLE_NAME)

    return None

def configure_teammate_styles_from_argv(argv=None):
    if argv is None:
        argv = sys.argv

    mode = None
    style_name = "SET_A_PLAY_DIRECTOR"
    seed = None
    adaptive_pnr = 0
    manual_ratio = 0.5

    cleaned = [argv[0]]

    def require_value(index, flag):
        if index + 1 >= len(argv):
            raise ValueError(f"{flag} requires a value")
        return argv[index + 1]

    i = 1
    while i < len(argv):
        arg = argv[i]

        if arg == "--style-mode":
            mode = require_value(i, arg)
            i += 2
            continue

        if arg == "--style-name":
            style_name = require_value(i, arg)
            i += 2
            continue

        if arg == "--style-seed":
            seed = int(require_value(i, arg))
            i += 2
            continue

        if arg == "--adaptive-pnr":
            adaptive_pnr = int(require_value(i, arg))
            i += 2
            continue

        if arg == "--manual-ratio":
            manual_ratio = float(require_value(i, arg))
            i += 2
            continue

        cleaned.append(arg)
        i += 1

    if mode is not None:
        configure_teammate_styles(
            mode=mode,
            style_name=style_name,
            seed=seed,
            adaptive_pnr=adaptive_pnr,
            manual_ratio=manual_ratio,
        )

    sys.argv = cleaned
    return cleaned[1:]


class RealisticStylePlayer:
    """
    Continuous teammate model.

    GivingStyle controls how this player gives hints.
    ReceivingStyle controls how this player reacts to hints.
    These two dimensions are independent.
    """
    def __init__(self, name, pnr, style_pair=None):
        self.name = name
        self.pnr = pnr
        if style_pair is None:
            style_pair = style_pair_from_global_config(pnr)
        self.style_pair = style_pair if style_pair is not None else sample_teammate_style()
        self.giving = self.style_pair.giving
        self.receiving = self.style_pair.receiving
        self.gothint = None
        self.explanation = []

    def get_action(self, nr, hands, knowledge, trash, played, board,
                   valid_actions, hints):
        handsize = len(knowledge[nr])
        possible = [get_possible(k) for k in knowledge[nr]]
        result = None

        # 1. Receiving style: interpret the latest hint received by this player.
        if self.gothint:
            hint_act, _ = self.gothint

            for i, k in enumerate(knowledge[nr]):
                if hint_act.type == HINT_COLOR:
                    pointed = sum(k[hint_act.col]) > 0
                elif hint_act.type == HINT_NUMBER:
                    pointed = sum(k[c][hint_act.num - 1] for c in ALL_COLORS) > 0
                else:
                    pointed = False

                if not pointed:
                    continue

                poss = get_possible(k)
                can_play = potentially_playable(poss, board)
                can_discard = potentially_discardable(poss, board)

                uncertainty = min(1.0, len(poss) / 25.0)

                ambiguity_excess = max(
                    0.0,
                    uncertainty - self.receiving.uncertainty_tolerance
                )

                confidence = self.receiving.trust_level * (1.0 - ambiguity_excess)
                confidence = max(0.0, min(1.0, confidence))

                if can_play and confidence >= self.receiving.playable_interpretation:
                    result = Action(PLAY, cnr=i)
                    break

                if can_discard and confidence >= self.receiving.safe_interpretation:
                    result = Action(DISCARD, cnr=i)
                    break

            self.gothint = None

        # 2. Deterministic play/discard from current knowledge.
        for i, p in enumerate(possible):
            if playable(p, board) and not result:
                result = Action(PLAY, cnr=i)

        discards = [i for i, p in enumerate(possible) if discardable(p, board)]

        hint_threshold = max(1, min(8, int(round(self.giving.conservatism))))

        if discards and hints < hint_threshold and not result:
            result = Action(DISCARD, cnr=random.choice(discards))

        # 3. Giving style: choose a hint according to continuous giving parameters.
        if hints >= hint_threshold and not result:
            best_hint = self._choose_hint(nr, hands, knowledge, board)
            if best_hint is not None:
                result = best_hint

        if result:
            return result

        if discards:
            return Action(DISCARD, cnr=random.choice(discards))

        scores = [
            pretend_discard(Action(DISCARD, cnr=i), knowledge[nr], board, trash)
            for i in range(handsize)
        ]
        scores.sort(key=lambda x: -x[1])
        return scores[0][0]

    def _choose_hint(self, nr, hands, knowledge, board):
        best_score = -1e9
        best_action = None
        g = self.giving

        for target in range(len(hands)):
            if target == nr or not hands[target]:
                continue
            if target >= len(knowledge) or not knowledge[target]:
                continue

            intentions = []
            for col, num in hands[target]:
                if board[col][1] + 1 == num:
                    intentions.append(PLAY)
                elif board[col][1] >= num:
                    intentions.append(DISCARD)
                else:
                    intentions.append(CANDISCARD)

            candidates = []
            for c in set(col for col, _ in hands[target]):
                candidates.append((HINT_COLOR, c))
            for r in set(num for _, num in hands[target]):
                candidates.append((HINT_NUMBER, r))

            for cand in candidates:
                try:
                    ok, safe_score, _ = pretend(
                        cand, knowledge[target], intentions, hands[target], board
                    )
                except Exception:
                    continue

                if not ok:
                    continue

                type_, value = cand
                touched = 0
                play_hits = 0
                discard_hits = 0

                for col, num in hands[target]:
                    if type_ == HINT_COLOR:
                        pointed = value == col
                    else:
                        pointed = value == num

                    if not pointed:
                        continue

                    touched += 1

                    if board[col][1] + 1 == num:
                        play_hits += 1
                    elif board[col][1] >= num:
                        discard_hits += 1

                channel_score = g.color_bias if type_ == HINT_COLOR else (1.0 - g.color_bias)
                intent_score = g.play_bias * play_hits + (1.0 - g.play_bias) * discard_hits
                max_hand = max(1, len(hands[target]))
                coverage_ratio = touched / max_hand

                # coverage_preference 高：喜欢广覆盖
                # coverage_preference 低：喜欢精确 hint
                coverage_score = g.coverage_preference * coverage_ratio
                precision_score = (1.0 - g.coverage_preference) * (1.0 - coverage_ratio)


                score = (
                    safe_score
                    + 2.40 * intent_score
                    + channel_score
                    + coverage_score
                    + precision_score
                )

                if score > best_score:
                    best_score = score
                    if type_ == HINT_COLOR:
                        best_action = Action(HINT_COLOR, pnr=target, col=value)
                    else:
                        best_action = Action(HINT_NUMBER, pnr=target, num=value)

        return best_action

    def inform(self, action, player, public_state):
        if action.type in [HINT_COLOR, HINT_NUMBER] and action.pnr == self.pnr:
            self.gothint = (copy.deepcopy(action), player)

    def get_explanation(self):
        return self.explanation

class TeammateGivingFeatureModel:
    """
    Bayesian continuous giving-style estimator.

    Latent giving traits are represented as posterior distributions:
      play_bias            ~ Beta(alpha_play, beta_play)
      color_bias           ~ Beta(alpha_color, beta_color)
      coverage_preference  ~ Beta(alpha_coverage, beta_coverage)
      conservatism         ~ Categorical over hint-token levels {1,...,8}
                             with a Dirichlet posterior.
    """
    def __init__(self, player_id):
        self.player_id = player_id
        self.n = 0

        # Uniform Beta priors over continuous traits in [0,1].
        self.alpha_play = 1.0
        self.beta_play = 1.0
        self.alpha_color = 1.0
        self.beta_color = 1.0
        self.alpha_coverage = 1.0
        self.beta_coverage = 1.0

        # Uniform Dirichlet prior over hint-token levels 1..8.
        self.conservatism_counts = np.ones(8, dtype=float)

        # Point estimates learned by inverse choice modeling.
        # The Beta counters above are kept as a weak/fallback diagnostic, but the
        # main estimate should come from how well a style explains the chosen hint
        # against the other hints that were available in that exact state.
        self.point_play = 0.5
        self.point_color = 0.5
        self.point_coverage = 0.5
        self.point_conservatism = 4.5
        self.choice_updates = 0

    def _clip01(self, x):
        return max(0.0, min(1.0, float(x)))

    def _beta_mean(self, alpha, beta):
        return alpha / (alpha + beta)

    def _beta_variance(self, alpha, beta):
        total = alpha + beta
        return (alpha * beta) / ((total ** 2) * (total + 1.0))

    def _conservatism_mean(self):
        levels = np.arange(1, 9, dtype=float)
        probs = self.conservatism_counts / np.sum(self.conservatism_counts)
        return float(np.dot(levels, probs))

    def update_from_hint(self, action, target_hand, board, hints_before_action,
                         weight=1.0, evidence_override=None):
        """
        Update the giving-style posterior from one observed hint.

        `weight` is an evidence strength. Use weight < 1 for noisy evidence,
        especially self-targeted hints whose target hand is sampled rather than
        directly observed.

        `evidence_override` can provide averaged Monte Carlo evidence with keys:
          play_evidence, color_evidence, coverage_evidence, conservatism_level
        """
        if action.type not in [HINT_COLOR, HINT_NUMBER]:
            return

        weight = max(0.0, float(weight))
        if weight <= 0.0:
            return

        if evidence_override is not None:
            play_evidence = self._clip01(evidence_override.get("play_evidence", 0.5))
            color_evidence = self._clip01(evidence_override.get("color_evidence", 1.0 if action.type == HINT_COLOR else 0.0))
            coverage_evidence = self._clip01(evidence_override.get("coverage_evidence", 0.5))
            level = int(round(max(1.0, min(8.0, float(evidence_override.get("conservatism_level", hints_before_action))))))
        else:
            if not target_hand:
                return

            pointed_cards = []
            for col, num in target_hand:
                if action.type == HINT_COLOR:
                    pointed = action.col == col
                else:
                    pointed = action.num == num
                if pointed:
                    pointed_cards.append((col, num))

            if not pointed_cards:
                return

            play_hits = sum(
                1 for col, num in pointed_cards
                if board[col][1] + 1 == num
            )
            total_pointed = max(1, len(pointed_cards))

            play_evidence = self._clip01(play_hits / total_pointed)
            color_evidence = 1.0 if action.type == HINT_COLOR else 0.0
            coverage_evidence = self._clip01(
                len(pointed_cards) / max(1, len(target_hand))
            )
            level = int(round(max(1.0, min(8.0, float(hints_before_action)))))

        self.alpha_play += weight * play_evidence
        self.beta_play += weight * (1.0 - play_evidence)

        self.alpha_color += weight * color_evidence
        self.beta_color += weight * (1.0 - color_evidence)

        self.alpha_coverage += weight * coverage_evidence
        self.beta_coverage += weight * (1.0 - coverage_evidence)

        self.conservatism_counts[level - 1] += weight

        self.n += weight

    def estimate(self):
        if self.choice_updates > 0:
            return GivingStyle(
                play_bias=float(self.point_play),
                color_bias=float(self.point_color),
                conservatism=float(self.point_conservatism),
                coverage_preference=float(self.point_coverage),
            )
        return GivingStyle(
            play_bias=self._beta_mean(self.alpha_play, self.beta_play),
            color_bias=self._beta_mean(self.alpha_color, self.beta_color),
            conservatism=self._conservatism_mean(),
            coverage_preference=self._beta_mean(
                self.alpha_coverage,
                self.beta_coverage,
            ),
        )

    def _action_same_hint(self, a, b):
        if a is None or b is None:
            return False
        return (a.type, a.pnr, a.col, a.num) == (b.type, b.pnr, b.col, b.num)

    def _candidate_features(self, action, target_hand, board):
        touched = 0
        play_hits = 0
        discard_hits = 0
        for col, num in target_hand:
            if action.type == HINT_COLOR:
                pointed = action.col == col
            elif action.type == HINT_NUMBER:
                pointed = action.num == num
            else:
                pointed = False
            if not pointed:
                continue
            touched += 1
            if board[col][1] + 1 == num:
                play_hits += 1
            elif board[col][1] >= num:
                discard_hits += 1

        hand_size = max(1, len(target_hand))
        coverage_ratio = touched / hand_size
        return {
            "play_hits": float(play_hits),
            "discard_hits": float(discard_hits),
            "play_minus_discard": (play_hits - discard_hits) / hand_size,
            "is_color": 1.0 if action.type == HINT_COLOR else 0.0,
            "is_color_signed": 1.0 if action.type == HINT_COLOR else -1.0,
            "coverage_signed": 2.0 * coverage_ratio - 1.0,
            "coverage_ratio": coverage_ratio,
        }
    # _style_score_from_features method removed

    # _refit_from_choice_history method removed

    def _style_score(self, action, safe_score, target_hand, board, style):
        touched = 0
        play_hits = 0
        discard_hits = 0
        for col, num in target_hand:
            if action.type == HINT_COLOR:
                pointed = action.col == col
            elif action.type == HINT_NUMBER:
                pointed = action.num == num
            else:
                pointed = False
            if not pointed:
                continue
            touched += 1
            if board[col][1] + 1 == num:
                play_hits += 1
            elif board[col][1] >= num:
                discard_hits += 1

        channel_score = style.color_bias if action.type == HINT_COLOR else (1.0 - style.color_bias)
        intent_score = style.play_bias * play_hits + (1.0 - style.play_bias) * discard_hits
        coverage_ratio = touched / max(1, len(target_hand))
        coverage_score = style.coverage_preference * coverage_ratio
        precision_score = (1.0 - style.coverage_preference) * (1.0 - coverage_ratio)
        return safe_score + 2.40 * intent_score + channel_score + coverage_score + precision_score

    def _enumerate_safe_hint_candidates(self, actor_id, hands, knowledge, board):
        candidates = []
        for target in range(len(hands)):
            if target == actor_id or not hands[target]:
                continue
            if target >= len(knowledge) or not knowledge[target]:
                continue

            intentions = []
            for col, num in hands[target]:
                if board[col][1] + 1 == num:
                    intentions.append(PLAY)
                elif board[col][1] >= num:
                    intentions.append(DISCARD)
                else:
                    intentions.append(CANDISCARD)

            raw_candidates = []
            for c in set(col for col, _ in hands[target]):
                raw_candidates.append((HINT_COLOR, c))
            for r in set(num for _, num in hands[target]):
                raw_candidates.append((HINT_NUMBER, r))

            for type_, value in raw_candidates:
                try:
                    ok, safe_score, _ = pretend(
                        (type_, value),
                        knowledge[target],
                        intentions,
                        hands[target],
                        board,
                    )
                except Exception:
                    continue
                if not ok:
                    continue
                if type_ == HINT_COLOR:
                    act = Action(HINT_COLOR, pnr=target, col=value)
                else:
                    act = Action(HINT_NUMBER, pnr=target, num=value)
                candidates.append((act, float(safe_score), list(hands[target])))
        return candidates

    def update_from_choice(self, action, actor_id, hands, knowledge, board, hints_before_action,
                           lr=0.06):
        """
        Inverse choice update for GivingStyle.

        This compares the observed chosen hint with the hint predicted by the
        current giving-style estimate among all safe candidate hints available in
        the same state. This is the first lightweight inverse-choice version:
        no history fitting, no absolute anchors, no decay, no extra stabilizers.
        """
        if action.type not in [HINT_COLOR, HINT_NUMBER]:
            return

        candidates = self._enumerate_safe_hint_candidates(actor_id, hands, knowledge, board)
        if not candidates:
            return

        chosen = None
        for cand in candidates:
            if self._action_same_hint(cand[0], action):
                chosen = cand
                break
        if chosen is None:
            return

        style = GivingStyle(
            play_bias=float(self.point_play),
            color_bias=float(self.point_color),
            conservatism=float(self.point_conservatism),
            coverage_preference=float(self.point_coverage),
        )

        predicted = max(
            candidates,
            key=lambda x: self._style_score(x[0], x[1], x[2], board, style)
        )

        chosen_feat = self._candidate_features(chosen[0], chosen[2], board)
        pred_feat = self._candidate_features(predicted[0], predicted[2], board)

        # Perceptron-style inverse planning update.
        # Keep the same first inverse-choice logic, but use a decaying learning
        # rate so beliefs learn quickly early and then stop jumping forever.
        # This is intentionally minimal: no history fitting, no anchors, no meta
        # corrections, only lr decay for convergence stability.
        lr_eff = lr / np.sqrt(1.0 + self.choice_updates / 120.0)
        self.point_play += lr_eff * 2.40 * (chosen_feat["play_minus_discard"] - pred_feat["play_minus_discard"])
        self.point_color += lr_eff * (chosen_feat["is_color_signed"] - pred_feat["is_color_signed"])
        self.point_coverage += lr_eff * (chosen_feat["coverage_signed"] - pred_feat["coverage_signed"])

        # A hint given at a certain token count is weak evidence for the giving
        # threshold, but do not let this dominate the other traits.
        observed_level = max(1.0, min(8.0, float(hints_before_action)))
        self.point_conservatism += 0.02 * (observed_level - self.point_conservatism)

        self.point_play = self._clip01(self.point_play)
        self.point_color = self._clip01(self.point_color)
        self.point_coverage = self._clip01(self.point_coverage)
        self.point_play = max(0.02, min(0.98, self.point_play))
        self.point_color = max(0.02, min(0.98, self.point_color))
        self.point_coverage = max(0.02, min(0.98, self.point_coverage))
        self.point_conservatism = max(1.0, min(8.0, self.point_conservatism))

        self.choice_updates += 1
        self.n += 1.0

    def posterior_uncertainty(self):
        variances = [
            self._beta_variance(self.alpha_play, self.beta_play),
            self._beta_variance(self.alpha_color, self.beta_color),
            self._beta_variance(self.alpha_coverage, self.beta_coverage),
        ]
        return float(np.mean(variances))

    def effective_sample_size(self):
        return self.n

    def __repr__(self):
        est = self.estimate()
        return (
            f"BayesianGivingModel(player={self.player_id}, obs={self.n}, "
            f"play_mean={est.play_bias:.2f}, color_mean={est.color_bias:.2f}, "
            f"cons_mean={est.conservatism:.2f}, coverage_mean={est.coverage_preference:.2f}, "
            f"unc={self.posterior_uncertainty():.4f})"
        )

class TeammateResponseModel:
    """
    Bayesian response-style model.

    For each hint channel h in {number, color}:
        pi_h ~ Dirichlet(alpha_h)
        response | h ~ Categorical(pi_h)

    The adaptive player uses the Dirichlet posterior predictive mean.
    """
    RESPONSE_TYPES = [PLAY, DISCARD, None]

    def __init__(self, player_id):
        self.player_id = player_id
        self.alpha = {
            HINT_NUMBER: {PLAY: 1.0, DISCARD: 1.0, None: 1.0},
            HINT_COLOR:  {PLAY: 1.0, DISCARD: 1.0, None: 1.0},
        }
        self.observation_count = 0
        
    @property
    def count(self):
        """Backward-compatible alias used by old diagnostics."""
        return self.observation_count

    def update_from_response(self, hint_action, response_action, hinted_hand, board_at_hint):
        if hint_action.type not in [HINT_COLOR, HINT_NUMBER]:
            return
        if response_action.type not in [PLAY, DISCARD]:
            self.update_no_response(hint_action)
            return
        if response_action.cnr is None or response_action.cnr >= len(hinted_hand):
            self.update_no_response(hint_action)
            return

        card = hinted_hand[response_action.cnr]
        col, num = card
        pointed = False
        if hint_action.type == HINT_COLOR and hint_action.col == col:
            pointed = True
        if hint_action.type == HINT_NUMBER and hint_action.num == num:
            pointed = True

        if not pointed:
            self.alpha[hint_action.type][None] += 1.0
            self.observation_count += 1
            return

        if response_action.type == PLAY:
            self.alpha[hint_action.type][PLAY] += 1.0
        elif response_action.type == DISCARD:
            self.alpha[hint_action.type][DISCARD] += 1.0
        self.observation_count += 1

    def update_no_response(self, hint_action):
        if hint_action.type not in [HINT_COLOR, HINT_NUMBER]:
            return
        self.alpha[hint_action.type][None] += 1.0
        self.observation_count += 1

    def prob(self, hint_type, response_type):
        bucket = self.alpha[hint_type]
        total = sum(bucket.values())
        return bucket[response_type] / total

    def posterior_strength(self, hint_type=None):
        if hint_type is not None:
            return float(sum(self.alpha[hint_type].values()))

        total = 0.0
        for _, bucket in self.alpha.items():
            total += float(sum(bucket.values()))
        return total

    def posterior_uncertainty(self, hint_type):
        probs = np.array(
            [self.prob(hint_type, r) for r in self.RESPONSE_TYPES],
            dtype=float,
        )
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        return entropy / np.log(len(self.RESPONSE_TYPES))

    def score_hint(self, hint_action, target_hand, board):
        if hint_action.type not in [HINT_COLOR, HINT_NUMBER]:
            return -1e9

        p_play = self.prob(hint_action.type, PLAY)
        p_discard = self.prob(hint_action.type, DISCARD)
        p_none = self.prob(hint_action.type, None)
        score = 0.0
        touched = 0

        for col, num in target_hand:
            pointed = False
            if hint_action.type == HINT_COLOR and hint_action.col == col:
                pointed = True
            if hint_action.type == HINT_NUMBER and hint_action.num == num:
                pointed = True
            if not pointed:
                continue
            touched += 1

            if board[col][1] + 1 == num:
                score += 5.0 * p_play
                score -= 4.0 * p_discard
                score -= 1.0 * p_none
            elif board[col][1] >= num:
                score += 2.5 * p_discard
                score -= 2.0 * p_play
                score -= 0.5 * p_none
            elif num == 5:
                score -= 3.0 * p_discard
                score -= 0.5 * p_play
            else:
                score += 0.2 * (1.0 - p_none)

        if touched == 0:
            return -1e9
        return score + 0.05 * touched

    def __repr__(self):
        return (
            f"BayesianResponseModel(player={self.player_id}, obs={self.observation_count}, "
            f"P(play|num)={self.prob(HINT_NUMBER, PLAY):.2f}, "
            f"P(discard|num)={self.prob(HINT_NUMBER, DISCARD):.2f}, "
            f"P(none|num)={self.prob(HINT_NUMBER, None):.2f}, "
            f"P(play|color)={self.prob(HINT_COLOR, PLAY):.2f}, "
            f"P(discard|color)={self.prob(HINT_COLOR, DISCARD):.2f}, "
            f"P(none|color)={self.prob(HINT_COLOR, None):.2f})"
        )

# ═══════════════════════════════════════════════════════════════════════
#  MetaUtilityCalibrator: Online meta-calibration for adaptive hint utility
class MetaUtilityCalibrator:
    """
    Online meta-calibration for teammate-model features.

    This is a lightweight meta-learning layer over the Bayesian teammate models.
    It learns not only partner-specific beliefs, but also how much to trust each
    modeling feature when converting those beliefs into hint utility.

    Three mechanisms are implemented:
    1. Uncertainty-driven learning rate: update more when model uncertainty is high.
    2. Feature reliability memory: down-weight features that historically predict poorly.
    3. Context-conditioned adaptation: keep separate utility weights for different
       hint/game contexts instead of using one global weight vector everywhere.
    """
    FEATURE_KEYS = [
        "safe_score",
        "bounded_response",
        "response_conf",
        "channel_match",
        "coverage_match",
        "ambiguity",
        "response_uncertainty",
        "giving_uncertainty",
    ]

    BASE_WEIGHTS = {
        "safe_score": 1.00,
        "bounded_response": 1.35,
        "response_conf": 0.35,
        "channel_match": 0.12,
        "coverage_match": 0.12,
        "ambiguity": -0.85,
        "response_uncertainty": -0.25,
        "giving_uncertainty": -0.10,
    }

    def __init__(self, lr=0.015, l2=0.002, weight_clip=2.5):
        self.lr = float(lr)
        self.l2 = float(l2)
        self.weight_clip = float(weight_clip)

        # Global weights are used as a backoff. Context-specific weights are
        # created lazily and initialized from these global weights.
        self.weights = dict(self.BASE_WEIGHTS)
        self.context_weights = {}

        # Feature reliability is in (0, 1]. A feature that repeatedly contributes
        # to large prediction errors is trusted less in future utility prediction.
        self.feature_reliability = {k: 1.0 for k in self.FEATURE_KEYS}
        self.feature_error_ema = {k: 0.0 for k in self.FEATURE_KEYS}
        self.reliability_decay = 0.97
        self.reliability_floor = 0.20

        self.n_updates = 0
        self.total_abs_error = 0.0
        self.error_history = []

    def _context_key(self, features):
        hint_type = features.get("hint_type", "unknown")
        hint_bucket = features.get("hint_bucket", "mid")
        ambiguity_bucket = features.get("ambiguity_bucket", "amb_mid")
        return (hint_type, hint_bucket, ambiguity_bucket)

    def _weights_for_context(self, context_key):
        if context_key not in self.context_weights:
            self.context_weights[context_key] = dict(self.weights)
        return self.context_weights[context_key]

    def _effective_feature(self, key, features):
        return float(features.get(key, 0.0)) * self.feature_reliability.get(key, 1.0)

    def predict(self, features):
        context_key = self._context_key(features)
        weights = self._weights_for_context(context_key)
        return float(
            sum(weights[k] * self._effective_feature(k, features) for k in self.FEATURE_KEYS)
        )

    def _uncertainty_multiplier(self, features):
        response_unc = float(features.get("response_uncertainty", 0.0))
        giving_unc = float(features.get("giving_uncertainty", 0.0))
        ambiguity = float(features.get("ambiguity", 0.0))
        uncertainty = max(0.0, min(1.0, 0.45 * response_unc + 0.35 * giving_unc + 0.20 * ambiguity))
        return 0.50 + 1.50 * uncertainty

    def _update_feature_reliability(self, features, abs_error):
        # Reliability memory is feature-specific: if a feature was active during a
        # high-error prediction, slightly reduce trust in that feature.
        for key in self.FEATURE_KEYS:
            x = abs(float(features.get(key, 0.0)))
            contribution_error = abs_error * min(1.0, x)
            old = self.feature_error_ema[key]
            new = self.reliability_decay * old + (1.0 - self.reliability_decay) * contribution_error
            self.feature_error_ema[key] = new
            self.feature_reliability[key] = max(
                self.reliability_floor,
                1.0 / (1.0 + new),
            )

    def update(self, features, outcome):
        pred = self.predict(features)
        target = float(max(-3.0, min(5.0, outcome)))
        error = target - pred
        abs_error = abs(error)

        context_key = self._context_key(features)
        weights = self._weights_for_context(context_key)
        lr_eff = self.lr * self._uncertainty_multiplier(features)

        for key in self.FEATURE_KEYS:
            x = self._effective_feature(key, features)
            weights[key] += lr_eff * error * x - self.l2 * weights[key]
            weights[key] = max(-self.weight_clip, min(self.weight_clip, weights[key]))

        # Keep signs of penalty terms interpretable.
        weights["ambiguity"] = min(0.0, weights["ambiguity"])
        weights["response_uncertainty"] = min(0.0, weights["response_uncertainty"])
        weights["giving_uncertainty"] = min(0.0, weights["giving_uncertainty"])

        # Slowly let the global backoff follow the average learned context behavior.
        for key in self.FEATURE_KEYS:
            self.weights[key] = 0.995 * self.weights[key] + 0.005 * weights[key]

        self._update_feature_reliability(features, abs_error)

        self.n_updates += 1
        self.total_abs_error += abs_error
        self.error_history.append(abs_error)
        return error

    def mean_abs_error(self):
        if self.n_updates == 0:
            return 0.0
        return self.total_abs_error / self.n_updates

    def reliability_summary(self):
        return ", ".join(f"{k}={self.feature_reliability[k]:.2f}" for k in self.FEATURE_KEYS)

    def __repr__(self):
        parts = ", ".join(f"{k}={v:+.2f}" for k, v in self.weights.items())
        return (
            f"MetaUtilityCalibrator(updates={self.n_updates}, mae={self.mean_abs_error():.3f}, "
            f"contexts={len(self.context_weights)}, {parts}; reliability: {self.reliability_summary()})"
        )

class FrozenMetaUtilityCalibrator(MetaUtilityCalibrator):
    """
    Same initial utility weights as MetaUtilityCalibrator, but update() does not
    change weights or feature reliability. This is the clean AdaptiveNoMeta ablation.
    """
    def update(self, features, outcome):
        pred = self.predict(features)
        target = float(max(-3.0, min(5.0, outcome)))
        error = target - pred
        abs_error = abs(error)
        self.n_updates += 1
        self.total_abs_error += abs_error
        self.error_history.append(abs_error)
        return error

# ═══════════════════════════════════════════════════════════════════════
#
#  PART 3: 自适应玩家 (AdaptivePlayer)

class AdaptivePlayer:
    """
    Continuous Bounded-Rational Teammate Model for ad-hoc Hanabi coordination.

    The agent maintains three persistent teammate models during repeated play:

    1. Giving model:
       Estimates why a teammate gives hints. It is used for self-recognition and
       for modeling the teammate's public coordination style.

    2. Response model:
       Estimates how a teammate reacts after receiving our hint. It is used for
       posterior-predictive response modeling.

    3. Safety-aware adaptive hint selection:
       Chooses among valid safe hints using posterior-predictive teammate response,
       learned giving-style compatibility, ambiguity, and uncertainty.

    The decision rule is deliberately not framed as RL. It is a Bayesian cognitive
    utility model: candidate hints are evaluated using posterior-predictive teammate
    responses, posterior means of giving-style traits, ambiguity, posterior
    uncertainty, and the original intentional safety filter.
    """
    def __init__(self, name, pnr, n_players=2):
        self.name      = name
        self.pnr       = pnr
        self.n_players = n_players

        # Bayesian continuous giving model: posterior over how each teammate gives hints.
        self.giving_feature_models: dict[int, TeammateGivingFeatureModel] = {}

        # Bayesian response model: Dirichlet posterior over teammate responses to my hints.
        self.response_models: dict[int, TeammateResponseModel] = {}

        # Online meta-calibration: learns how much to trust each modeling feature
        # when converting teammate beliefs into hint utility.
        self.meta_calibrator = MetaUtilityCalibrator()
        self.pending_meta_hints = {}

        # Transient per-game state. These are reset at the start of each Game.
        self.pending_hints = {}
        self.gothint = None
        self.last_knowledge = None
        self.last_board = None
        self.last_trash = None
        self.last_played = None
        self.last_hints = None
        self.last_self_hint_key = None
        self.explanation = []


    def reset_episode_state(self):
        """
        Clear transient per-game state while keeping learned teammate models.
        This prevents a hint or pending response from the previous game from
        leaking into the next episode.
        """
        self.pending_hints = {}
        self.gothint = None
        self.last_knowledge = None
        self.last_board = None
        self.last_trash = None
        self.last_played = None
        self.last_hints = None
        self.last_self_hint_key = None
        self.explanation = []
        self.pending_meta_hints = {}


    def _ensure_giving_feature_model(self, player_id):
        if player_id not in self.giving_feature_models:
            self.giving_feature_models[player_id] = TeammateGivingFeatureModel(player_id)

    def _ensure_response_model(self, player_id):
        if player_id not in self.response_models:
            self.response_models[player_id] = TeammateResponseModel(player_id)




    def _make_continuous_fake_giver(self, player_id):
        """
        用已经学到的 continuous giving model 构造一个 fake teammate。
        用途：收到队友 hint 后做 self-recognition：
        哪些 possible self-hands 会让这个队友倾向于给出同样的 hint？
        """
        self._ensure_giving_feature_model(player_id)
        estimated_giving = self.giving_feature_models[player_id].estimate()

        # receiving style 在“模拟对方给 hint”时不用，所以设成中性。
        neutral_receiving = ReceivingStyle(
            trust_level=0.5,
            playable_interpretation=0.5,
            safe_interpretation=0.5,
            uncertainty_tolerance=0.5,
        )

        pair = TeammateStylePair(
            giving=estimated_giving,
            receiving=neutral_receiving,
        )

        return RealisticStylePlayer("fake_continuous", player_id, pair)


    def _hint_points_card(self, hint_action, card):
        col, num = card
        if hint_action.type == HINT_COLOR:
            return hint_action.col == col
        if hint_action.type == HINT_NUMBER:
            return hint_action.num == num
        return False

    def _giving_hint_evidence_from_hand(self, hint_action, target_hand, board, hints_before_action):
        """
        Convert one possible target hand into soft evidence for the giving model.
        This is intentionally soft: it describes what the chosen hint touched,
        without pretending that one sampled hand is certainly the real hand.
        """
        pointed_cards = []
        for col, num in target_hand:
            if self._hint_points_card(hint_action, (col, num)):
                pointed_cards.append((col, num))

        if not pointed_cards:
            return None

        play_hits = sum(
            1 for col, num in pointed_cards
            if board[col][1] + 1 == num
        )
        total_pointed = max(1, len(pointed_cards))

        return {
            "play_evidence": play_hits / total_pointed,
            "color_evidence": 1.0 if hint_action.type == HINT_COLOR else 0.0,
            "coverage_evidence": len(pointed_cards) / max(1, len(target_hand)),
            "conservatism_level": hints_before_action,
        }

    def _average_giving_hint_evidence(self, hint_action, possible_hands, board, hints_before_action):
        """
        Average evidence over many possible self-hands.
        This replaces the old random-choice update that often locked onto one
        unlucky sampled hand and caused wrong convergence.
        """
        evidences = []
        for h in possible_hands:
            ev = self._giving_hint_evidence_from_hand(
                hint_action, h, board, hints_before_action
            )
            if ev is not None:
                evidences.append(ev)

        if not evidences:
            return None

        keys = ["play_evidence", "color_evidence", "coverage_evidence", "conservatism_level"]
        return {
            k: float(np.mean([ev[k] for ev in evidences]))
            for k in keys
        }

    def _softmax_likelihood_of_hint(self, fake_hinter, hinter_id, candidate_hands,
                                    hands, knowledge, trash, played, board, hints,
                                    observed_hint):
        """
        Estimate whether the observed hint is actually likely under the current
        giving model, compared with the hint the fake giver would choose under
        each sampled self-hand.
        This gates noisy self-target evidence instead of always trusting it.
        """
        if not candidate_hands:
            return 0.0

        matches = 0
        trials = 0
        for h in candidate_hands:
            newhands = list(hands)
            newhands[observed_hint.pnr] = h
            if 0 <= hinter_id < len(newhands):
                newhands[hinter_id] = []
            try:
                fake_act = fake_hinter.get_action(
                    hinter_id, newhands, knowledge, trash, played, board, [], hints
                )
            except Exception:
                continue
            trials += 1
            if fake_act and fake_act == observed_hint:
                matches += 1

        if trials == 0:
            return 0.0
        return matches / trials

    def _hint_ambiguity(self, hint_action, target_hand, board):
        """
        Ambiguity of a hint under bounded-rational interpretation.
        Low ambiguity: the hint clearly points to playable or safely discardable cards.
        High ambiguity: the hint points to mixed or future-use cards.
        """
        touched = []
        for card in target_hand:
            if self._hint_points_card(hint_action, card):
                touched.append(card)

        if not touched:
            return 1.0

        playable_hits = 0
        safe_hits = 0
        future_hits = 0
        critical_hits = 0

        for col, num in touched:
            if board[col][1] + 1 == num:
                playable_hits += 1
            elif board[col][1] >= num:
                safe_hits += 1
            elif num == 5:
                critical_hits += 1
            else:
                future_hits += 1

        touched_n = max(1, len(touched))
        useful_ratio = (playable_hits + safe_hits) / touched_n
        mixed_intent_penalty = 1.0 if playable_hits > 0 and safe_hits > 0 else 0.0
        future_penalty = (future_hits + critical_hits) / touched_n

        ambiguity = 1.0 - useful_ratio
        ambiguity += 0.25 * mixed_intent_penalty
        ambiguity += 0.35 * future_penalty
        return max(0.0, min(1.0, ambiguity))

    def _bounded_response_confidence(self, target, hint_action, target_hand, board):
        """
        Confidence that the target teammate will produce a useful bounded-rational
        response after receiving this hint.
        """
        self._ensure_response_model(target)
        response_model = self.response_models[target]

        p_play = response_model.prob(hint_action.type, PLAY)
        p_discard = response_model.prob(hint_action.type, DISCARD)
        strength = response_model.posterior_strength(hint_action.type)
        obs_conf = min(1.0, max(0.0, (strength - 3.0) / 60.0))
        response_uncertainty = response_model.posterior_uncertainty(hint_action.type)
        ambiguity = self._hint_ambiguity(hint_action, target_hand, board)

        expected = 0.0
        touched = 0
        for col, num in target_hand:
            if not self._hint_points_card(hint_action, (col, num)):
                continue
            touched += 1
            if board[col][1] + 1 == num:
                expected += p_play
                expected -= 0.8 * p_discard
            elif board[col][1] >= num:
                expected += 0.7 * p_discard
                expected -= 0.5 * p_play
            elif num == 5:
                expected -= 0.8 * p_discard
            else:
                expected += 0.05

        if touched == 0:
            return -1.0

        # With few observations, fall back toward a neutral bounded-rational prior.
        prior = 0.35 * (1.0 - ambiguity)
        learned = expected / max(1, touched)
        posterior_predictive = (1.0 - obs_conf) * prior + obs_conf * learned
        return posterior_predictive - 0.15 * response_uncertainty

    def _cognitive_hint_utility(self, target, hint_action, safe_score, target_hand, board, hints=None):
        """
        Unified cognitive utility for adaptive hint selection.

        The base features come from Bayesian teammate models and the original
        intentional safety filter. Their weights are not fixed forever: the
        MetaUtilityCalibrator updates them online from prediction error after
        observing teammate responses to our hints.
        """
        self._ensure_response_model(target)
        self._ensure_giving_feature_model(target)

        response_model = self.response_models[target]
        giving_model = self.giving_feature_models[target]
        giving_est = giving_model.estimate()

        response_strength = response_model.posterior_strength(hint_action.type)
        response_conf = min(1.0, max(0.0, (response_strength - 3.0) / 60.0))
        response_uncertainty = response_model.posterior_uncertainty(hint_action.type)
        giving_uncertainty = giving_model.posterior_uncertainty()
        bounded_response = self._bounded_response_confidence(
            target, hint_action, target_hand, board
        )
        ambiguity = self._hint_ambiguity(hint_action, target_hand, board)

        channel_match = (
            giving_est.color_bias
            if hint_action.type == HINT_COLOR
            else 1.0 - giving_est.color_bias
        )

        touched = sum(
            1 for card in target_hand
            if self._hint_points_card(hint_action, card)
        )
        coverage = touched / max(1, len(target_hand))
        coverage_match = 1.0 - abs(coverage - giving_est.coverage_preference)

        hint_type_name = "color" if hint_action.type == HINT_COLOR else "number"
        if hints is None:
            hint_bucket = "hint_unknown"
        elif hints <= 2:
            hint_bucket = "hint_low"
        elif hints <= 5:
            hint_bucket = "hint_mid"
        else:
            hint_bucket = "hint_high"

        if ambiguity < 0.25:
            ambiguity_bucket = "amb_low"
        elif ambiguity < 0.60:
            ambiguity_bucket = "amb_mid"
        else:
            ambiguity_bucket = "amb_high"

        features = {
            "safe_score": float(safe_score),
            "bounded_response": float(bounded_response),
            "response_conf": float(response_conf),
            "channel_match": float(channel_match),
            "coverage_match": float(coverage_match),
            "ambiguity": float(ambiguity),
            "response_uncertainty": float(response_uncertainty),
            "giving_uncertainty": float(giving_uncertainty),
            "hint_type": hint_type_name,
            "hint_bucket": hint_bucket,
            "ambiguity_bucket": ambiguity_bucket,
        }
        return self.meta_calibrator.predict(features), features

    def _choose_hint_for_target(self, target, nr, hands, knowledge,
                                board, hints):
        """
        Bounded-rational adaptive hint selection.

        The original intentional `pretend()` safety filter is preserved. Among safe
        candidate hints, choose the hint with the highest Bayesian cognitive
        utility under response and giving-style posteriors.
        """
        if not hands[target] or target >= len(knowledge) or not knowledge[target]:
            return None

        self._ensure_response_model(target)
        self._ensure_giving_feature_model(target)

        best_score = -1e9
        best_action = None
        best_features = None

        candidates = []
        for c in set(col for col, _ in hands[target]):
            candidates.append(Action(HINT_COLOR, pnr=target, col=c))
        for r in set(num for _, num in hands[target]):
            candidates.append(Action(HINT_NUMBER, pnr=target, num=r))

        intentions = [None] * len(hands[target])
        for j, (col, num) in enumerate(hands[target]):
            if board[col][1] + 1 == num:
                intentions[j] = PLAY
            elif board[col][1] >= num:
                intentions[j] = DISCARD
            else:
                intentions[j] = CANDISCARD

        for act in candidates:
            action_tuple = ((HINT_COLOR, act.col) if act.type == HINT_COLOR
                            else (HINT_NUMBER, act.num))
            try:
                ok, safe_score, _ = pretend(action_tuple, knowledge[target],
                                            intentions, hands[target], board)
            except Exception:
                ok, safe_score = False, 0

            if not ok:
                continue

            score, features = self._cognitive_hint_utility(
                target,
                act,
                safe_score,
                hands[target],
                board,
                hints=hints,
            )

            if score > best_score:
                best_score = score
                best_action = act
                best_features = features

        if best_action is None:
            return None

        return best_action, best_score, best_features

    # ── 收到 hint 时：辅助推断自己的牌 ──────────────────────────────
    def _refine_own_knowledge_from_hint(self, nr, hands, knowledge,
                                        board, hints, trash, played):
        """
        基于「hint 发出者的给 hint 风格」的信念，做 sampling-based self-recognition。
        逻辑：枚举自己可能的手牌 → 对每种假设，用「对方最可能的风格模拟器」看
              它是否会给同样的 hint → 统计哪张牌的身份概率最高 → 更新 knowledge。
        """
        if not self.gothint:
            return knowledge

        (hint_act, hinter_id) = self.gothint
        self._ensure_giving_feature_model(hinter_id)

        # Always use continuous giving estimate for self-recognition.
        fake_hinter = self._make_continuous_fake_giver(hinter_id)

        snapshot_trash = self.last_trash if self.last_trash is not None else trash
        snapshot_played = self.last_played if self.last_played is not None else played
        snapshot_board = self.last_board if self.last_board is not None else board
        snapshot_hints = self.last_hints if self.last_hints is not None else hints + 1

        used = {}
        for c in (snapshot_trash + snapshot_played):
            used[c] = used.get(c, 0) + 1

        # 用 hint 发生前的 knowledge 做采样；如果长度已经和当前手牌不同，
        # 说明这条 gothint 已经过期，不能再用于 self-recognition。
        base_knowledge = self.last_knowledge[nr] if self.last_knowledge else knowledge[nr]
        if len(base_knowledge) != len(knowledge[nr]):
            self.gothint = None
            return knowledge

        possiblehands = []
        N_SAMPLES = 200
        for _ in range(N_SAMPLES):
            h = sample_hand(update_knowledge(base_knowledge, used))
            if len(h) != len(knowledge[nr]):
                continue
            newhands = list(hands)
            newhands[nr] = h

            # 在模拟 fake_hinter 的视角时，fake_hinter 不能看到自己的真实手牌。
            # Hanabi 中当前行动者只能看到其他人的手牌，不能看到自己的手牌。
            if 0 <= hinter_id < len(newhands):
                newhands[hinter_id] = []

            # 模拟对方在这个假设手牌下会给什么 hint
            fake_act = fake_hinter.get_action(
                hinter_id, newhands, self.last_knowledge,
                snapshot_trash, snapshot_played, snapshot_board,
                [], snapshot_hints)
            if fake_act and fake_act == hint_act:
                possiblehands.append(h)

        # 同一个 hint 只更新一次 teammate style belief。
        # 这一步用的是 sampled self hands 的 Monte Carlo likelihood，避免偷看真实手牌。
        hint_key = (
            hint_act.type,
            hint_act.pnr,
            hint_act.col,
            hint_act.num,
            len(snapshot_trash),
            len(snapshot_played),
            tuple(snapshot_board)
        )
        if hint_key != self.last_self_hint_key:
            # Self-targeted hints are used for self-recognition above,
            # but are not used to update the giving-style posterior:
            # sampled self-hands are too biased and caused wrong convergence.
            pass

            self.last_self_hint_key = hint_key

        # 统计每个位置最可能的牌；再次过滤长度异常的样本，防止旧 hint 引发越界。
        handsize = len(knowledge[nr])
        possiblehands = [h for h in possiblehands if len(h) == handsize]
        if len(possiblehands) < 3:
            return knowledge

        mostlikely = [(None, 0)] * handsize
        for i in range(handsize):
            counts = {}
            for h in possiblehands:
                c = h[i]
                counts[c] = counts.get(c, 0) + 1
            if not counts:
                continue
            best = max(counts.items(), key=lambda x: x[1])
            mostlikely[i] = best

        # 如果某张牌有极高置信度，才允许更新知识。
        # 原来的 0.6 太激进，会把 noisy self-recognition 结果当成真牌，导致误打/误丢。
        knowledge = copy.deepcopy(knowledge)
        for i, (card, cnt) in enumerate(mostlikely):
            if card is not None and cnt >= len(possiblehands) * 0.95 and len(possiblehands) >= 40:
                knowledge[nr][i] = iscard(card[0], card[1])

        return knowledge

    # ── 主决策函数 ────────────────────────────────────────────────────
    def get_action(self, nr, hands, knowledge, trash, played, board,
                   valid_actions, hints):
        """
        Strong adaptive policy:
        1. Keep the strong SelfIntentional-style action rule as the safe backbone.
        2. Use teammate belief mainly for adaptive hint selection.
        3. Avoid letting noisy self-recognition overwrite own-card knowledge unless it is very confident.
        """
        self.explanation = []
        handsize = len(knowledge[nr])
        result = None

        # Step 0: Update teammate belief from the latest self-targeted hint, but do not
        # trust the sampled self-recognition result enough to overwrite knowledge here.
        # The old version modified own-card knowledge aggressively and caused score drops.
        if self.gothint:
            knowledge = self._refine_own_knowledge_from_hint(
                nr, hands, knowledge, board, hints, trash, played)

        # Step 1: SelfIntentional-style interpretation of a received hint.
        # This is the strong original-source behavior used by the full baseline.
        hinted_actions = []
        if self.gothint:
            hint_act, _ = self.gothint
            if hint_act.type == HINT_COLOR:
                for k in knowledge[nr]:
                    hinted_actions.append(whattodo(k, sum(k[hint_act.col]) > 0, board))
            elif hint_act.type == HINT_NUMBER:
                for k in knowledge[nr]:
                    cnt = 0
                    for c in ALL_COLORS:
                        cnt += k[c][hint_act.num - 1]
                    hinted_actions.append(whattodo(k, cnt > 0, board))

        if hinted_actions:
            for i, a in enumerate(hinted_actions):
                if a == PLAY and (not result or result.type == DISCARD):
                    result = Action(PLAY, cnr=i)
                elif a == DISCARD and not result:
                    result = Action(DISCARD, cnr=i)

        self.gothint = None

        # Step 2: Deterministic play / discard from current public knowledge.
        possible = [get_possible(k) for k in knowledge[nr]]
        discards = []
        for i, p in enumerate(possible):
            if playable(p, board) and not result:
                result = Action(PLAY, cnr=i)
            if discardable(p, board):
                discards.append(i)

        hint_threshold = 8
        if discards and hints < hint_threshold and not result:
            result = Action(DISCARD, cnr=random.choice(discards))

        # Step 3: Adaptive hint selection under bounded-rational teammate modeling.
        # Compared with full/SelfIntentional, this uses learned response models,
        # ambiguity penalties, and weak giving-style compatibility.
        
        if hints > 0 and not result:
            best_hint = None
            best_score = -1e9
            meta_features = None

            for target in range(len(hands)):
                if target == nr or not hands[target]:
                    continue
                if target >= len(knowledge) or not knowledge[target]:
                    continue

                choice = self._choose_hint_for_target(
                    target, nr, hands, knowledge, board, hints)
                if choice is None:
                    continue
                act, cognitive_score, meta_feats = choice

                # Immediate public-state usefulness of the hinted cards.
                immediate = 0.0
                for col, n in hands[target]:
                    pointed = False
                    if act.type == HINT_COLOR and act.col == col:
                        pointed = True
                    if act.type == HINT_NUMBER and act.num == n:
                        pointed = True
                    if not pointed:
                        continue
                    if board[col][1] + 1 == n:
                        immediate += 4.0
                    elif board[col][1] >= n:
                        immediate += 1.5
                    elif n == 5:
                        immediate += 1.2
                    else:
                        immediate += 0.35

                score = immediate + cognitive_score

                if score > best_score:
                    best_score = score
                    best_hint = act
                    meta_features = meta_feats

            if best_hint is not None and best_score > 0:
                result = best_hint
                # Store meta-features for online calibration once the target responds.
                if result.pnr is not None and meta_features is not None:
                    self.pending_meta_hints[result.pnr] = {
                        "hint": copy.deepcopy(result),
                        "features": dict(meta_features),
                        "target_hand": list(hands[result.pnr]),
                        "board": copy.deepcopy(board),
                    }

        if result:
            return result

        # Step 4: Same safe discard fallback as Intentional/SelfIntentional.
        scores = [pretend_discard(Action(DISCARD, cnr=i), knowledge[nr], board, trash)
                  for i in range(handsize)]
        scores.sort(key=lambda x: -x[1])
        return scores[0][0]

    # ── 信息接收：更新队友模型 ────────────────────────────────────────
    def inform(self, action, player, public_state):
        """每次任何人行动后都会被调用"""
        # 1. 如果 teammate 对我上一条 hint 做出了 PLAY/DISCARD 反应，更新 response model。
        if player != self.pnr and action.type in [PLAY, DISCARD]:
            if player in self.pending_hints:
                pending = self.pending_hints.pop(player)
                self._ensure_response_model(player)
                self.response_models[player].update_from_response(
                    pending["hint"],
                    action,
                    pending["hand"],
                    pending["board"]
                )
                if player in self.pending_meta_hints:
                    meta_pending = self.pending_meta_hints.pop(player)
                    outcome = self._meta_outcome_from_response(
                        meta_pending["hint"],
                        action,
                        meta_pending["target_hand"],
                        meta_pending["board"],
                    )
                    self.meta_calibrator.update(meta_pending["features"], outcome)
        # If the target of my previous hint takes a non-card action, such as giving
        # another hint, then their next later play/discard should not be attributed
        # to my old hint. Count it as no direct response and clear the pending hint.
        if player != self.pnr and action.type in [HINT_COLOR, HINT_NUMBER]:
            if player in self.pending_hints:
                pending = self.pending_hints.pop(player)
                self._ensure_response_model(player)
                self.response_models[player].update_no_response(pending["hint"])
                if player in self.pending_meta_hints:
                    meta_pending = self.pending_meta_hints.pop(player)
                    self.meta_calibrator.update(meta_pending["features"], -0.5)
        if action.type in [HINT_COLOR, HINT_NUMBER]:
            # 如果别人后来又给了同一个 target hint，那么 target 的下一步反应
            # 可能是由后来的 hint 触发的。为避免错误归因，丢弃我之前给该 target 的 pending hint。
            if player != self.pnr and action.pnr in self.pending_hints:
                pending = self.pending_hints.pop(action.pnr)
                self._ensure_response_model(action.pnr)
                self.response_models[action.pnr].update_no_response(pending["hint"])
                if action.pnr in self.pending_meta_hints:
                    meta_pending = self.pending_meta_hints.pop(action.pnr)
                    self.meta_calibrator.update(meta_pending["features"], -0.5)

            # 如果这个 hint 是我给 teammate 的，记录下来，等 TA 下一步行动后更新 response model。
            if player == self.pnr and action.pnr != self.pnr:
                self.pending_hints[action.pnr] = {
                    "hint": copy.deepcopy(action),
                    "hand": list(public_state.hands[action.pnr]),
                    "board": copy.deepcopy(public_state.board),
                }
            # 若是队友给 hint，更新对应信念模型。
            # 注意：如果队友 hint 的目标是自己，则不能直接用 public_state.hands[self.pnr]
            # 来计算 likelihood，因为那等于偷看自己的真实手牌。
            # 所以这里先只用「队友 hint 给其他队友」的行为做风格更新；
            # hint 给自己的行为交给下面的 self-recognition 逻辑处理。
            if player != self.pnr and action.pnr != self.pnr:
                self._ensure_giving_feature_model(player)
                visible_hands = list(public_state.hands)
                visible_hands[self.pnr] = []

                self.giving_feature_models[player].update_from_choice(
                    action,
                    player,
                    visible_hands,
                    public_state.knowledge,
                    public_state.board,
                    public_state.hints,
                    lr=0.06,
                )
            # 若是对方 hint 给自己，记录以备 self-recognition
            if action.pnr == self.pnr:
                # 收到别人给自己的 hint 时，不直接用真实 self hand。
                # 后续在 _refine_own_knowledge_from_hint 中用 sampled possible hands
                # 更新 continuous giving model。
                self._ensure_giving_feature_model(player)

                self.gothint      = (action, player)
                self.last_knowledge = copy.deepcopy(public_state.knowledge)
                self.last_board     = copy.deepcopy(public_state.board)
                self.last_trash     = list(public_state.trash)
                self.last_played    = list(public_state.played)
                self.last_hints     = public_state.hints
    def _meta_outcome_from_response(self, hint_action, response_action, hinted_hand, board_at_hint):
        """
        Convert an observed teammate response into a scalar learning signal for
        meta-calibrating the utility features.
        Positive means the hint produced a useful response; negative means it led
        to a harmful or irrelevant response.
        """
        if response_action.type not in [PLAY, DISCARD]:
            return -0.5
        if response_action.cnr is None or response_action.cnr >= len(hinted_hand):
            return -0.5

        card = hinted_hand[response_action.cnr]
        if not self._hint_points_card(hint_action, card):
            return -0.75

        col, num = card
        if response_action.type == PLAY:
            if board_at_hint[col][1] + 1 == num:
                return 3.0
            return -2.0

        if response_action.type == DISCARD:
            if board_at_hint[col][1] >= num:
                return 1.5
            if num == 5:
                return -2.0
            return -0.5

        return -0.5

    def get_explanation(self):
        return self.explanation

    def print_beliefs(self):
        print(f"\n=== {self.name} (player {self.pnr}) Continuous Giving Feature Models ===")
        for pid, model in self.giving_feature_models.items():
            print(model)

        print(f"\n=== {self.name} (player {self.pnr}) Response Models ===")
        for pid, model in self.response_models.items():
            print(model)

        print(f"\n=== {self.name} (player {self.pnr}) Meta Utility Calibrator ===")
        print(self.meta_calibrator)

class AdaptiveNoMetaPlayer(AdaptivePlayer):
    """
    AdaptivePlayer without online meta-calibration.

    It still learns teammate giving/response beliefs, but the feature-to-utility
    mapping stays fixed.
    """
    def __init__(self, name, pnr, n_players=2):
        super().__init__(name, pnr, n_players)
        self.meta_calibrator = FrozenMetaUtilityCalibrator()

# ═══════════════════════════════════════════════════════════════════════
#
#  PART 4: 原版 Player 类（保留兼容性）
#
# ═══════════════════════════════════════════════════════════════════════

class Player:
    def __init__(self, name, pnr):
        self.name = name
        self.explanation = []

    def get_action(self, nr, hands, knowledge, trash, played, board,
                   valid_actions, hints):
        return random.choice(valid_actions)

    def inform(self, action, player, public_state):
        pass

    def get_explanation(self):
        return self.explanation


class InnerStatePlayer(Player):
    def get_action(self, nr, hands, knowledge, trash, played, board,
                   valid_actions, hints):
        possible = [get_possible(k) for k in knowledge[nr]]
        discards = []
        for i, p in enumerate(possible):
            if playable(p, board):   return Action(PLAY, cnr=i)
            if discardable(p, board): discards.append(i)
        if discards:
            return Action(DISCARD, cnr=random.choice(discards))
        playables = [(i, j) for i, h in enumerate(hands) if i != nr
                     for j, (col, n) in enumerate(h) if board[col][1] + 1 == n]
        if playables and hints > 0:
            i, j = playables[0]
            return (Action(HINT_COLOR, pnr=i, col=hands[i][j][0])
                    if random.random() < 0.5
                    else Action(HINT_NUMBER, pnr=i, num=hands[i][j][1]))
        return random.choice([Action(DISCARD, cnr=i) for i in range(len(knowledge[nr]))])


class IntentionalPlayer(Player):
    def __init__(self, name, pnr):
        self.name = name
        self.pnr  = pnr
        self.explanation = []

    def get_action(self, nr, hands, knowledge, trash, played, board,
                   valid_actions, hints):
        handsize = len(knowledge[nr])
        possible = [get_possible(k) for k in knowledge[nr]]
        result = None
        self.explanation = []

        # 1. Deterministic own play / discard
        for i, p in enumerate(possible):
            if playable(p, board):
                result = Action(PLAY, cnr=i)
        discards = [i for i, p in enumerate(possible) if discardable(p, board)]
        hint_threshold = 8
        if discards and hints < hint_threshold and not result:
            result = Action(DISCARD, cnr=random.choice(discards))

        # 2. Multi-player intentional hint selection.
        # Original pyhanabi code used 1-nr, which only works for 2 players.
        if hints > 0 and not result:
            valid = []
            for target in range(len(hands)):
                if target == nr or not hands[target]:
                    continue
                if target >= len(knowledge) or not knowledge[target]:
                    continue

                intentions = [None] * len(hands[target])
                for j, (col, num) in enumerate(hands[target]):
                    if board[col][1] + 1 == num:
                        intentions[j] = PLAY
                    elif board[col][1] >= num:
                        intentions[j] = DISCARD
                    elif num < 5:
                        intentions[j] = CANDISCARD

                for c in ALL_COLORS:
                    act = (HINT_COLOR, c)
                    try:
                        ok, sc, _ = pretend(act, knowledge[target], intentions,
                                            hands[target], board)
                    except Exception:
                        continue
                    if ok:
                        valid.append((target, act, sc))
                for r in range(1, 6):
                    act = (HINT_NUMBER, r)
                    try:
                        ok, sc, _ = pretend(act, knowledge[target], intentions,
                                            hands[target], board)
                    except Exception:
                        continue
                    if ok:
                        valid.append((target, act, sc))

            if valid:
                valid.sort(key=lambda x: -x[2])
                target, a, _ = valid[0]
                result = (Action(HINT_COLOR, pnr=target, col=a[1])
                          if a[0] == HINT_COLOR
                          else Action(HINT_NUMBER, pnr=target, num=a[1]))

        if result:
            return result

        scores = [pretend_discard(Action(DISCARD, cnr=i), knowledge[nr], board, trash)
                  for i in range(handsize)]
        scores.sort(key=lambda x: -x[1])
        return scores[0][0]

    def inform(self, action, player, public_state):
        pass


class OuterStatePlayer(Player):
    """
    Baseline from the original pyhanabi source.
    It uses visible outer-state information: if it sees a teammate has a playable card,
    it gives a color/rank hint for that card, while avoiding repeating the same hint type
    for the same card position.
    """
    def __init__(self, name, pnr):
        self.name = name
        self.pnr = pnr
        self.hints = {}
        self.explanation = []

    def get_action(self, nr, hands, knowledge, trash, played, board,
                   valid_actions, hints):
        handsize = len(knowledge[nr])
        possible = [get_possible(k) for k in knowledge[nr]]

        discards = []
        for i, p in enumerate(possible):
            if playable(p, board):
                return Action(PLAY, cnr=i)
            if discardable(p, board):
                discards.append(i)

        if discards:
            return Action(DISCARD, cnr=random.choice(discards))

        playables = []
        for i, h in enumerate(hands):
            if i == nr:
                continue
            for j, (col, n) in enumerate(h):
                if board[col][1] + 1 == n:
                    playables.append((i, j))
        playables.sort(key=lambda ij: -hands[ij[0]][ij[1]][1])

        while playables and hints > 0:
            i, j = playables[0]
            hint_types = [HINT_COLOR, HINT_NUMBER]
            if (j, i) not in self.hints:
                self.hints[(j, i)] = []
            for old_hint in self.hints[(j, i)]:
                if old_hint in hint_types:
                    hint_types.remove(old_hint)

            if hint_types:
                chosen = random.choice(hint_types)
                self.hints[(j, i)].append(chosen)
                if chosen == HINT_COLOR:
                    return Action(HINT_COLOR, pnr=i, col=hands[i][j][0])
                return Action(HINT_NUMBER, pnr=i, num=hands[i][j][1])

            playables = playables[1:]

        for i, k in enumerate(knowledge):
            if i == nr or not hands[i]:
                continue
            cards = list(range(len(k)))
            random.shuffle(cards)
            c = cards[0]
            col, num = hands[i][c]
            hint_types = [HINT_COLOR, HINT_NUMBER]
            if (c, i) not in self.hints:
                self.hints[(c, i)] = []
            for old_hint in self.hints[(c, i)]:
                if old_hint in hint_types:
                    hint_types.remove(old_hint)
            if hint_types and hints > 0:
                chosen = random.choice(hint_types)
                self.hints[(c, i)].append(chosen)
                if chosen == HINT_COLOR:
                    return Action(HINT_COLOR, pnr=i, col=col)
                return Action(HINT_NUMBER, pnr=i, num=num)

        return random.choice([Action(DISCARD, cnr=i) for i in range(handsize)])

    def inform(self, action, player, public_state):
        if action.type in [PLAY, DISCARD]:
            if (action.cnr, player) in self.hints:
                self.hints[(action.cnr, player)] = []
            for i in range(10):
                key_old = (action.cnr + i, player)
                key_new = (action.cnr + i + 1, player)
                if key_new in self.hints:
                    self.hints[key_old] = self.hints[key_new]
                    self.hints[key_new] = []


class SelfIntentionalPlayer(Player):
    """
    Baseline from the original pyhanabi source.
    It extends IntentionalPlayer by first interpreting hints received by itself:
    a pointed card that is potentially playable is treated as intended PLAY;
    a pointed card that is potentially discardable is treated as intended DISCARD.
    """
    def __init__(self, name, pnr):
        self.name = name
        self.pnr = pnr
        self.hints = {}
        self.gothint = None
        self.last_knowledge = []
        self.last_board = []
        self.last_trash = []
        self.last_played = []
        self.explanation = []

    def get_action(self, nr, hands, knowledge, trash, played, board,
                   valid_actions, hints):
        handsize = len(knowledge[nr])
        result = None

        hinted_actions = []
        if self.gothint:
            act, _ = self.gothint
            if act.type == HINT_COLOR:
                for k in knowledge[nr]:
                    hinted_actions.append(whattodo(k, sum(k[act.col]) > 0, board))
            elif act.type == HINT_NUMBER:
                for k in knowledge[nr]:
                    cnt = 0
                    for c in ALL_COLORS:
                        cnt += k[c][act.num - 1]
                    hinted_actions.append(whattodo(k, cnt > 0, board))

        if hinted_actions:
            for i, a in enumerate(hinted_actions):
                if a == PLAY and (not result or result.type == DISCARD):
                    result = Action(PLAY, cnr=i)
                elif a == DISCARD and not result:
                    result = Action(DISCARD, cnr=i)

        self.gothint = None

        possible = [get_possible(k) for k in knowledge[nr]]
        discards = []
        for i, p in enumerate(possible):
            if playable(p, board) and not result:
                result = Action(PLAY, cnr=i)
            if discardable(p, board):
                discards.append(i)

        hint_threshold = 8
        if discards and hints < hint_threshold and not result:
            result = Action(DISCARD, cnr=random.choice(discards))

        target = None
        for i in range(len(hands)):
            if i != nr and hands[i]:
                target = i
                break

        if hints > 0 and not result and target is not None:
            intentions = [None] * len(hands[target])
            othercards = trash + played
            for j, (col, n) in enumerate(hands[target]):
                if board[col][1] + 1 == n:
                    intentions[j] = PLAY
                elif board[col][1] >= n:
                    intentions[j] = DISCARD
                elif n < 5 and (col, n) not in othercards:
                    intentions[j] = CANDISCARD

            valid = []
            for c in ALL_COLORS:
                act = (HINT_COLOR, c)
                ok, sc, _ = pretend(act, knowledge[target], intentions,
                                    hands[target], board)
                if ok:
                    valid.append((act, sc))
            for r in range(1, 6):
                act = (HINT_NUMBER, r)
                ok, sc, _ = pretend(act, knowledge[target], intentions,
                                    hands[target], board)
                if ok:
                    valid.append((act, sc))

            if valid:
                valid.sort(key=lambda x: -x[1])
                a, _ = valid[0]
                if a[0] == HINT_COLOR:
                    result = Action(HINT_COLOR, pnr=target, col=a[1])
                else:
                    result = Action(HINT_NUMBER, pnr=target, num=a[1])

        if result:
            return result

        scores = [pretend_discard(Action(DISCARD, cnr=i), knowledge[nr], board, trash)
                  for i in range(handsize)]
        scores.sort(key=lambda x: -x[1])
        return scores[0][0]

    def inform(self, action, player, public_state):
        if action.type in [PLAY, DISCARD]:
            if (action.cnr, player) in self.hints:
                self.hints[(action.cnr, player)] = []
            for i in range(10):
                key_old = (action.cnr + i, player)
                key_new = (action.cnr + i + 1, player)
                if key_new in self.hints:
                    self.hints[key_old] = self.hints[key_new]
                    self.hints[key_new] = []
        elif action.pnr == self.pnr:
            self.gothint = (action, player)
            self.last_knowledge = copy.deepcopy(public_state.knowledge)
            self.last_board = copy.deepcopy(public_state.board)
            self.last_trash = list(public_state.trash)
            self.last_played = list(public_state.played)


# ═══════════════════════════════════════════════════════════════════════
#
#  PART 5: Sampling 工具函数
#
# ═══════════════════════════════════════════════════════════════════════

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
    other = do_sample(knowledge[1:])
    if other is None:
        return None
    return [random.choice(possible)] + other

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


# ═══════════════════════════════════════════════════════════════════════
#
#  PART 6: Game 引擎（基本同原版，增加 study 模式）
#
# ═══════════════════════════════════════════════════════════════════════

class NullStream:
    def write(self, *args): pass
    def flush(self):        pass


class Game:
    def __init__(self, players, log=sys.stdout, format=0):
        self.players        = players
        for p in self.players:
            if hasattr(p, "reset_episode_state"):
                p.reset_episode_state()
        self.hits           = 3
        self.hints          = 8
        self.current_player = 0
        self.board          = [(c, 0) for c in ALL_COLORS]
        self.played         = []
        self.deck           = make_deck()
        self.extra_turns    = 0
        self.hands          = []
        self.knowledge      = []
        self.trash          = []
        self.log            = log
        self.turn           = 1
        self.format         = format
        self._make_hands()

    def _make_hands(self):
        handsize = 4 if len(self.players) >= 4 else 5
        for i, _ in enumerate(self.players):
            self.hands.append([])
            self.knowledge.append([])
            for _ in range(handsize):
                self._draw_card(i)

    def _draw_card(self, pnr=None):
        if pnr is None: pnr = self.current_player
        if not self.deck: return
        self.hands[pnr].append(self.deck[0])
        self.knowledge[pnr].append(initial_knowledge())
        del self.deck[0]

    def public_state_for(self, observer_pnr):
        """
        Return a public pre-action snapshot for one observer.

        The observer can see every other player's hand, but not their own hand.
        The deck and other hidden internals are intentionally not exposed.
        """
        visible_hands = [
            [] if i == observer_pnr else list(hand)
            for i, hand in enumerate(self.hands)
        ]
        return PublicState(
            hands=visible_hands,
            knowledge=copy.deepcopy(self.knowledge),
            board=copy.deepcopy(self.board),
            trash=list(self.trash),
            played=list(self.played),
            hints=int(self.hints),
            current_player=int(self.current_player),
        )

    def perform(self, action):
        for observer_pnr, p in enumerate(self.players):
            p.inform(action, self.current_player, self.public_state_for(observer_pnr))

        if action.type == HINT_COLOR:
            self.hints -= 1
            print(f"{self.players[self.current_player].name} hints "
                  f"{self.players[action.pnr].name} about "
                  f"{COLORNAMES[action.col]} cards  [hints={self.hints}]",
                  file=self.log)
            for (col, num), k in zip(self.hands[action.pnr],
                                     self.knowledge[action.pnr]):
                if col == action.col:
                    for i, row in enumerate(k):
                        if i != col:
                            k[i] = [0] * len(row)
                else:
                    k[action.col] = [0] * len(k[action.col])

        elif action.type == HINT_NUMBER:
            self.hints -= 1
            print(f"{self.players[self.current_player].name} hints "
                  f"{self.players[action.pnr].name} about "
                  f"{action.num}s  [hints={self.hints}]",
                  file=self.log)
            for (col, num), k in zip(self.hands[action.pnr],
                                     self.knowledge[action.pnr]):
                if num == action.num:
                    for row in k:
                        for i in range(len(COUNTS)):
                            if i + 1 != num: row[i] = 0
                else:
                    for row in k:
                        row[action.num - 1] = 0

        elif action.type == PLAY:
            col, num = self.hands[self.current_player][action.cnr]
            if self.board[col][1] == num - 1:
                self.board[col] = (col, num)
                self.played.append((col, num))
                if num == 5:
                    self.hints = min(self.hints + 1, 8)
                print(f"{self.players[self.current_player].name} plays "
                      f"{format_card((col,num))} ✓  board={format_hand(self.board)}",
                      file=self.log)
            else:
                self.trash.append((col, num))
                self.hits -= 1
                print(f"{self.players[self.current_player].name} plays "
                      f"{format_card((col,num))} ✗  hits={self.hits}",
                      file=self.log)
            del self.hands[self.current_player][action.cnr]
            del self.knowledge[self.current_player][action.cnr]
            self._draw_card()

        else:  # DISCARD
            self.hints = min(self.hints + 1, 8)
            card = self.hands[self.current_player][action.cnr]
            self.trash.append(card)
            print(f"{self.players[self.current_player].name} discards "
                  f"{format_card(card)}",
                  file=self.log)
            del self.hands[self.current_player][action.cnr]
            del self.knowledge[self.current_player][action.cnr]
            self._draw_card()

    def valid_actions(self):
        valid = []
        cp = self.current_player
        for i in range(len(self.hands[cp])):
            valid.append(Action(PLAY,    cnr=i))
            valid.append(Action(DISCARD, cnr=i))
        if self.hints > 0:
            for i, _ in enumerate(self.players):
                if i != cp:
                    for col in set(c for c, _ in self.hands[i]):
                        valid.append(Action(HINT_COLOR,  pnr=i, col=col))
                    for num in set(n for _, n in self.hands[i]):
                        valid.append(Action(HINT_NUMBER, pnr=i, num=num))
        return valid

    def done(self):
        if self.extra_turns == len(self.players) or self.hits == 0:
            return True
        return all(num == 5 for _, num in self.board)

    def score(self):
        return sum(num for _, num in self.board)

    def run(self, turns=-1):
        self.turn = 1
        while not self.done() and (turns < 0 or self.turn < turns):
            self.turn += 1
            if not self.deck:
                self.extra_turns += 1
            hands = [[] if i == self.current_player else h
                     for i, h in enumerate(self.hands)]
            action = self.players[self.current_player].get_action(
                self.current_player, hands, self.knowledge,
                self.trash, self.played, self.board,
                self.valid_actions(), self.hints)
            self.perform(action)
            self.current_player = (self.current_player + 1) % len(self.players)
        print(f"Game done. hits={self.hits} score={self.score()}", file=self.log)
        return self.score()

    def finish(self): pass


# ═══════════════════════════════════════════════════════════════════════
#
#  PART 7: 实验框架
#
# ═══════════════════════════════════════════════════════════════════════

playertypes = {
    "random":      Player,
    "inner":       InnerStatePlayer,
    "outer":       OuterStatePlayer,
    "intentional": IntentionalPlayer,
    "full":        SelfIntentionalPlayer,
    "realistic":   RealisticStylePlayer,
    "adaptive":    AdaptivePlayer,
    "adaptive_nometa": AdaptiveNoMetaPlayer,
}


names = ["Shangdi", "Yu Di", "Tian", "Nu Wa", "Pangu"]

def make_base_seed(base_seed=None):
    """
    If base_seed is None, create a fresh run seed so repeated command-line runs
    are not identical. If base_seed is provided, use it for reproducibility.
    """
    if base_seed is not None:
        return int(base_seed)
    return int.from_bytes(os.urandom(4), "little")

def significance_marker(p_value, alpha=0.05):
    eps = 1e-12
    if not np.isfinite(p_value):
        return "ns"
    if p_value <= 0.001 + eps:
        return "***"
    if p_value <= 0.01 + eps:
        return "**"
    if p_value <= alpha + eps:
        return "*"
    return "ns"

def autocorr_adjusted_se(diff, max_lag=None):
    """
    Newey-West style standard error for paired repeated-game differences.
    """
    x = np.asarray(diff, dtype=float)
    n = len(x)
    if n <= 1:
        return 0.0, 1.0

    x = x - float(np.mean(x))

    if max_lag is None:
        max_lag = int(min(n - 1, max(1, np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))))
    max_lag = int(max(0, min(max_lag, n - 1)))

    gamma0 = float(np.dot(x, x) / n)
    long_run_var = gamma0

    for lag in range(1, max_lag + 1):
        cov = float(np.dot(x[lag:], x[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1.0)
        long_run_var += 2.0 * weight * cov

    long_run_var = max(long_run_var, 0.0)
    se = float(np.sqrt(long_run_var / n))

    if long_run_var > 1e-12:
        effective_n = min(float(n), max(1.0, gamma0 / long_run_var * n))
    else:
        effective_n = float(n)

    return se, effective_n

def add_multiple_testing_corrections(stats_rows):
    """
    Add Bonferroni and Benjamini-Hochberg FDR corrected p-values in-place.
    """
    m = len(stats_rows)
    if m == 0:
        return stats_rows

    raw = np.array([row.get("p_value", 1.0) for row in stats_rows], dtype=float)
    raw = np.where(np.isfinite(raw), raw, 1.0)

    bonf = np.minimum(raw * m, 1.0)

    order = np.argsort(raw)
    fdr = np.ones(m, dtype=float)
    running = 1.0

    for rank_from_end, idx in enumerate(order[::-1], start=1):
        rank = m - rank_from_end + 1
        adjusted = raw[idx] * m / max(1, rank)
        running = min(running, adjusted)
        fdr[idx] = min(running, 1.0)

    for i, row in enumerate(stats_rows):
        row["p_bonf"] = float(bonf[i])
        row["p_fdr"] = float(fdr[i])
        row["sig_bonf"] = significance_marker(row["p_bonf"])
        row["sig_fdr"] = significance_marker(row["p_fdr"])

    return stats_rows

def paired_stats(scores_a, scores_b):
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    diff = a - b
    n = len(diff)

    mean_diff = float(np.mean(diff)) if n > 0 else 0.0
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0

    se, effective_n = autocorr_adjusted_se(diff)
    ci_low = mean_diff - 1.96 * se
    ci_high = mean_diff + 1.96 * se

    if n > 1 and se > 1e-12:
        df = max(1.0, effective_n - 1.0)
        t_stat = mean_diff / se
        p_value = 2.0 * float(student_t.sf(abs(t_stat), df))
        effect_dz = mean_diff / std_diff if std_diff > 1e-12 else 0.0
    else:
        p_value = 1.0
        effect_dz = 0.0

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "se": se,
        "effective_n": float(effective_n),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": float(p_value),
        "sig": significance_marker(float(p_value)),
        "effect_dz": float(effect_dz),
        "win_rate": float(np.mean(diff > 0.0)) if n > 0 else 0.0,
    }

def make_player(player_str, i, n_players=None):

    if player_str == "adaptive":
        return AdaptivePlayer(names[i], i, n_players or 4)
    if player_str == "adaptive_nometa":
        return AdaptiveNoMetaPlayer(names[i], i, n_players or 4)

    # Plain realistic teammate: style comes from the global style configuration.
    # With --style-mode none, this falls back to the old sample_teammate_style().
    if player_str == "realistic":
        return RealisticStylePlayer(names[i], i)

    # Configured teammate used by make_teammate_sets() when --style-mode is active.
    # The explicit seed makes each teamset/player reproducible while still obeying
    # manual / realistic / hybrid / fixed modes.
    if player_str.startswith("configured("):
        raw = player_str[11:-1].strip()
        seed = int(raw) if raw else None
        style_pair = style_pair_from_mode_seed(i, seed)
        return RealisticStylePlayer(names[i], i, style_pair=style_pair)

    # Backward-compatible old continuous teammate string.
    if player_str.startswith("realistic("):
        raw = player_str[10:-1].strip()
        seed = int(raw) if raw else None
        teammate_seed = seed * 1009 + i * 9176 if seed is not None else None

        return RealisticStylePlayer(
            names[i],
            i,
            sample_teammate_style(teammate_seed),
        )

    if player_str in playertypes:
        return playertypes[player_str](names[i], i)

    return Player(names[i], i)


def make_teammate_sets(base_seed=0):
    """
    Return five 4-player experiment teammate sets.

    When TEAMMATE_STYLE_MODE == "none", preserve the old behavior:
    each teammate string is realistic(seed), which directly samples continuous
    realistic parameters.

    When --style-mode manual / realistic / hybrid / fixed is active, return
    configured(seed) teammates. make_player() will then generate each teammate's
    style according to the selected style mode, so the command-line option
    actually affects all main experiments.
    """
    offset = int(base_seed) * 1000
    seeds = [
        [offset + 11,  offset + 29,  offset + 47],
        [offset + 61,  offset + 83,  offset + 107],
        [offset + 131, offset + 157, offset + 181],
        [offset + 211, offset + 241, offset + 271],
        [offset + 307, offset + 347, offset + 389],
    ]

    prefix = "realistic" if TEAMMATE_STYLE_MODE == "none" else "configured"
    return [[f"{prefix}({s})" for s in team] for team in seeds]


def build_team(first_player_str, teammates):
    """
    Build one fresh 4-player team under a fixed continuous teammate set.
    """
    players = [make_player(first_player_str, 0, 4)]
    for pid, pstr in enumerate(teammates, start=1):
        players.append(make_player(pstr, pid, 4))
    return players

# ─── 实验1：单次对局演示 ────────────────────────────────────────────
def run_demo(player_strs=None, seed=42, verbose=True):
    if player_strs is None:
        player_strs = ["adaptive", "realistic(11)", "realistic(29)", "realistic(47)"]
    random.seed(seed)
    players = [make_player(p, i, len(player_strs)) for i, p in enumerate(player_strs)]
    log = sys.stdout if verbose else NullStream()
    g = Game(players, log=log)
    score = g.run()
    if verbose:
        print(f"\nFinal score: {score}")
        for p in players:
            if isinstance(p, AdaptivePlayer):
                p.print_beliefs()
    return score


# ─── 实验2：N 局批量测试 ────────────────────────────────────────────
def run_batch(player_strs, n_games=500, reset_beliefs=False, verbose=False):
    """
    reset_beliefs=False → learning agents keep their teammate models across games.
    reset_beliefs=True  → learning agents are recreated every game.

    Non-learning players are rebuilt every game to avoid leaking transient state.
    """
    log = NullStream()
    scores = []

    persistent_players = {}

    if not reset_beliefs:
        for i, pstr in enumerate(player_strs):
            if pstr in ["adaptive", "adaptive_nometa"]:
                persistent_players[i] = make_player(pstr, i, len(player_strs))

    for game_idx in range(n_games):
        if (game_idx + 1) % 100 == 0:
            print(f"  game {game_idx+1}/{n_games}")

        players = []
        for i, pstr in enumerate(player_strs):
            if not reset_beliefs and i in persistent_players:
                players.append(persistent_players[i])
            else:
                players.append(make_player(pstr, i, len(player_strs)))

        random.seed(random.randint(0, 2**31 - 1))
        g = Game(players, log=log)
        sc = g.run()
        scores.append(sc)

    arr = np.array(scores)
    print(f"  avg={arr.mean():.2f}  std={arr.std(ddof=1):.2f}  "
          f"min={arr.min()}  max={arr.max()}")
    return scores


# ─── 实验3：信念收敛分析 ────────────────────────────────────────────
def run_belief_convergence(teamset_id=1, n_games=30, base_seed=None):
    """
    Continuous setting: fixed realistic teammates, observe learned giving features.
    """
    base_seed = make_base_seed(base_seed)
    teammate_sets = make_teammate_sets(base_seed)
    teammates = teammate_sets[(teamset_id - 1) % len(teammate_sets)]

    print(f"\n=== Giving Feature Convergence: continuous teamset {teamset_id} ===")
    print(f"Base seed: {base_seed}")
    random.seed(base_seed)
    adaptive = AdaptivePlayer(names[0], 0, 4)

    for game_idx in range(n_games):
        random.seed(base_seed + game_idx + 1)

        players = [adaptive]
        for pid, pstr in enumerate(teammates, start=1):
            players.append(make_player(pstr, pid, 4))

        g = Game(players, log=NullStream())
        g.run()

        model_summaries = []
        for pid in range(1, 4):
            if pid in adaptive.giving_feature_models:
                m = adaptive.giving_feature_models[pid]
                est = m.estimate()
                model_summaries.append(
                    f"P{pid}:obs={m.n},play={est.play_bias:.2f},"
                    f"color={est.color_bias:.2f},cons={est.conservatism:.2f},"
                    f"cov={est.coverage_preference:.2f}"
                )
            else:
                model_summaries.append(f"P{pid}:obs=0")

        print(f"  Game {game_idx+1:2d} | " + " | ".join(model_summaries))


def run_compare_table(n_games=300, base_seed=None):
    """
    Communication hierarchy comparison:

    OuterState
        ↓
    Intentional
        ↓
    SelfIntentional (Full)
        ↓
    Adaptive

    This evaluates whether partner-specific adaptation improves
    over fixed shared-convention intentional reasoning.
    """
    base_seed = make_base_seed(base_seed)

    print(f"Base seed: {base_seed}")

    print("\n=== Hanabi Communication Comparison ===")

    print(
        f"{'TeamSet':8s}  "
        f"{'Outer':>8s}  "
        f"{'Intentional':>12s}  "
        f"{'Full':>8s}  "
        f"{'Adaptive':>10s}"
    )

    print("-" * 65)

    for set_idx, teammates in enumerate(make_teammate_sets(base_seed), start=1):

        results = {}

        models = [
            ("outer", "Outer"),
            ("intentional", "Intentional"),
            ("full", "Full"),
            ("adaptive", "Adaptive"),
        ]

        for model_key, model_name in models:

            scores = []

            if model_key == "adaptive":
                persistent_player = AdaptivePlayer(names[0], 0, 4)

            for game_idx in range(n_games):

                seed = base_seed + game_idx + 1
                random.seed(seed)

                players = []

                if model_key == "adaptive":
                    players.append(persistent_player)
                else:
                    players.append(make_player(model_key, 0, 4))

                for pid, pstr in enumerate(teammates, start=1):
                    players.append(make_player(pstr, pid, 4))

                g = Game(players, log=NullStream())

                scores.append(g.run())

            results[model_name] = float(np.mean(scores))

        print(
            f"{'set'+str(set_idx):8s}  "
            f"{results['Outer']:8.2f}  "
            f"{results['Intentional']:12.2f}  "
            f"{results['Full']:8.2f}  "
            f"{results['Adaptive']:10.2f}"
        )

def run_meta_ablation_table(n_games=300, base_seed=None):
    """
    Compare Full, AdaptiveNoMeta, and AdaptiveMeta.
    """
    base_seed = make_base_seed(base_seed)

    print(f"Base seed: {base_seed}")
    print("\n=== Meta-Calibration Ablation: Full vs AdaptiveNoMeta vs AdaptiveMeta ===")
    print(
        f"{'TeamSet':8s}  {'Full':>8s}  {'NoMeta':>10s}  {'Meta':>10s}  "
        f"{'Meta-NoMeta':>12s}  {'SE':>7s}  {'95% CI':>19s}  "
        f"{'p':>11s}  {'pFDR':>11s}  {'dz':>7s}  {'Win%':>7s}"
    )
    print("-" * 125)

    rows = []
    all_full = []
    all_nometa = []
    all_meta = []

    for set_idx, teammates in enumerate(make_teammate_sets(base_seed), start=1):
        scores_full = []
        scores_nometa = []
        scores_meta = []

        meta_player = AdaptivePlayer(names[0], 0, 4)
        nometa_player = AdaptiveNoMetaPlayer(names[0], 0, 4)

        for game_idx in range(n_games):
            seed = base_seed + set_idx * 100000 + game_idx + 1

            random.seed(seed)
            players_f = build_team("full", teammates)
            scores_full.append(Game(players_f, log=NullStream()).run())

            random.seed(seed)
            players_n = [nometa_player]
            for pid, pstr in enumerate(teammates, start=1):
                players_n.append(make_player(pstr, pid, 4))
            scores_nometa.append(Game(players_n, log=NullStream()).run())

            random.seed(seed)
            players_m = [meta_player]
            for pid, pstr in enumerate(teammates, start=1):
                players_m.append(make_player(pstr, pid, 4))
            scores_meta.append(Game(players_m, log=NullStream()).run())

        all_full.extend(scores_full)
        all_nometa.extend(scores_nometa)
        all_meta.extend(scores_meta)

        st = paired_stats(scores_meta, scores_nometa)
        ci = f"[{st['ci_low']:+.2f},{st['ci_high']:+.2f}]"

        rows.append({
            "label": f"set{set_idx:<4d}",
            "full": float(np.mean(scores_full)),
            "nometa": float(np.mean(scores_nometa)),
            "meta": float(np.mean(scores_meta)),
            "ci": ci,
            **st,
        })

    add_multiple_testing_corrections(rows)

    for row in rows:
        print(
            f"{row['label']:8s}  {row['full']:8.2f}  {row['nometa']:10.2f}  {row['meta']:10.2f}  "
            f"{row['mean_diff']:+12.2f}  {row['se']:7.3f}  {row['ci']:>19s}  "
            f"p={row['p_value']:.4g} {row['sig']:>3s}  "
            f"q={row['p_fdr']:.4g} {row['sig_fdr']:>3s}  "
            f"{row['effect_dz']:7.3f}  {100.0 * row['win_rate']:6.1f}%"
        )

    st_all = paired_stats(all_meta, all_nometa)
    ci_all = f"[{st_all['ci_low']:+.2f},{st_all['ci_high']:+.2f}]"

    print("-" * 125)
    print(
        f"{'ALL':8s}  {np.mean(all_full):8.2f}  {np.mean(all_nometa):10.2f}  {np.mean(all_meta):10.2f}  "
        f"{st_all['mean_diff']:+12.2f}  {st_all['se']:7.3f}  {ci_all:>19s}  "
        f"p={st_all['p_value']:.4g} {st_all['sig']:>3s}  "
        f"{'':>15s}  {st_all['effect_dz']:7.3f}  {100.0 * st_all['win_rate']:6.1f}%"
    )

    print("\nFinal meta calibrator diagnostics:")
    print(f"  Meta:   {meta_player.meta_calibrator}")
    print(f"  NoMeta: {nometa_player.meta_calibrator}")

def _make_fixed_realistic_teammates(teammates):
    fixed_players = []
    for pid, pstr in enumerate(teammates, start=1):
        fixed_players.append(make_player(pstr, pid, 4))
    return fixed_players


def _clone_fixed_teammates(fixed_teammates):
    cloned = []
    for p in fixed_teammates:
        if isinstance(p, RealisticStylePlayer):
            cloned.append(RealisticStylePlayer(p.name, p.pnr, style_pair=p.style_pair))
        else:
            cloned.append(copy.deepcopy(p))
    return cloned


def run_adaptive_score_curve(n_games=200, base_seed=None, teamset_id=1, window=20):
    base_seed = make_base_seed(base_seed)
    teammate_sets = make_teammate_sets(base_seed)
    teammates = teammate_sets[(teamset_id - 1) % len(teammate_sets)]

    fixed_teammates = _make_fixed_realistic_teammates(teammates)
    adaptive = AdaptivePlayer(names[0], 0, 4)

    scores = []

    print(f"Base seed: {base_seed}")
    print(f"\n=== Adaptive Score Curve: repeated coordination, teamset {teamset_id} ===")
    print(f"{'Games':>8s}  {'WindowMean':>10s}  {'CumulativeMean':>14s}  {'LastScore':>9s}")
    print("-" * 52)

    for game_idx in range(n_games):
        random.seed(base_seed + teamset_id * 100000 + game_idx + 1)

        players = [adaptive] + _clone_fixed_teammates(fixed_teammates)
        score = Game(players, log=NullStream()).run()
        scores.append(score)

        if (game_idx + 1) % window == 0 or game_idx == n_games - 1:
            recent = scores[-window:]
            print(
                f"{game_idx + 1:8d}  "
                f"{np.mean(recent):10.2f}  "
                f"{np.mean(scores):14.2f}  "
                f"{score:9d}"
            )

    arr = np.asarray(scores, dtype=float)
    print("-" * 52)
    print(
        f"Overall mean={arr.mean():.2f}, "
        f"std={arr.std(ddof=1):.2f}, "
        f"min={int(arr.min())}, max={int(arr.max())}"
    )

    return scores


def run_adhoc_switch_table(n_games=300, base_seed=None, train_teamset=1, test_teamset=2, switch_after=None):
    base_seed = make_base_seed(base_seed)

    if switch_after is None:
        switch_after = max(1, n_games // 2)

    switch_after = int(max(1, min(switch_after, n_games - 1)))
    test_games = n_games - switch_after

    teammate_sets = make_teammate_sets(base_seed)
    source_teammates = teammate_sets[(train_teamset - 1) % len(teammate_sets)]
    target_teammates = teammate_sets[(test_teamset - 1) % len(teammate_sets)]

    fixed_source = _make_fixed_realistic_teammates(source_teammates)
    fixed_target = _make_fixed_realistic_teammates(target_teammates)

    adaptive_carry = AdaptivePlayer(names[0], 0, 4)
    adaptive_reset = AdaptivePlayer(names[0], 0, 4)

    pre_scores = []
    carry_scores = []
    reset_scores = []
    full_scores = []

    print(f"Base seed: {base_seed}")
    print(f"\n=== Ad-hoc Partner Switch: train set{train_teamset} -> test set{test_teamset} ===")
    print(f"Train games: {switch_after}; Test games: {test_games}")

    # Phase 1: learn with source teammates.
    for game_idx in range(switch_after):
        seed = base_seed + train_teamset * 100000 + game_idx + 1
        random.seed(seed)

        players = [adaptive_carry] + _clone_fixed_teammates(fixed_source)
        pre_scores.append(Game(players, log=NullStream()).run())

    # Phase 2: switch to unseen target teammates.
    for test_idx in range(test_games):
        seed = base_seed + test_teamset * 100000 + test_idx + 1

        random.seed(seed)
        players_carry = [adaptive_carry] + _clone_fixed_teammates(fixed_target)
        carry_scores.append(Game(players_carry, log=NullStream()).run())

        random.seed(seed)
        players_reset = [adaptive_reset] + _clone_fixed_teammates(fixed_target)
        reset_scores.append(Game(players_reset, log=NullStream()).run())

        random.seed(seed)
        players_full = [make_player("full", 0, 4)] + _clone_fixed_teammates(fixed_target)
        full_scores.append(Game(players_full, log=NullStream()).run())

    st_carry_full = paired_stats(carry_scores, full_scores)
    st_carry_reset = paired_stats(carry_scores, reset_scores)
    st_reset_full = paired_stats(reset_scores, full_scores)

    rows = [
        {"label": "Carry-Full", **st_carry_full},
        {"label": "Carry-Reset", **st_carry_reset},
        {"label": "Reset-Full", **st_reset_full},
    ]

    add_multiple_testing_corrections(rows)

    print("\nPhase means:")
    print(f"  Source adaptation mean: {np.mean(pre_scores):.2f}")
    print(f"  Target Full mean:       {np.mean(full_scores):.2f}")
    print(f"  Target AdaptiveReset:   {np.mean(reset_scores):.2f}")
    print(f"  Target AdaptiveCarry:   {np.mean(carry_scores):.2f}")

    print("\nTarget-phase paired statistics:")
    print(
        f"{'Contrast':12s}  {'Diff':>8s}  {'SE':>7s}  "
        f"{'95% CI':>19s}  {'p':>11s}  {'pFDR':>11s}  "
        f"{'dz':>7s}  {'Win%':>7s}"
    )
    print("-" * 98)

    for row in rows:
        ci = f"[{row['ci_low']:+.2f},{row['ci_high']:+.2f}]"
        print(
            f"{row['label']:12s}  "
            f"{row['mean_diff']:+8.2f}  "
            f"{row['se']:7.3f}  "
            f"{ci:>19s}  "
            f"p={row['p_value']:.4g} {row['sig']:>3s}  "
            f"q={row['p_fdr']:.4g} {row['sig_fdr']:>3s}  "
            f"{row['effect_dz']:7.3f}  "
            f"{100.0 * row['win_rate']:6.1f}%"
        )

    window = max(5, min(25, test_games // 5 if test_games >= 25 else test_games))

    print("\nTarget-phase adaptation curve:")
    print(
        f"{'GamesAfterSwitch':>16s}  "
        f"{'CarryWinMean':>12s}  "
        f"{'ResetWinMean':>12s}  "
        f"{'FullWinMean':>11s}"
    )
    print("-" * 62)

    for end in range(window, test_games + 1, window):
        cs = carry_scores[end - window:end]
        rs = reset_scores[end - window:end]
        fs = full_scores[end - window:end]
        print(
            f"{end:16d}  "
            f"{np.mean(cs):12.2f}  "
            f"{np.mean(rs):12.2f}  "
            f"{np.mean(fs):11.2f}"
        )

    if test_games % window != 0:
        end = test_games
        start = (test_games // window) * window
        cs = carry_scores[start:end]
        rs = reset_scores[start:end]
        fs = full_scores[start:end]
        print(
            f"{end:16d}  "
            f"{np.mean(cs):12.2f}  "
            f"{np.mean(rs):12.2f}  "
            f"{np.mean(fs):11.2f}"
        )

    return {
        "pre_scores": pre_scores,
        "carry_scores": carry_scores,
        "reset_scores": reset_scores,
        "full_scores": full_scores,
    }


 # ─── 实验5：学习曲线（得分随对局数的变化）────────────────────────────
def run_persistent_reset_curve(teamset_id=1, n_games=200, window=20, base_seed=None):
    """
    Persistent Adaptive vs reset Adaptive under one fixed continuous teamset.
    """
    base_seed = make_base_seed(base_seed)
    teammate_sets = make_teammate_sets(base_seed)
    teammates = teammate_sets[(teamset_id - 1) % len(teammate_sets)]

    print(f"\n=== Learning Curve: continuous teamset {teamset_id} ===")
    print(f"Base seed: {base_seed}")

    adaptive_keep = AdaptivePlayer(names[0], 0, 4)
    scores_keep = []
    scores_reset = []

    for game_idx in range(n_games):
        seed = base_seed + game_idx + 1

        random.seed(seed)
        players_keep = [adaptive_keep]
        for pid, pstr in enumerate(teammates, start=1):
            players_keep.append(make_player(pstr, pid, 4))
        scores_keep.append(Game(players_keep, log=NullStream()).run())

        random.seed(seed)
        players_reset = [AdaptivePlayer(names[0], 0, 4)]
        for pid, pstr in enumerate(teammates, start=1):
            players_reset.append(make_player(pstr, pid, 4))
        scores_reset.append(Game(players_reset, log=NullStream()).run())

    print(f"  (window={window})")
    print(f"  {'games':>13s} | {'persistent':>10s} | {'reset':>10s} | {'diff':>8s}")
    print("  " + "-" * 51)

    curve_rows = []

    for i in range(0, n_games, window):
        keep_chunk = scores_keep[i:i+window]
        reset_chunk = scores_reset[i:i+window]
        keep_avg = float(np.mean(keep_chunk))
        reset_avg = float(np.mean(reset_chunk))
        start = i + 1
        end = i + len(keep_chunk)
        diff = keep_avg - reset_avg

        curve_rows.append((start, end, keep_avg, reset_avg, diff))

        print(f"  {start:3d}-{end:3d}       | "
              f"{keep_avg:10.2f} | {reset_avg:10.2f} | {diff:+8.2f}")

    # --- Save learning curve plot for the simple `curve` command ---
    try:
        import matplotlib.pyplot as plt

        xs = [end for _, end, _, _, _ in curve_rows]
        persistent_vals = [p_score for _, _, p_score, _, _ in curve_rows]
        reset_vals = [r_score for _, _, _, r_score, _ in curve_rows]
        diff_vals = [d for _, _, _, _, d in curve_rows]

        plt.figure(figsize=(8, 5))
        plt.plot(xs, persistent_vals, marker="o", label="Persistent Adaptive")
        plt.plot(xs, reset_vals, marker="o", label="Reset Adaptive")
        plt.xlabel("Games")
        plt.ylabel("Window Average Score")
        plt.title(f"Learning Curve: teamset {teamset_id}, window={window}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_name = f"learning_curve_teamset{teamset_id}_seed{base_seed}_window{window}.png"
        plt.savefig(plot_name, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"\nSaved learning curve plot: {plot_name}")

        diff_plot_name = f"learning_curve_diff_teamset{teamset_id}_seed{base_seed}_window{window}.png"
        plt.figure(figsize=(8, 4))
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.plot(xs, diff_vals, marker="o", label="Persistent - Reset")
        plt.xlabel("Games")
        plt.ylabel("Score Difference")
        plt.title(f"Persistent Advantage: teamset {teamset_id}, window={window}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(diff_plot_name, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Saved difference plot: {diff_plot_name}")

    except Exception as e:
        print(f"\nFailed to save learning curve plot: {e}")


def run_method_score_curve(n_games=2000, base_seed=None, teamset_id=1, window=100):
    import matplotlib.pyplot as plt

    base_seed = make_base_seed(base_seed)
    teammate_sets = make_teammate_sets(base_seed)
    teammates = teammate_sets[(teamset_id - 1) % len(teammate_sets)]
    fixed_teammates = _make_fixed_realistic_teammates(teammates)

    method_builders = {
        "Adaptive": lambda: AdaptivePlayer(names[0], 0, 4),
        "Full": lambda: SelfIntentionalPlayer(names[0], 0),
        "Intentional": lambda: IntentionalPlayer(names[0], 0),
        "Outer": lambda: OuterStatePlayer(names[0], 0),
        "Inner": lambda: InnerStatePlayer(names[0], 0),
        "Random": lambda: Player(names[0], 0),
    }

    print(f"Base seed: {base_seed}")
    print(f"\n=== Method Score Curves: teamset {teamset_id} ===")
    print(f"n_games={n_games}, window={window}")

    all_scores = {}

    for method_name, builder in method_builders.items():
        print(f"Running {method_name}...")
        agent0 = builder()
        scores = []

        for game_idx in range(n_games):
            game_seed = base_seed + teamset_id * 100000 + game_idx + 1
            random.seed(game_seed)
            np.random.seed(game_seed % (2**32 - 1))

            players = [agent0] + _clone_fixed_teammates(fixed_teammates)
            score = Game(players, log=NullStream()).run()
            scores.append(score)

        all_scores[method_name] = scores

    xs = []
    curve_values = {method_name: [] for method_name in method_builders}

    for i in range(0, n_games, window):
        end = min(i + window, n_games)
        xs.append(end)

        for method_name in method_builders:
            chunk = all_scores[method_name][i:end]
            avg = float(np.mean(chunk))
            curve_values[method_name].append(avg)

    csv_name = (
        f"method_score_curve_teamset{teamset_id}_"
        f"seed{base_seed}_window{window}.csv"
    )

    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["games"] + list(method_builders.keys()))

        for idx, x in enumerate(xs):
            writer.writerow(
                [x] + [curve_values[m][idx] for m in method_builders]
            )

    plt.figure(figsize=(10, 6))

    for method_name in method_builders:
        plt.plot(
            xs,
            curve_values[method_name],
            marker="o",
            linewidth=2.2 if method_name == "Adaptive" else 1.5,
            label=method_name,
        )

    plt.xlabel("Games")
    plt.ylabel("Window Average Score")
    plt.title(f"Method Score Curves (teamset {teamset_id}, window={window})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_name = (
        f"method_score_curve_teamset{teamset_id}_"
        f"seed{base_seed}_window{window}.png"
    )

    plt.savefig(plot_name, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"\nSaved method score curve: {plot_name}")
    print(f"Saved curve data: {csv_name}")

    # Clean version without Random
    plt.figure(figsize=(10, 6))

    for method_name in ["Adaptive", "Full", "Intentional", "Outer", "Inner"]:
        plt.plot(
            xs,
            curve_values[method_name],
            marker="o",
            linewidth=2.4 if method_name == "Adaptive" else 1.6,
            label=method_name,
        )

    plt.xlabel("Games")
    plt.ylabel("Window Average Score")
    plt.title(f"Method Score Curves without Random (teamset {teamset_id})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    clean_plot_name = (
        f"method_score_curve_no_random_teamset{teamset_id}_"
        f"seed{base_seed}_window{window}.png"
    )

    plt.savefig(clean_plot_name, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Saved clean method score curve: {clean_plot_name}")

    return all_scores


def run_belief_convergence_curve(n_games=2000, base_seed=None, teamset_id=1, record_every=20):
    import matplotlib.pyplot as plt

    base_seed = make_base_seed(base_seed)
    teammate_sets = make_teammate_sets(base_seed)
    teammates = teammate_sets[(teamset_id - 1) % len(teammate_sets)]
    fixed_teammates = _make_fixed_realistic_teammates(teammates)

    adaptive = AdaptivePlayer(names[0], 0, 4)

    teammate_ids = [p.pnr for p in fixed_teammates]

    true_play = {p.pnr: p.giving.play_bias for p in fixed_teammates}
    true_color = {p.pnr: p.giving.color_bias for p in fixed_teammates}

    history = {
        pid: {
            "games": [],
            "obs": [],
            "play_bias": [],
            "color_bias": [],
            "uncertainty": [],
        }
        for pid in teammate_ids
    }

    print(f"Base seed: {base_seed}")
    print(f"\n=== Belief Convergence Curve: teamset {teamset_id} ===")

    print("True teammate giving styles:")
    for p in fixed_teammates:
        print(
            f"  P{p.pnr}: "
            f"play_bias={p.giving.play_bias:.3f}, "
            f"color_bias={p.giving.color_bias:.3f}"
        )

    for game_idx in range(n_games):
        game_seed = base_seed + teamset_id * 100000 + game_idx + 1

        random.seed(game_seed)
        np.random.seed(game_seed % (2**32 - 1))

        players = [adaptive] + _clone_fixed_teammates(fixed_teammates)

        Game(players, log=NullStream()).run()

        if (game_idx + 1) % record_every == 0 or game_idx == n_games - 1:
            for pid in teammate_ids:
                adaptive._ensure_giving_feature_model(pid)

                model = adaptive.giving_feature_models[pid]
                est = model.estimate()

                history[pid]["games"].append(game_idx + 1)
                history[pid]["obs"].append(model.effective_sample_size())
                history[pid]["play_bias"].append(est.play_bias)
                history[pid]["color_bias"].append(est.color_bias)
                history[pid]["uncertainty"].append(model.posterior_uncertainty())

    csv_name = (
        f"belief_curve_teamset{teamset_id}_"
        f"seed{base_seed}.csv"
    )

    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "game",
            "player",
            "obs",
            "est_play_bias",
            "true_play_bias",
            "est_color_bias",
            "true_color_bias",
            "posterior_uncertainty",
        ])

        for pid in teammate_ids:
            for i, game in enumerate(history[pid]["games"]):
                writer.writerow([
                    game,
                    pid,
                    history[pid]["obs"][i],
                    history[pid]["play_bias"][i],
                    true_play[pid],
                    history[pid]["color_bias"][i],
                    true_color[pid],
                    history[pid]["uncertainty"][i],
                ])

    # --- plot play_bias ---
    plt.figure(figsize=(10, 6))

    color_map = {
        1: "C0",
        2: "C1",
        3: "C2",
    }

    for pid in teammate_ids:
        c = color_map.get(pid, None)
        plt.plot(
            history[pid]["games"],
            history[pid]["play_bias"],
            marker="o",
            color=c,
            label=f"P{pid} estimated",
        )

        plt.axhline(
            true_play[pid],
            color=c,
            linestyle="--",
            linewidth=1.5,
            label=f"P{pid} true",
        )

    plt.xlabel("Games")
    plt.ylabel("play_bias")
    plt.title(f"Belief Convergence: play_bias (teamset {teamset_id})")
    plt.ylim(-0.05, 1.05)
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    play_plot = (
        f"belief_curve_play_bias_teamset{teamset_id}_"
        f"seed{base_seed}.png"
    )

    plt.savefig(play_plot, dpi=220, bbox_inches="tight")
    plt.close()

    # --- plot color_bias ---
    plt.figure(figsize=(10, 6))

    color_map = {
        1: "C0",
        2: "C1",
        3: "C2",
    }

    for pid in teammate_ids:
        c = color_map.get(pid, None)
        plt.plot(
            history[pid]["games"],
            history[pid]["color_bias"],
            marker="o",
            color=c,
            label=f"P{pid} estimated",
        )

        plt.axhline(
            true_color[pid],
            color=c,
            linestyle="--",
            linewidth=1.5,
            label=f"P{pid} true",
        )

    plt.xlabel("Games")
    plt.ylabel("color_bias")
    plt.title(f"Belief Convergence: color_bias (teamset {teamset_id})")
    plt.ylim(-0.05, 1.05)
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    color_plot = (
        f"belief_curve_color_bias_teamset{teamset_id}_"
        f"seed{base_seed}.png"
    )

    plt.savefig(color_plot, dpi=220, bbox_inches="tight")
    plt.close()

    # --- plot uncertainty ---
    plt.figure(figsize=(10, 5))

    color_map = {
        1: "C0",
        2: "C1",
        3: "C2",
    }

    for pid in teammate_ids:
        c = color_map.get(pid, None)
        plt.plot(
            history[pid]["games"],
            history[pid]["uncertainty"],
            marker="o",
            color=c,
            label=f"P{pid}",
        )

    plt.xlabel("Games")
    plt.ylabel("Posterior Uncertainty")
    plt.title(f"Posterior Uncertainty over Games (teamset {teamset_id})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    unc_plot = (
        f"belief_curve_uncertainty_teamset{teamset_id}_"
        f"seed{base_seed}.png"
    )

    plt.savefig(unc_plot, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Saved: {play_plot}")
    print(f"Saved: {color_plot}")
    print(f"Saved: {unc_plot}")
    print(f"Saved data: {csv_name}")

    return history

# ─── 实验6：消融实验表（persistent vs reset vs baseline）────────────────────────────

def run_ablation_table(n_games=300, base_seed=None):
    """
    Persistent Adaptive vs reset Adaptive vs Full/SelfIntentional under continuous teammates.
    """
    base_seed = make_base_seed(base_seed)
    print(f"Base seed: {base_seed}")
    print("\n=== Ablation Table: Continuous Teammates ===")
    print(f"{'TeamSet':8s}  {'Persistent':>10s}  {'Reset':>10s}  {'Full':>12s}  {'P-Reset':>8s}  {'SE':>7s}  {'95% CI':>19s}  {'p':>11s}  {'pFDR':>11s} {'dz':>7s}  {'Win%':>7s}  {'P-Base':>8s}  {'Giving':>22s}")
    print("-" * 160)
    rows = []
    for set_idx, teammates in enumerate(make_teammate_sets(base_seed), start=1):
        scores_persistent = []
        scores_reset = []
        scores_base = []
        adaptive_keep = AdaptivePlayer(names[0], 0, 4)

        for game_idx in range(n_games):
            seed = base_seed + game_idx + 1

            random.seed(seed)
            players_keep = [adaptive_keep]
            for pid, pstr in enumerate(teammates, start=1):
                players_keep.append(make_player(pstr, pid, 4))
            scores_persistent.append(Game(players_keep, log=NullStream()).run())

            random.seed(seed)
            players_reset = [AdaptivePlayer(names[0], 0, 4)]
            for pid, pstr in enumerate(teammates, start=1):
                players_reset.append(make_player(pstr, pid, 4))
            scores_reset.append(Game(players_reset, log=NullStream()).run())

            random.seed(seed)
            players_base = build_team("full", teammates)
            scores_base.append(Game(players_base, log=NullStream()).run())

        p = np.mean(scores_persistent)
        r = np.mean(scores_reset)
        b = np.mean(scores_base)

        inferred_parts = []
        for pid in range(1, 4):
            if pid in adaptive_keep.giving_feature_models:
                est = adaptive_keep.giving_feature_models[pid].estimate()
                inferred_parts.append(f"P{pid}:p{est.play_bias:.2f}/c{est.color_bias:.2f}")
            else:
                inferred_parts.append(f"P{pid}:NONE")

        inferred = ",".join(inferred_parts)

        st_reset = paired_stats(scores_persistent, scores_reset)
        ci = f"[{st_reset['ci_low']:+.2f},{st_reset['ci_high']:+.2f}]"
        rows.append({
            "label": f"set{set_idx:<4d}",
            "persistent": p,
            "reset": r,
            "base": b,
            "ci": ci,
            "p_base_diff": p - b,
            "inferred": inferred,
            **st_reset,
        })
    add_multiple_testing_corrections(rows)

    for row in rows:
        print(
            f"{row['label']:8s}  {row['persistent']:10.2f}  {row['reset']:10.2f}  {row['base']:12.2f}  "
            f"{row['mean_diff']:+8.2f}  {row['se']:7.3f}  {row['ci']:>19s}  "
            f"p={row['p_value']:.4g} {row['sig']:>3s}  "
            f"q={row['p_fdr']:.4g} {row['sig_fdr']:>3s}  "
            f"{row['effect_dz']:7.3f}  {100.0 * row['win_rate']:6.1f}%  "
            f"{row['p_base_diff']:+8.2f}  {row['inferred']:>22s}"
        )

def run_baseline_table(n_games=300, base_seed=None):
    """
    AdaptivePlayer vs original source-code baselines under continuous teammates.
    """
    base_seed = make_base_seed(base_seed)
    print(f"Base seed: {base_seed}")
    baselines = ["random", "inner", "outer", "intentional", "full"]

    print("\n=== Baseline Table: Continuous Teammate Distribution ===")
    header = f"{'TeamSet':8s}  {'Adaptive':>9s}"
    for b in baselines:
        header += f"  {b:>11s}"
    header += f"  {'A-Full':>8s}  {'SE':>7s}  {'95% CI':>19s}  {'Full p':>11s}  {'dz':>7s}  {'Win%':>7s}"
    print(header)
    print("-" * len(header))

    all_adaptive = []
    all_baselines = {b: [] for b in baselines}

    for set_idx, teammates in enumerate(make_teammate_sets(base_seed), start=1):
        scores_adaptive = []
        adaptive = AdaptivePlayer(names[0], 0, 4)

        for game_idx in range(n_games):
            random.seed(base_seed + game_idx + 1)
            players_adaptive = [adaptive]
            for pid, pstr in enumerate(teammates, start=1):
                players_adaptive.append(make_player(pstr, pid, 4))
            scores_adaptive.append(Game(players_adaptive, log=NullStream()).run())

        all_adaptive.extend(scores_adaptive)
        row = f"set{set_idx:<4d}  {np.mean(scores_adaptive):9.2f}"
        baseline_score_map = {}

        for baseline_name in baselines:
            scores_baseline = []
            for game_idx in range(n_games):
                random.seed(base_seed + game_idx + 1)
                players_baseline = build_team(baseline_name, teammates)
                scores_baseline.append(Game(players_baseline, log=NullStream()).run())
            all_baselines[baseline_name].extend(scores_baseline)
            baseline_score_map[baseline_name] = scores_baseline
            row += f"  {np.mean(scores_baseline):11.2f}"

        if "full" in baseline_score_map:
            st_full = paired_stats(scores_adaptive, baseline_score_map["full"])
            ci = f"[{st_full['ci_low']:+.2f},{st_full['ci_high']:+.2f}]"
            row += (
                f"  {st_full['mean_diff']:+8.2f}  {st_full['se']:7.3f}  {ci:>19s}  "
                f"p={st_full['p_value']:.4g} {st_full['sig']:>3s}  {st_full['effect_dz']:7.3f}  "
                f"{100.0 * st_full['win_rate']:6.1f}%"
            )
        print(row)

    print("-" * len(header))
    row = f"{'ALL':8s}  {np.mean(all_adaptive):9.2f}"
    for b in baselines:
        row += f"  {np.mean(all_baselines[b]):11.2f}"
    if "full" in all_baselines:
        st_full = paired_stats(all_adaptive, all_baselines["full"])
        ci = f"[{st_full['ci_low']:+.2f},{st_full['ci_high']:+.2f}]"
        row += (
            f"  {st_full['mean_diff']:+8.2f}  {st_full['se']:7.3f}  {ci:>19s}  "
            f"p={st_full['p_value']:.4g} {st_full['sig']:>3s}  {st_full['effect_dz']:7.3f}  "
            f"{100.0 * st_full['win_rate']:6.1f}%"
        )
    print(row)



# ─── 实验7：详细学习曲线与输出 ─────────────────────────────────────────────

def cumulative_average(values):
    """Cumulative average: avg(values[0:i+1]). Main learning curve."""
    result = []
    running_sum = 0.0
    for i, v in enumerate(values):
        running_sum += float(v)
        result.append(running_sum / (i + 1))
    return result


def rolling_average(values, window=20):
    """
    True rolling average.
    The first window-1 points are NaN because there is not enough data yet.
    """
    result = []
    for i in range(len(values)):
        if i < window - 1:
            result.append(np.nan)
        else:
            start = i - window + 1
            result.append(float(np.mean(values[start:i + 1])))
    return result


def first_window_baseline(values, window=20):
    """Horizontal baseline: average score over the first `window` games."""
    actual_window = min(window, len(values))
    return float(np.mean(values[:actual_window]))


def summarize_scores(label, scores):
    arr = np.array(scores, dtype=float)
    return {
        "label": label,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": int(arr.min()),
        "max": int(arr.max()),
    }


def run_detailed_curve(teamset_id=1, n_games=300, window=20, out_dir="results4p", base_seed=None):
    """
    Detailed continuous learning curve.
    """
    base_seed = make_base_seed(base_seed)
    teammate_sets = make_teammate_sets(base_seed)
    teammates = teammate_sets[(teamset_id - 1) % len(teammate_sets)]

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"curve_teamset{teamset_id}_{n_games}_seed{base_seed}.csv")
    summary_path = os.path.join(out_dir, f"summary_teamset{teamset_id}_{n_games}_seed{base_seed}.csv")
    png_path = os.path.join(out_dir, f"curve_teamset{teamset_id}_{n_games}_seed{base_seed}.png")

    adaptive_keep = AdaptivePlayer(names[0], 0, 4)

    rows = []
    scores_keep = []
    scores_reset = []
    scores_full = []
    scores_intentional = []

    for game_idx in range(n_games):
        seed = base_seed + game_idx + 1

        random.seed(seed)
        players_keep = [adaptive_keep]
        for pid, pstr in enumerate(teammates, start=1):
            players_keep.append(make_player(pstr, pid, 4))
        s_keep = Game(players_keep, log=NullStream()).run()
        scores_keep.append(s_keep)

        random.seed(seed)
        players_reset = [AdaptivePlayer(names[0], 0, 4)]
        for pid, pstr in enumerate(teammates, start=1):
            players_reset.append(make_player(pstr, pid, 4))
        s_reset = Game(players_reset, log=NullStream()).run()
        scores_reset.append(s_reset)

        random.seed(seed)
        players_full = build_team("full", teammates)
        s_full = Game(players_full, log=NullStream()).run()
        scores_full.append(s_full)

        random.seed(seed)
        players_int = build_team("intentional", teammates)
        s_int = Game(players_int, log=NullStream()).run()
        scores_intentional.append(s_int)

        row = {
            "game": game_idx + 1,
            "seed": seed,
            "teamset": teamset_id,
            "adaptive_persistent": s_keep,
            "adaptive_reset": s_reset,
            "full": s_full,
            "intentional": s_int,
        }

        for pid in range(1, 4):
            if pid in adaptive_keep.response_models:
                rm = adaptive_keep.response_models[pid]
                row[f"p{pid}_resp_obs"] = rm.observation_count
                row[f"p{pid}_p_play_num"] = rm.prob(HINT_NUMBER, PLAY)
                row[f"p{pid}_p_disc_num"] = rm.prob(HINT_NUMBER, DISCARD)
                row[f"p{pid}_p_play_color"] = rm.prob(HINT_COLOR, PLAY)
                row[f"p{pid}_p_disc_color"] = rm.prob(HINT_COLOR, DISCARD)
            else:
                row[f"p{pid}_resp_obs"] = 0
                row[f"p{pid}_p_play_num"] = 1 / 3
                row[f"p{pid}_p_disc_num"] = 1 / 3
                row[f"p{pid}_p_play_color"] = 1 / 3
                row[f"p{pid}_p_disc_color"] = 1 / 3

            if pid in adaptive_keep.giving_feature_models:
                gm = adaptive_keep.giving_feature_models[pid]
                est = gm.estimate()
                row[f"p{pid}_giving_obs"] = gm.n
                row[f"p{pid}_giving_play_bias"] = est.play_bias
                row[f"p{pid}_giving_color_bias"] = est.color_bias
                row[f"p{pid}_giving_conservatism"] = est.conservatism
                row[f"p{pid}_giving_coverage"] = est.coverage_preference
            else:
                row[f"p{pid}_giving_obs"] = 0
                row[f"p{pid}_giving_play_bias"] = 0.5
                row[f"p{pid}_giving_color_bias"] = 0.5
                row[f"p{pid}_giving_conservatism"] = 4.0
                row[f"p{pid}_giving_coverage"] = 0.5

        rows.append(row)

    cum_keep = cumulative_average(scores_keep)
    cum_reset = cumulative_average(scores_reset)
    cum_full = cumulative_average(scores_full)
    cum_int = cumulative_average(scores_intentional)

    roll_keep = rolling_average(scores_keep, window)
    roll_reset = rolling_average(scores_reset, window)
    roll_full = rolling_average(scores_full, window)
    roll_int = rolling_average(scores_intentional, window)

    base_keep = first_window_baseline(scores_keep, window)
    base_reset = first_window_baseline(scores_reset, window)
    base_full = first_window_baseline(scores_full, window)
    base_int = first_window_baseline(scores_intentional, window)

    for i, row in enumerate(rows):
        row["cum_adaptive_persistent"] = cum_keep[i]
        row["cum_adaptive_reset"] = cum_reset[i]
        row["cum_full"] = cum_full[i]
        row["cum_intentional"] = cum_int[i]

        row[f"roll{window}_adaptive_persistent"] = roll_keep[i]
        row[f"roll{window}_adaptive_reset"] = roll_reset[i]
        row[f"roll{window}_full"] = roll_full[i]
        row[f"roll{window}_intentional"] = roll_int[i]

        row[f"first{window}_adaptive_persistent"] = base_keep
        row[f"first{window}_adaptive_reset"] = base_reset
        row[f"first{window}_full"] = base_full
        row[f"first{window}_intentional"] = base_int

        row["diff_persistent_reset"] = scores_keep[i] - scores_reset[i]
        row["diff_persistent_full"] = scores_keep[i] - scores_full[i]
        row["diff_persistent_intentional"] = scores_keep[i] - scores_intentional[i]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = [
        summarize_scores("adaptive_persistent", scores_keep),
        summarize_scores("adaptive_reset", scores_reset),
        summarize_scores("full", scores_full),
        summarize_scores("intentional", scores_intentional),
    ]

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "mean", "std", "min", "max"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n=== Detailed Curve: teamset {teamset_id} ({n_games} games, window={window}) ===")
    print(f"Base seed: {base_seed}")
    for r in summary_rows:
        print(f"  {r['label']:20s} mean={r['mean']:.2f} std={r['std']:.2f} range={r['min']}-{r['max']}")

    print(f"  first-{window} baselines:")
    print(f"    adaptive_persistent={base_keep:.2f}, adaptive_reset={base_reset:.2f}, "
          f"full={base_full:.2f}, intentional={base_int:.2f}")

    print(f"  saved CSV: {csv_path}")
    print(f"  saved summary: {summary_path}")

    try:
        import matplotlib.pyplot as plt
        xs = list(range(1, n_games + 1))
        plt.figure(figsize=(11, 6.5))

        # Main learning curves: cumulative averages.
        plt.plot(xs, cum_keep, label="Adaptive persistent cumulative")
        plt.plot(xs, cum_reset, label="Adaptive reset cumulative")
        plt.plot(xs, cum_full, label="Full baseline cumulative")
        plt.plot(xs, cum_int, label="Intentional baseline cumulative")

        # Local trend: true rolling averages.
        plt.plot(xs, roll_keep, linestyle="--", alpha=0.65,
                 label=f"Adaptive persistent rolling-{window}")
        plt.plot(xs, roll_reset, linestyle="--", alpha=0.65,
                 label=f"Adaptive reset rolling-{window}")
        plt.plot(xs, roll_full, linestyle="--", alpha=0.65,
                 label=f"Full rolling-{window}")
        plt.plot(xs, roll_int, linestyle="--", alpha=0.65,
                 label=f"Intentional rolling-{window}")

        # First-window reference baselines.
        plt.axhline(base_keep, linestyle=":", alpha=0.8,
                    label=f"Adaptive first-{window} baseline")
        plt.axhline(base_reset, linestyle=":", alpha=0.8,
                    label=f"Reset first-{window} baseline")
        plt.axhline(base_full, linestyle=":", alpha=0.8,
                    label=f"Full first-{window} baseline")
        plt.axhline(base_int, linestyle=":", alpha=0.8,
                    label=f"Intentional first-{window} baseline")

        plt.xlabel("Game")
        plt.ylabel("Average score")
        plt.title(f"4-player Hanabi continuous learning curve: teamset {teamset_id}")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(png_path, dpi=160)
        plt.close()
        print(f"  saved figure: {png_path}")
    except Exception as e:
        print(f"  matplotlib figure skipped: {e}")

    return rows

def run_all_detailed_curves(n_games=300, window=20, out_dir="results4p", base_seed=None):
    base_seed = make_base_seed(base_seed)
    for teamset_id in range(1, len(make_teammate_sets(base_seed)) + 1):
        run_detailed_curve(teamset_id, n_games=n_games, window=window, out_dir=out_dir, base_seed=base_seed)

# ═══════════════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════════════

def main(args=None):
    if args is None:
        args = configure_teammate_styles_from_argv()
    else:
        old_argv = sys.argv
        sys.argv = [old_argv[0]] + list(args)
        args = configure_teammate_styles_from_argv()
        sys.argv = old_argv

    if not args or args[0] in {"help", "-h", "--help"}:
        print("Usage:")
        print("  python hanabi_adaptive4p.py converge [TEAMSET_ID] [N_GAMES] [BASE_SEED]")
        print("  python hanabi_adaptive4p.py curve    [TEAMSET_ID] [N_GAMES] [BASE_SEED]")
        print("  python hanabi_adaptive4pp.py compare [N_GAMES] [BASE_SEED]")
        print("  python hanabi_adaptive4pp.py curve [N_GAMES] [BASE_SEED] [TEAMSET_ID] [WINDOW]")
        print("  python hanabi_adaptive4pp.py adaptive_curve [N_GAMES] [BASE_SEED] [TEAMSET_ID] [WINDOW]")
        print("  python hanabi_adaptive4pp.py adhoc [N_GAMES] [BASE_SEED] [TRAIN_TEAMSET] [TEST_TEAMSET] [SWITCH_AFTER]")
        print("  python hanabi_adaptive4p.py ablation [N_GAMES] [BASE_SEED]")
        print("  python hanabi_adaptive4p.py baselines [N_GAMES] [BASE_SEED]")
        print("  python hanabi_adaptive4p.py realistic [N_GAMES] [BASE_SEED]")
        print("  python hanabi_adaptive4p.py detail   [TEAMSET_ID] [N_GAMES] [WINDOW] [BASE_SEED]")
        print("  python hanabi_adaptive4p.py detail_all [N_GAMES] [WINDOW] [BASE_SEED]")
        print("  python hanabi_adaptive4pp.py meta_ablation [N_GAMES] [BASE_SEED]")
        print("  python hanabi_adaptive4p.py demo")
        print("")
        print("Style options:")
        print("  --style-mode manual --style-seed 12345")
        print("      All RealisticStylePlayer teammates are randomly selected from SET_A ... SET_F.")
        print("  --style-mode realistic --style-seed 12345")
        print("      All RealisticStylePlayer teammates are generated by sample_teammate_style(seed).")
        print("  --style-mode hybrid --manual-ratio 0.5 --style-seed 12345")
        print("      Each teammate is manual with probability manual_ratio, otherwise realistic.")
        print("  --style-mode fixed --style-name SET_A_PLAY_DIRECTOR")
        print("      Optional diagnostic mode: all teammates use one chosen predefined style.")
        print("")
        print("Examples:")
        print("  python hanabi_adaptive4pp.py compare 300 12345 --style-mode manual --style-seed 12345")
        print("  python hanabi_adaptive4pp.py compare 300 12345 --style-mode realistic --style-seed 12345")
        print("  python hanabi_adaptive4pp.py compare 300 12345 --style-mode hybrid --manual-ratio 0.5 --style-seed 12345")
        return

    args = list(args)
    if args[0] == "demo":
        run_demo()
    elif args[0] == "converge":
        teamset = int(args[1]) if len(args) > 1 else 1
        n = int(args[2]) if len(args) > 2 else 30
        base_seed = int(args[3]) if len(args) > 3 else None
        run_belief_convergence(teamset, n, base_seed=base_seed)
    elif args[0] == "compare":
        n = int(args[1]) if len(args) > 1 else 300
        base_seed = int(args[2]) if len(args) > 2 else None
        run_compare_table(n_games=n, base_seed=base_seed)
    elif args[0] == "ablation":
        n = int(args[1]) if len(args) > 1 else 300
        base_seed = int(args[2]) if len(args) > 2 else None
        run_ablation_table(n, base_seed=base_seed)
    elif args[0] == "baselines":
        n = int(args[1]) if len(args) > 1 else 300
        base_seed = int(args[2]) if len(args) > 2 else None
        run_baseline_table(n, base_seed=base_seed)
    elif args[0] == "realistic":
        n = int(args[1]) if len(args) > 1 else 300
        base_seed = int(args[2]) if len(args) > 2 else None
        run_compare_table(n_games=n, base_seed=base_seed)
    elif args[0] == "detail":
        teamset = int(args[1]) if len(args) > 1 else 1
        n = int(args[2]) if len(args) > 2 else 300
        window = int(args[3]) if len(args) > 3 else 20
        base_seed = int(args[4]) if len(args) > 4 else None
        run_detailed_curve(teamset, n_games=n, window=window, base_seed=base_seed)
    elif args[0] == "detail_all":
        n = int(args[1]) if len(args) > 1 else 300
        window = int(args[2]) if len(args) > 2 else 20
        base_seed = int(args[3]) if len(args) > 3 else None
        run_all_detailed_curves(n_games=n, window=window, base_seed=base_seed)
    elif args[0] == "curve":
        n = int(args[1]) if len(args) > 1 else 200
        base_seed = int(args[2]) if len(args) > 2 else None
        teamset_id = int(args[3]) if len(args) > 3 else 1
        window = int(args[4]) if len(args) > 4 else 20
        run_persistent_reset_curve(
            n_games=n,
            base_seed=base_seed,
            teamset_id=teamset_id,
            window=window,
        )

    elif args[0] == "adaptive_curve":
        n = int(args[1]) if len(args) > 1 else 200
        base_seed = int(args[2]) if len(args) > 2 else None
        teamset_id = int(args[3]) if len(args) > 3 else 1
        window = int(args[4]) if len(args) > 4 else 20
        run_adaptive_score_curve(
            n_games=n,
            base_seed=base_seed,
            teamset_id=teamset_id,
            window=window,
        )

    elif args[0] == "method_curve":
        n = int(args[1]) if len(args) > 1 else 2000
        base_seed = int(args[2]) if len(args) > 2 else None
        teamset_id = int(args[3]) if len(args) > 3 else 1
        window = int(args[4]) if len(args) > 4 else 100

        run_method_score_curve(
            n_games=n,
            base_seed=base_seed,
            teamset_id=teamset_id,
            window=window,
        )
    elif args[0] == "belief_curve":
        n = int(args[1]) if len(args) > 1 else 2000
        base_seed = int(args[2]) if len(args) > 2 else None
        teamset_id = int(args[3]) if len(args) > 3 else 1
        record_every = int(args[4]) if len(args) > 4 else 20

        run_belief_convergence_curve(
            n_games=n,
            base_seed=base_seed,
            teamset_id=teamset_id,
            record_every=record_every,
        )
    elif args[0] == "adhoc":
        n = int(args[1]) if len(args) > 1 else 300
        base_seed = int(args[2]) if len(args) > 2 else None
        train_teamset = int(args[3]) if len(args) > 3 else 1
        test_teamset = int(args[4]) if len(args) > 4 else 2
        switch_after = int(args[5]) if len(args) > 5 else None

        run_adhoc_switch_table(
            n_games=n,
            base_seed=base_seed,
            train_teamset=train_teamset,
            test_teamset=test_teamset,
            switch_after=switch_after,
        )
    elif args[0] == "meta_ablation":
        n = int(args[1]) if len(args) > 1 else 300
        base_seed = int(args[2]) if len(args) > 2 else None
        run_meta_ablation_table(n_games=n, base_seed=base_seed)
    else:
        print("Unknown command. Use 'help' for usage.")


if __name__ == "__main__":
    main()