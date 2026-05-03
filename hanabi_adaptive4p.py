import random
import sys
import copy
import time
import os
import csv
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

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
                    value = prob * (6 - rank) / (dist * dist) if cnt > 1 else (6 - rank)
                    if rank == 5: value += HINT_VALUE
                    value *= prob
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
    """Continuous receiving style: how this player tends to interpret hints."""
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


def format_style_pair(pair):
    g = pair.giving
    r = pair.receiving
    return (
        f"G(play={g.play_bias:.2f}, color={g.color_bias:.2f}, "
        f"cons={g.conservatism:.1f}, cov={g.coverage_preference:.2f}); "
        f"R(trust={r.trust_level:.2f}, play={r.playable_interpretation:.2f}, "
        f"safe={r.safe_interpretation:.2f}, tol={r.uncertainty_tolerance:.2f})"
    )




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

                # trust_level 是 confidence gate：
                # 信任低 + 信息模糊 → 不轻易行动
                ambiguity_excess = max(
                    0.0,
                    uncertainty - self.receiving.uncertainty_tolerance
                )

                confidence = self.receiving.trust_level * (1.0 - ambiguity_excess)
                confidence = max(0.0, min(1.0, confidence))

                # trust 高的人行动门槛低；trust 低的人行动门槛高
                action_threshold = 0.65 - 0.25 * self.receiving.trust_level

                play_score = (
                    confidence * self.receiving.playable_interpretation
                    if can_play else 0.0
                )

                discard_score = (
                    confidence * self.receiving.safe_interpretation
                    if can_discard else 0.0
                )

                if play_score >= action_threshold and play_score >= discard_score:
                    result = Action(PLAY, cnr=i)
                    break

                if discard_score >= action_threshold:
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
                    + 3.0 * intent_score
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

    def inform(self, action, player, game):
        if action.type in [HINT_COLOR, HINT_NUMBER] and action.pnr == self.pnr:
            self.gothint = (copy.deepcopy(action), player)

    def get_explanation(self):
        return self.explanation

class TeammateGivingFeatureModel:
    """
    Continuous giving-style estimator.
    It estimates play_bias, color_bias, conservatism, and coverage_preference.
    """
    def __init__(self, player_id):
        self.player_id = player_id
        self.n = 0
        self.play_sum = 0.0
        self.color_sum = 0.0
        self.coverage_sum = 0.0
        self.conservatism_sum = 0.0

    def update_from_hint(self, action, target_hand, board, hints_before_action):
        if action.type not in [HINT_COLOR, HINT_NUMBER] or not target_hand:
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

        discard_hits = sum(
            1 for col, num in pointed_cards
            if board[col][1] >= num
        )

        total_pointed = max(1, len(pointed_cards))

        # Estimate play tendency over all pointed cards. Future-information hints
        # should not be forced into the discard side simply because they are not
        # immediately playable/discardable.
        play_feature = play_hits / total_pointed
        color_feature = 1.0 if action.type == HINT_COLOR else 0.0
        coverage_feature = min(
            1.0,
            len(pointed_cards) / max(1, len(target_hand))
        )

        # 直接记录给 hint 时剩余 hint tokens，保持 [1,8] 尺度
        conservatism_feature = max(
            1.0,
            min(8.0, float(hints_before_action))
        )

        self.n += 1
        self.play_sum += play_feature
        self.color_sum += color_feature
        self.coverage_sum += coverage_feature
        self.conservatism_sum += conservatism_feature

    def estimate(self):
        if self.n == 0:
            return GivingStyle(
                play_bias=0.5,
                color_bias=0.5,
                conservatism=4.0,
                coverage_preference=0.5,
            )

        return GivingStyle(
            play_bias=self.play_sum / self.n,
            color_bias=self.color_sum / self.n,
            conservatism=self.conservatism_sum / self.n,
            coverage_preference=self.coverage_sum / self.n,
        )

    def __repr__(self):
        est = self.estimate()
        return (
            f"GivingFeatureModel(player={self.player_id}, obs={self.n}, "
            f"play_bias={est.play_bias:.2f}, color_bias={est.color_bias:.2f}, "
            f"conservatism={est.conservatism:.2f}, coverage={est.coverage_preference:.2f})"
        )
    
class TeammateResponseModel:
    """
    独立建模 teammate receiving / interpretation style。

    这个模型只学习：当我给某个 teammate 一个 hint 后，TA 下一步更可能如何反应。
    它不参与解释 teammate 为什么给我 hint。

    维护四个 Beta-Bernoulli 风格的倾向：
      P(play | number hint)
      P(play | color hint)
      P(discard | number hint)
      P(discard | color hint)
    """
    def __init__(self, player_id):
        self.player_id = player_id
        self.counts = {
            HINT_NUMBER: {PLAY: 1.0, DISCARD: 1.0, None: 1.0},
            HINT_COLOR:  {PLAY: 1.0, DISCARD: 1.0, None: 1.0},
        }
        self.observation_count = 0

    def update_from_response(self, hint_action, response_action, hinted_hand, board_at_hint):
        if hint_action.type not in [HINT_COLOR, HINT_NUMBER]:
            return
        if response_action.type not in [PLAY, DISCARD]:
            return
        if response_action.cnr is None or response_action.cnr >= len(hinted_hand):
            return

        card = hinted_hand[response_action.cnr]
        col, num = card
        pointed = False
        if hint_action.type == HINT_COLOR and hint_action.col == col:
            pointed = True
        if hint_action.type == HINT_NUMBER and hint_action.num == num:
            pointed = True
        if not pointed:
            self.counts[hint_action.type][None] += 1.0
            self.observation_count += 1
            return

        if response_action.type == PLAY:
            self.counts[hint_action.type][PLAY] += 1.0
        elif response_action.type == DISCARD:
            self.counts[hint_action.type][DISCARD] += 1.0
        self.observation_count += 1

    def prob(self, hint_type, response_type):
        bucket = self.counts[hint_type]
        total = sum(bucket.values())
        return bucket[response_type] / total

    def score_hint(self, hint_action, target_hand, board):
        """
        Estimate how useful this hint is for this teammate's receiving style.
        Higher score means the target is likely to react in a useful way.
        """
        if hint_action.type not in [HINT_COLOR, HINT_NUMBER]:
            return -1e9

        p_play = self.prob(hint_action.type, PLAY)
        p_discard = self.prob(hint_action.type, DISCARD)
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
            elif board[col][1] >= num:
                score += 2.5 * p_discard
                score -= 2.0 * p_play
            elif num == 5:
                score -= 3.0 * p_discard
            else:
                score += 0.2

        if touched == 0:
            return -1e9
        return score + 0.05 * touched

    def __repr__(self):
        return (f"ResponseModel(player={self.player_id}, obs={self.observation_count}, "
                f"P(play|num)={self.prob(HINT_NUMBER, PLAY):.2f}, "
                f"P(discard|num)={self.prob(HINT_NUMBER, DISCARD):.2f}, "
                f"P(play|color)={self.prob(HINT_COLOR, PLAY):.2f}, "
                f"P(discard|color)={self.prob(HINT_COLOR, DISCARD):.2f})")


# ═══════════════════════════════════════════════════════════════════════
#
#  PART 3: 自适应玩家 (AdaptivePlayer)

class AdaptivePlayer:
    """
    主要贡献类：把 teammate modeling 拆成两个互不混用的维度。

    1. giving model:
       学 teammate 为什么给 hint。只用于解释别人给自己的 hint。

    2. response model:
       学 teammate 收到我给的 hint 后如何反应。只用于决定我该给什么 hint。

    跨游戏：两个模型都持续存活，实现 repeated interaction adaptation。
    """
    def __init__(self, name, pnr, n_players=2):
        self.name      = name
        self.pnr       = pnr
        self.n_players = n_players

        # continuous giving model：estimate how each teammate gives hints
        self.giving_feature_models: dict[int, TeammateGivingFeatureModel] = {}

        # response model：建模 teammate 如何响应我给出的 hint
        self.response_models: dict[int, TeammateResponseModel] = {}

        # 记录我刚给某个 teammate 的 hint，用于观察 TA 下一步如何响应
        self.pending_hints = {}

        # 最近一次 hint（用于 self-recognition）
        self.gothint = None
        self.last_knowledge = None
        self.last_board     = None
        self.last_trash     = None
        self.last_played    = None
        self.last_hints     = None

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


    # ── 给 hint 时：只用 response model ────────────────────────────
    def _choose_hint_for_target(self, target, nr, hands, knowledge,
                                board, hints):
        """
        给 hint 时只使用 response model。
        注意：这里不再使用 teammate 的 giving-style belief。
        giving style 和 receiving style 是两个不同维度，不能混用。
        """
        if not hands[target] or target >= len(knowledge) or not knowledge[target]:
            return None

        self._ensure_response_model(target)
        response_model = self.response_models[target]

        best_score = -1e9
        best_action = None

        candidates = []
        for c in set(col for col, _ in hands[target]):
            candidates.append(Action(HINT_COLOR, pnr=target, col=c))
        for r in set(num for _, num in hands[target]):
            candidates.append(Action(HINT_NUMBER, pnr=target, num=r))

        # Keep the original Intentional safety check: do not give hints that pretend()
        # predicts as misleading. Among safe hints, choose the one best matched to
        # this teammate's learned response model.
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
                continue
            if not ok:
                continue

            response_score = response_model.score_hint(act, hands[target], board)
            score = safe_score + response_score
            if score > best_score:
                best_score = score
                best_action = act

        if best_action is None:
            return None
        return best_action, best_score

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
            sampled_for_belief = []
            sampled_pool = update_knowledge(base_knowledge, used)

            for _ in range(160):
                h = sample_hand(sampled_pool)
                if len(h) == len(base_knowledge):
                    sampled_for_belief.append(h)

            if sampled_for_belief:
                # 用 sampled possible self-hands 更新 continuous giving model。

                # 新增：用 sampled possible self-hands 更新 continuous giving model。
                # 注意：这里没有看真实 self hand，不作弊。
                compatible_hands = []

                for h in sampled_for_belief:
                    for col, num in h:
                        if hint_act.type == HINT_COLOR and hint_act.col == col:
                            compatible_hands.append(h)
                            break
                        if hint_act.type == HINT_NUMBER and hint_act.num == num:
                            compatible_hands.append(h)
                            break

                if compatible_hands:
                    sampled_target_hand = random.choice(compatible_hands)

                    self.giving_feature_models[hinter_id].update_from_hint(
                        hint_act,
                        sampled_target_hand,
                        snapshot_board,
                        snapshot_hints
                    )

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
            if card is not None and cnt >= len(possiblehands) * 0.9 and len(possiblehands) >= 25:
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

        if discards and hints < 8 and not result:
            result = Action(DISCARD, cnr=random.choice(discards))

        # Step 3: Adaptive hint selection.
        # Compared with full/SelfIntentional, this part uses the learned response model.
        # Giving models are intentionally not used here because giving and receiving
        # styles are separate dimensions.
        
        if hints > 0 and not result:
            best_hint = None
            best_score = -1e9

            for target in range(len(hands)):
                if target == nr or not hands[target]:
                    continue
                if target >= len(knowledge) or not knowledge[target]:
                    continue

                choice = self._choose_hint_for_target(
                    target, nr, hands, knowledge, board, hints)
                if choice is None:
                    continue
                act, learned_hint_score = choice

                # Score the selected hint using both immediate usefulness and learned response score.
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
                    elif n < 5:
                        immediate += 0.3

                self._ensure_response_model(target)
                response_confidence = min(
                    1.0,
                    self.response_models[target].observation_count / 10.0
                )
                score = immediate + learned_hint_score + 0.5 * response_confidence

                if score > best_score:
                    best_score = score
                    best_hint = act

            if best_hint is not None and best_score > 0:
                result = best_hint

        if result:
            return result

        # Step 4: Same safe discard fallback as Intentional/SelfIntentional.
        scores = [pretend_discard(Action(DISCARD, cnr=i), knowledge[nr], board, trash)
                  for i in range(handsize)]
        scores.sort(key=lambda x: -x[1])
        return scores[0][0]

    # ── 信息接收：更新队友模型 ────────────────────────────────────────
    def inform(self, action, player, game):
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
        # If the target of my previous hint takes a non-card action, such as giving
        # another hint, then their next later play/discard should not be attributed
        # to my old hint. Count it as no direct response and clear the pending hint.
        if player != self.pnr and action.type in [HINT_COLOR, HINT_NUMBER]:
            if player in self.pending_hints:
                pending = self.pending_hints.pop(player)
                self._ensure_response_model(player)
                self.response_models[player].counts[pending["hint"].type][None] += 1.0
                self.response_models[player].observation_count += 1
        if action.type in [HINT_COLOR, HINT_NUMBER]:
            # 如果别人后来又给了同一个 target hint，那么 target 的下一步反应
            # 可能是由后来的 hint 触发的。为避免错误归因，丢弃我之前给该 target 的 pending hint。
            if player != self.pnr and action.pnr in self.pending_hints:
                pending = self.pending_hints.pop(action.pnr)
                self._ensure_response_model(action.pnr)
                self.response_models[action.pnr].counts[pending["hint"].type][None] += 1.0
                self.response_models[action.pnr].observation_count += 1

            # 如果这个 hint 是我给 teammate 的，记录下来，等 TA 下一步行动后更新 response model。
            if player == self.pnr and action.pnr != self.pnr:
                self.pending_hints[action.pnr] = {
                    "hint": copy.deepcopy(action),
                    "hand": list(game.hands[action.pnr]),
                    "board": copy.deepcopy(game.board),
                }
            # 若是队友给 hint，更新对应信念模型。
            # 注意：如果队友 hint 的目标是自己，则不能直接用 game.hands[self.pnr]
            # 来计算 likelihood，因为那等于偷看自己的真实手牌。
            # 所以这里先只用「队友 hint 给其他队友」的行为做风格更新；
            # hint 给自己的行为交给下面的 self-recognition 逻辑处理。
            if player != self.pnr and action.pnr != self.pnr:
                self._ensure_giving_feature_model(player)
                self.giving_feature_models[player].update_from_hint(
                    action,
                    list(game.hands[action.pnr]),
                    game.board,
                    game.hints,
                )
            # 若是对方 hint 给自己，记录以备 self-recognition
            if action.pnr == self.pnr:
                # 收到别人给自己的 hint 时，不直接用真实 self hand。
                # 后续在 _refine_own_knowledge_from_hint 中用 sampled possible hands
                # 更新 continuous giving model。
                self._ensure_giving_feature_model(player)

                self.gothint      = (action, player)
                self.last_knowledge = copy.deepcopy(game.knowledge)
                self.last_board     = copy.deepcopy(game.board)
                self.last_trash     = list(game.trash)
                self.last_played    = list(game.played)
                self.last_hints     = game.hints

    def get_explanation(self):
        return self.explanation

    def print_beliefs(self):
        print(f"\n=== {self.name} (player {self.pnr}) Continuous Giving Feature Models ===")
        for pid, model in self.giving_feature_models.items():
            print(model)

        print(f"\n=== {self.name} (player {self.pnr}) Response Models ===")
        for pid, model in self.response_models.items():
            print(model)


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

    def inform(self, action, player, game):
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
        if discards and hints < 8 and not result:
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

    def inform(self, action, player, game):
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

    def inform(self, action, player, game):
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

        if discards and hints < 8 and not result:
            result = Action(DISCARD, cnr=random.choice(discards))

        target = None
        for i in range(len(hands)):
            if i != nr and hands[i]:
                target = i
                break

        if hints > 0 and not result and target is not None:
            intentions = [None] * len(hands[target])
            othercards = trash + board
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

    def inform(self, action, player, game):
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
            self.last_knowledge = copy.deepcopy(game.knowledge)
            self.last_board = copy.deepcopy(game.board)
            self.last_trash = list(game.trash)
            self.last_played = list(game.played)


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

def sample_hand(knowledge):
    result = None
    while result is None:
        result = do_sample(knowledge)
    return result


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

    def perform(self, action):
        for p in self.players:
            p.inform(action, self.current_player, self)

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


def make_player(player_str, i, n_players=None):

    if player_str == "adaptive":
        return AdaptivePlayer(names[i], i, n_players or 4)

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
    Continuous teammate profiles used across all experiments.
    The same base_seed gives the same teamsets; different base_seeds produce
    different teammate personalities.
    """
    offset = int(base_seed) * 1000
    return [
        [f"realistic({offset + 10})", f"realistic({offset + 20})", f"realistic({offset + 30})"],
        [f"realistic({offset + 40})", f"realistic({offset + 50})", f"realistic({offset + 60})"],
        [f"realistic({offset + 70})", f"realistic({offset + 80})", f"realistic({offset + 90})"],
        [f"realistic({offset + 100})", f"realistic({offset + 110})", f"realistic({offset + 120})"],
        [f"realistic({offset + 130})", f"realistic({offset + 140})", f"realistic({offset + 150})"],
    ]


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
        player_strs = ["adaptive", "realistic(10)", "realistic(20)", "realistic(30)"]
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
    reset_beliefs=False → AdaptivePlayer 跨游戏保留模型（repeated interaction）
    reset_beliefs=True  → AdaptivePlayer 每局重置（对照组）

    重要：
    非 Adaptive 玩家每局都重建，避免 gothint / hints 等临时状态跨局泄漏。
    如果 reset_beliefs=False，只有 AdaptivePlayer 对象跨局复用。
    """
    log = NullStream()
    scores = []

    persistent_players = {}

    if not reset_beliefs:
        for i, pstr in enumerate(player_strs):
            if pstr == "adaptive":
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

        random.seed(game_idx + 1)
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


# ─── 实验4：跨风格对照表 ────────────────────────────────────────────
def run_comparison_table(n_games=300, base_seed=None):
    """
    Adaptive vs Intentional under continuous teammate distributions.
    """
    base_seed = make_base_seed(base_seed)
    print(f"Base seed: {base_seed}")
    print("\n=== Comparison Table: Continuous Teammate Distribution ===")
    print(f"{'TeamSet':8s}  {'Adaptive':>9s}  {'Intentional':>12s}  {'Difference':>10s}")
    print("-" * 48)

    for set_idx, teammates in enumerate(make_teammate_sets(base_seed), start=1):
        scores_a = []
        scores_b = []
        adaptive = AdaptivePlayer(names[0], 0, 4)

        for game_idx in range(n_games):
            seed = base_seed + game_idx + 1

            random.seed(seed)
            players_a = [adaptive]
            for pid, pstr in enumerate(teammates, start=1):
                players_a.append(make_player(pstr, pid, 4))
            scores_a.append(Game(players_a, log=NullStream()).run())

            random.seed(seed)
            players_b = build_team("intentional", teammates)
            scores_b.append(Game(players_b, log=NullStream()).run())

        avg_a = np.mean(scores_a)
        avg_b = np.mean(scores_b)
        print(f"set{set_idx:<4d}  {avg_a:9.2f}  {avg_b:12.2f}  {avg_a-avg_b:+10.2f}")


# ─── 实验5：学习曲线（得分随对局数的变化）────────────────────────────
def run_learning_curve(teamset_id=1, n_games=200, window=20, base_seed=None):
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

    for i in range(0, n_games, window):
        keep_chunk = scores_keep[i:i+window]
        reset_chunk = scores_reset[i:i+window]
        keep_avg = np.mean(keep_chunk)
        reset_avg = np.mean(reset_chunk)
        print(f"  {i+1:3d}-{i+len(keep_chunk):3d}       | "
              f"{keep_avg:10.2f} | {reset_avg:10.2f} | {keep_avg-reset_avg:+8.2f}")

# ─── 实验6：消融实验表（persistent vs reset vs baseline）────────────────────────────

def run_ablation_table(n_games=300, base_seed=None):
    """
    Persistent Adaptive vs reset Adaptive vs Intentional under continuous teammates.
    """
    base_seed = make_base_seed(base_seed)
    print(f"Base seed: {base_seed}")
    print("\n=== Ablation Table: Continuous Teammates ===")
    print(f"{'TeamSet':8s}  {'Persistent':>10s}  {'Reset':>10s}  {'Intentional':>12s}  {'P-Reset':>8s}  {'P-Base':>8s}  {'Giving':>22s}")
    print("-" * 98)

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
            players_base = build_team("intentional", teammates)
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

        print(f"set{set_idx:<4d}  {p:10.2f}  {r:10.2f}  {b:12.2f}  "
              f"{p-r:+8.2f}  {p-b:+8.2f}  {inferred:>22s}")

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

        for baseline_name in baselines:
            scores_baseline = []
            for game_idx in range(n_games):
                random.seed(base_seed + game_idx + 1)
                players_baseline = build_team(baseline_name, teammates)
                scores_baseline.append(Game(players_baseline, log=NullStream()).run())
            all_baselines[baseline_name].extend(scores_baseline)
            row += f"  {np.mean(scores_baseline):11.2f}"

        print(row)

    print("-" * len(header))
    row = f"{'ALL':8s}  {np.mean(all_adaptive):9.2f}"
    for b in baselines:
        row += f"  {np.mean(all_baselines[b]):11.2f}"
    print(row)



# ─── 实验7：详细学习曲线与输出 ─────────────────────────────────────────────

def moving_average(values, window=20):
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(values[start:i + 1])))
    return result


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

    ma_keep = moving_average(scores_keep, window)
    ma_reset = moving_average(scores_reset, window)
    ma_full = moving_average(scores_full, window)
    ma_int = moving_average(scores_intentional, window)

    for i, row in enumerate(rows):
        row[f"ma{window}_adaptive_persistent"] = ma_keep[i]
        row[f"ma{window}_adaptive_reset"] = ma_reset[i]
        row[f"ma{window}_full"] = ma_full[i]
        row[f"ma{window}_intentional"] = ma_int[i]
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
    print(f"  saved CSV: {csv_path}")
    print(f"  saved summary: {summary_path}")

    try:
        import matplotlib.pyplot as plt
        xs = list(range(1, n_games + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(xs, ma_keep, label="Adaptive persistent")
        plt.plot(xs, ma_reset, label="Adaptive reset")
        plt.plot(xs, ma_full, label="Full baseline")
        plt.plot(xs, ma_int, label="Intentional baseline")
        plt.xlabel("Game")
        plt.ylabel(f"Moving average score (window={window})")
        plt.title(f"4-player Hanabi continuous learning curve: teamset {teamset_id}")
        plt.legend()
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
        args = sys.argv[1:]

    if not args or args[0] in {"help", "-h", "--help"}:
        print("Usage:")
        print("  python hanabi_adaptive4p.py converge [TEAMSET_ID] [N_GAMES] [BASE_SEED]")
        print("  python hanabi_adaptive4p.py curve    [TEAMSET_ID] [N_GAMES] [BASE_SEED]")
        print("  python hanabi_adaptive4p.py compare  [N_GAMES] [BASE_SEED]")
        print("  python hanabi_adaptive4p.py ablation [N_GAMES] [BASE_SEED]")
        print("  python hanabi_adaptive4p.py baselines [N_GAMES] [BASE_SEED]")
        print("  python hanabi_adaptive4p.py realistic [N_GAMES] [BASE_SEED]")
        print("  python hanabi_adaptive4p.py detail   [TEAMSET_ID] [N_GAMES] [WINDOW] [BASE_SEED]")
        print("  python hanabi_adaptive4p.py detail_all [N_GAMES] [WINDOW] [BASE_SEED]")
        print("  python hanabi_adaptive4p.py demo")
        return

    args = list(args)
    if args[0] == "demo":
        run_demo()
    elif args[0] == "converge":
        teamset = int(args[1]) if len(args) > 1 else 1
        n = int(args[2]) if len(args) > 2 else 30
        base_seed = int(args[3]) if len(args) > 3 else None
        run_belief_convergence(teamset, n, base_seed=base_seed)
    elif args[0] == "curve":
        teamset = int(args[1]) if len(args) > 1 else 1
        n = int(args[2]) if len(args) > 2 else 100
        base_seed = int(args[3]) if len(args) > 3 else None
        run_learning_curve(teamset, n, base_seed=base_seed)
    elif args[0] == "compare":
        n = int(args[1]) if len(args) > 1 else 300
        base_seed = int(args[2]) if len(args) > 2 else None
        run_comparison_table(n, base_seed=base_seed)
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
        run_baseline_table(n, base_seed=base_seed)
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
    else:
        print("Unknown command. Use 'help' for usage.")


if __name__ == "__main__":
    main()