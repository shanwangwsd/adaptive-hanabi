import random
import sys
import copy
import time
import numpy as np
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
#  ████████╗███████╗ █████╗ ███╗   ███╗███╗   ███╗ █████╗ ████████╗███████╗
#     ██╔══╝██╔════╝██╔══██╗████╗ ████║████╗ ████║██╔══██╗╚══██╔══╝██╔════╝
#     ██║   █████╗  ███████║██╔████╔██║██╔████╔██║███████║   ██║   █████╗
#     ██║   ██╔══╝  ██╔══██║██║╚██╔╝██║██║╚██╔╝██║██╔══██║   ██║   ██╔══╝
#     ██║   ███████╗██║  ██║██║ ╚═╝ ██║██║ ╚═╝ ██║██║  ██║   ██║   ███████╗
#     ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝
#
#  PART 1: 参数化 Hint 风格策略 (HintStylePlayer)
#
# ═══════════════════════════════════════════════════════════════════════

class HintStyle:
    """
    描述一个玩家的 hint 行为风格。
    所有参数都是 [0,1] 的连续倾向，但我们在方案A中只用离散的几个预设配置。

    play_bias        : 倾向给 playable 牌的 hint（0=discard倾向, 1=play倾向）
    color_bias       : 倾向给颜色 hint（0=只给数字, 1=只给颜色）
    hint_threshold   : hints 数量 >= 此值才主动 hint（保守程度）
    coverage_bias    : 倾向给覆盖多张的 hint（0=精准, 1=广覆盖）
    """
    def __init__(self, name, play_bias=0.5, color_bias=0.5,
                 hint_threshold=1, coverage_bias=0.5):
        self.name           = name
        self.play_bias      = play_bias
        self.color_bias     = color_bias
        self.hint_threshold = hint_threshold
        self.coverage_bias  = coverage_bias

    def __repr__(self):
        return (f"HintStyle({self.name}, play={self.play_bias:.1f}, "
                f"color={self.color_bias:.1f}, "
                f"thresh={self.hint_threshold}, cov={self.coverage_bias:.1f})")


# ─── 离散策略空间：5 种典型风格 ───────────────────────────────────────
#
#   θ0  PLAY_NUM    : 强 play 倾向 + 偏数字 hint
#   θ1  PLAY_COL    : 强 play 倾向 + 偏颜色 hint
#   θ2  DISC_NUM    : 强 discard 倾向 + 偏数字 hint
#   θ3  DISC_COL    : 强 discard 倾向 + 偏颜色 hint
#   θ4  INFO        : 信息导向（保守, 广覆盖）
#
HINT_STYLES = [
    HintStyle("PLAY_NUM",  play_bias=0.9, color_bias=0.1, hint_threshold=1, coverage_bias=0.3),
    HintStyle("PLAY_COL",  play_bias=0.9, color_bias=0.9, hint_threshold=1, coverage_bias=0.3),
    HintStyle("DISC_NUM",  play_bias=0.1, color_bias=0.1, hint_threshold=3, coverage_bias=0.5),
    HintStyle("DISC_COL",  play_bias=0.1, color_bias=0.9, hint_threshold=3, coverage_bias=0.5),
    HintStyle("INFO",      play_bias=0.5, color_bias=0.5, hint_threshold=5, coverage_bias=0.9),
]
N_STYLES = len(HINT_STYLES)


def score_hint_candidate(action_tuple, hands, board, knowledge_other,
                         intentions, style: HintStyle):
    """
    给一个候选 hint 打分，分数受 HintStyle 参数调制。
    action_tuple = (HINT_COLOR, col) 或 (HINT_NUMBER, rank)
    返回 (valid: bool, score: float)
    """
    (type_, value) = action_tuple

    # 先用 pretend 检查合法性（不会让对方误打 / 误丢）
    valid, base_score, _ = pretend(action_tuple, knowledge_other,
                                   intentions, hands, board)
    if not valid:
        return False, -999.0

    score = float(base_score)

    # ── play_bias 调制 ──────────────────────────────────────────────
    # 如果这个 hint 主要服务 playable 牌，乘以 play_bias；否则乘以 (1-play_bias)
    serves_play = False
    serves_discard = False
    for j, (col, num) in enumerate(hands):
        is_positive = (value == col) if type_ == HINT_COLOR else (value == num)
        if is_positive:
            if board[col][1] + 1 == num:
                serves_play = True
            elif board[col][1] >= num:
                serves_discard = True

    if serves_play:
        score += style.play_bias * 4.0
    if serves_discard:
        score += (1.0 - style.play_bias) * 2.0

    # ── color_bias 调制 ─────────────────────────────────────────────
    if type_ == HINT_COLOR:
        score *= (0.5 + style.color_bias)          # color_bias 高 → 颜色 hint 加权
    else:
        score *= (0.5 + (1.0 - style.color_bias))  # color_bias 低 → 数字 hint 加权

    # ── coverage_bias 调制 ──────────────────────────────────────────
    coverage = sum(1 for (col, num) in hands
                   if (value == col if type_ == HINT_COLOR else value == num))
    score += style.coverage_bias * coverage * 0.5

    return True, score


class HintStylePlayer:
    """
    参数化风格玩家。
    hint 决策由 style 参数驱动，但仍然在「不乱打/不乱丢」约束下运行。
    play / discard 决策与 IntentionalPlayer 基本一致（保证博弈合理性）。
    """
    def __init__(self, name, pnr, style: HintStyle = None):
        self.name  = name
        self.pnr   = pnr
        self.style = style if style else random.choice(HINT_STYLES)
        self.hints_given = {}   # (cnr, player) → [HINT_COLOR/HINT_NUMBER]
        self.explanation = []

    # ── 核心决策 ────────────────────────────────────────────────────
    def get_action(self, nr, hands, knowledge, trash, played, board,
                   valid_actions, hints):
        handsize = len(knowledge[0])
        possible = [get_possible(k) for k in knowledge[nr]]

        # 1. 确定性 play
        for i, p in enumerate(possible):
            if playable(p, board):
                return Action(PLAY, cnr=i)

        # 2. 确定性 discard（只在 hints 不紧张时）
        discards = [i for i, p in enumerate(possible) if discardable(p, board)]
        if discards and hints < self.style.hint_threshold:
            return Action(DISCARD, cnr=random.choice(discards))

        # 3. 给 hint（如果有足够的 hints）
        if hints >= self.style.hint_threshold:
            best_action = self._choose_hint(nr, hands, knowledge, board, hints)
            if best_action:
                return best_action

        # 4. fallback discard
        if discards:
            return Action(DISCARD, cnr=random.choice(discards))

        # 5. 最差情况：按 pretend_discard 打分丢牌
        scores = [pretend_discard(Action(DISCARD, cnr=i), knowledge[nr], board, trash)
                  for i in range(handsize)]
        scores.sort(key=lambda x: -x[1])
        return scores[0][0]

    def _choose_hint(self, nr, hands, knowledge, board, hints):
        """根据自身 style 从所有合法 hint 中选最高分的那个"""
        best_score  = -1e9
        best_action = None

        n_players = len(hands)
        for target in range(n_players):
            if target == nr:
                continue
            if not hands[target]:
                continue
            if target >= len(knowledge) or not knowledge[target]:
                continue

            # 计算 intentions（target 的每张牌的意图）
            intentions = [None] * len(hands[target])
            for j, (col, n) in enumerate(hands[target]):
                if board[col][1] + 1 == n:
                    intentions[j] = PLAY
                elif board[col][1] >= n:
                    intentions[j] = DISCARD
                else:
                    intentions[j] = CANDISCARD

            # 候选：所有颜色 + 所有数字
            candidates = ([(HINT_COLOR, c) for c in ALL_COLORS] +
                          [(HINT_NUMBER, r) for r in range(1, 6)])

            for cand in candidates:
                try:
                    valid, sc = score_hint_candidate(
                        cand, hands[target], board,
                        knowledge[target], intentions, self.style)
                except (IndexError, Exception):
                    continue
                if valid and sc > best_score:
                    best_score = sc
                    if cand[0] == HINT_COLOR:
                        best_action = Action(HINT_COLOR, pnr=target, col=cand[1])
                    else:
                        best_action = Action(HINT_NUMBER, pnr=target, num=cand[1])

        return best_action

    def inform(self, action, player, game):
        if action.type in [PLAY, DISCARD]:
            if (action.cnr, player) in self.hints_given:
                del self.hints_given[(action.cnr, player)]
            for i in range(10):
                key_new = (action.cnr + i + 1, player)
                key_old = (action.cnr + i,     player)
                if key_new in self.hints_given:
                    self.hints_given[key_old] = self.hints_given.pop(key_new)

    def get_explanation(self):
        return self.explanation


# ═══════════════════════════════════════════════════════════════════════
#
#  PART 2: 贝叶斯队友模型 (TeammateBeliefModel)
#
# ═══════════════════════════════════════════════════════════════════════

class TeammateBeliefModel:
    """
    对单个队友维护一个关于其 HintStyle 的信念分布。

    内部维护一个长度 N_STYLES 的概率向量 P(θ_k | observations)。
    每次观察到队友的一个 hint 动作，就做贝叶斯更新：
        P(θ_k | a_obs) ∝ P(a_obs | θ_k) * P(θ_k)

    Likelihood P(a_obs | θ_k) 通过 score_hint_candidate 软化得到：
        L(a_obs | θ_k) = softmax(scores)[index of a_obs]
    """
    def __init__(self, player_id, prior=None):
        self.player_id = player_id
        # 先验：均匀分布
        if prior is None:
            self.beliefs = np.ones(N_STYLES) / N_STYLES
        else:
            self.beliefs = np.array(prior, dtype=float)
            self.beliefs /= self.beliefs.sum()

        # 观察历史（用于调试 / 实验分析）
        self.observation_count = 0
        self.belief_history    = [self.beliefs.copy()]

    # ── 核心更新 ────────────────────────────────────────────────────
    def update(self, observed_action: Action, hands, knowledge_of_target,
               board, intentions_of_target):
        """
        observed_action : 队友刚给出的 hint
        hands           : 观察到的全部手牌（不含自己）
        knowledge_of_target : 被 hint 玩家的知识状态
        board           : 当前棋盘
        intentions_of_target : 被 hint 玩家每张牌的意图标注
        """
        if observed_action.type not in [HINT_COLOR, HINT_NUMBER]:
            return  # 只对 hint 做推断

        target = observed_action.pnr
        if target not in range(len(hands)) or not hands[target]:
            return

        # 把 observed_action 转成 (type, value) 供打分
        if observed_action.type == HINT_COLOR:
            obs_tuple = (HINT_COLOR, observed_action.col)
        else:
            obs_tuple = (HINT_NUMBER, observed_action.num)

        # 对每个策略假设 θ_k 计算 likelihood
        likelihoods = np.zeros(N_STYLES)
        for k, style in enumerate(HINT_STYLES):
            # 收集该风格下所有合法候选的得分
            all_scores = []
            for c in ALL_COLORS:
                cand = (HINT_COLOR, c)
                v, sc = score_hint_candidate(
                    cand, hands[target], board,
                    knowledge_of_target, intentions_of_target, style)
                all_scores.append(sc if v else -1e9)
            for r in range(1, 6):
                cand = (HINT_NUMBER, r)
                v, sc = score_hint_candidate(
                    cand, hands[target], board,
                    knowledge_of_target, intentions_of_target, style)
                all_scores.append(sc if v else -1e9)

            # obs_tuple 在候选列表里的索引
            color_candidates = [(HINT_COLOR, c) for c in ALL_COLORS]
            number_candidates = [(HINT_NUMBER, r) for r in range(1, 6)]
            all_candidates = color_candidates + number_candidates

            try:
                obs_idx = all_candidates.index(obs_tuple)
            except ValueError:
                obs_idx = -1

            if obs_idx < 0:
                likelihoods[k] = 1e-9
                continue

            # softmax → 转成概率分布 → 取 obs 对应概率
            arr = np.array(all_scores)
            arr = arr - arr.max()          # 数值稳定
            exp_arr = np.exp(np.clip(arr, -50, 0))
            prob = exp_arr / (exp_arr.sum() + 1e-12)
            likelihoods[k] = float(prob[obs_idx]) + 1e-9   # 防零
        # 防止单个 hint 的 softmax likelihood 过于极端。
        likelihoods = 0.90 * likelihoods + 0.10 * (np.ones(N_STYLES) / N_STYLES)
        # 贝叶斯更新
        posterior = self.beliefs * likelihoods
        total = posterior.sum()
        if total < 1e-12:
            # 全部接近零：重置到先验（防退化）
            self.beliefs = np.ones(N_STYLES) / N_STYLES
        else:
            self.beliefs = posterior / total

        self.observation_count += 1
        self.belief_history.append(self.beliefs.copy())

    def update_from_sampled_self_hint(self, observed_action: Action,
                                      sampled_target_hands,
                                      knowledge_of_target,
                                      board,
                                      temperature=1.0):
        """
        用于 observer 自己是 hint target 的情况。
        这里不能使用真实 self hand，所以调用方传入从自己 knowledge 中采样出的
        candidate hands。likelihood 对这些 candidate hands 做 Monte Carlo 平均。
        """
        if observed_action.type not in [HINT_COLOR, HINT_NUMBER]:
            return
        if not sampled_target_hands:
            return

        if observed_action.type == HINT_COLOR:
            obs_tuple = (HINT_COLOR, observed_action.col)
        else:
            obs_tuple = (HINT_NUMBER, observed_action.num)

        all_candidates = ([(HINT_COLOR, c) for c in ALL_COLORS] +
                          [(HINT_NUMBER, r) for r in range(1, 6)])
        try:
            obs_idx = all_candidates.index(obs_tuple)
        except ValueError:
            return

        likelihoods = np.zeros(N_STYLES)

        for k, style in enumerate(HINT_STYLES):
            probs_for_samples = []
            for sampled_hand in sampled_target_hands:
                intentions = []
                for col, n in sampled_hand:
                    if board[col][1] + 1 == n:
                        intentions.append(PLAY)
                    elif board[col][1] >= n:
                        intentions.append(DISCARD)
                    else:
                        intentions.append(CANDISCARD)

                all_scores = []
                for c in ALL_COLORS:
                    cand = (HINT_COLOR, c)
                    v, sc = score_hint_candidate(
                        cand, sampled_hand, board,
                        knowledge_of_target, intentions, style)
                    all_scores.append(sc if v else -1e9)
                for r in range(1, 6):
                    cand = (HINT_NUMBER, r)
                    v, sc = score_hint_candidate(
                        cand, sampled_hand, board,
                        knowledge_of_target, intentions, style)
                    all_scores.append(sc if v else -1e9)

                arr = np.array(all_scores, dtype=float) / max(temperature, 1e-6)
                arr = arr - arr.max()
                exp_arr = np.exp(np.clip(arr, -50, 0))
                prob = exp_arr / (exp_arr.sum() + 1e-12)
                probs_for_samples.append(float(prob[obs_idx]))

            likelihoods[k] = float(np.mean(probs_for_samples)) + 1e-9
        # 防止单次 noisy self-hint 把 belief 过度推偏。
        # 混入少量均匀 likelihood，相当于 observation noise。
        likelihoods = 0.85 * likelihoods + 0.15 * (np.ones(N_STYLES) / N_STYLES)
        posterior = self.beliefs * likelihoods
        total = posterior.sum()
        if total < 1e-12:
            self.beliefs = np.ones(N_STYLES) / N_STYLES
        else:
            self.beliefs = posterior / total

        self.observation_count += 1
        self.belief_history.append(self.beliefs.copy())

    # ── 查询接口 ────────────────────────────────────────────────────
    def most_likely_style(self) -> HintStyle:
        return HINT_STYLES[int(np.argmax(self.beliefs))]

    def expected_style(self) -> HintStyle:
        """
        返回期望风格（各参数按信念加权平均）。
        用于"软适配"而非硬切换。
        """
        pb    = self.beliefs[0] * HINT_STYLES[0].play_bias
        cb    = self.beliefs[0] * HINT_STYLES[0].color_bias
        th    = self.beliefs[0] * HINT_STYLES[0].hint_threshold
        covb  = self.beliefs[0] * HINT_STYLES[0].coverage_bias
        for k in range(1, N_STYLES):
            pb   += self.beliefs[k] * HINT_STYLES[k].play_bias
            cb   += self.beliefs[k] * HINT_STYLES[k].color_bias
            th   += self.beliefs[k] * HINT_STYLES[k].hint_threshold
            covb += self.beliefs[k] * HINT_STYLES[k].coverage_bias
        return HintStyle("EXPECTED", pb, cb, round(th), covb)

    def entropy(self) -> float:
        b = self.beliefs
        return float(-np.sum(b * np.log(b + 1e-12)))

    def __repr__(self):
        lines = [f"GivingModel(player={self.player_id}, obs={self.observation_count})"]
        for k, style in enumerate(HINT_STYLES):
            lines.append(f"  {style.name:12s}: {self.beliefs[k]:.3f}")
        return "\n".join(lines)


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

        # giving model：建模 teammate 如何给 hint
        self.teammate_models: dict[int, TeammateBeliefModel] = {}

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

        # 用于避免同一个 hint 被重复用于 teammate belief update。
        self.last_self_hint_key = None

        self.explanation = []

    def _ensure_model(self, player_id):
        if player_id not in self.teammate_models:
            self.teammate_models[player_id] = TeammateBeliefModel(player_id)

    def _ensure_response_model(self, player_id):
        if player_id not in self.response_models:
            self.response_models[player_id] = TeammateResponseModel(player_id)

    def _confident_style(self, model, threshold=0.55):
        """
        只有在 posterior 足够集中时才硬切换到 most-likely style。
        否则使用 expected_style，避免早期 noisy belief 把 self-recognition 带偏。
        """
        if float(np.max(model.beliefs)) >= threshold:
            return model.most_likely_style()
        return model.expected_style()

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

        return best_action

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
        self._ensure_model(hinter_id)
        model = self.teammate_models[hinter_id]
        giving_style = self._confident_style(model, threshold=0.55)

        # 用参数化风格模拟器作为"对方模型"。
        # 注意：posterior 不够尖锐时使用 expected_style，避免早期误判。
        fake_hinter = HintStylePlayer("fake", hinter_id, giving_style)

        used = {}
        for c in (trash + played):
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
            # 模拟对方在这个假设手牌下会给什么 hint
            fake_act = fake_hinter.get_action(
                hinter_id, newhands, self.last_knowledge,
                self.last_trash, self.last_played, self.last_board,
                [], hints + 1)
            if fake_act and fake_act == hint_act:
                possiblehands.append(h)

        # 同一个 hint 只更新一次 teammate style belief。
        # 这一步用的是 sampled self hands 的 Monte Carlo likelihood，避免偷看真实手牌。
        hint_key = (
            hint_act.type,
            hint_act.pnr,
            hint_act.col,
            hint_act.num,
            len(trash),
            len(played),
            tuple(board)
        )
        if hint_key != self.last_self_hint_key:
            sampled_for_belief = []
            sampled_pool = update_knowledge(base_knowledge, used)
            for _ in range(160):
                h = sample_hand(sampled_pool)
                if len(h) == len(base_knowledge):
                    sampled_for_belief.append(h)
            if sampled_for_belief:
                model.update_from_sampled_self_hint(
                    hint_act,
                    sampled_for_belief,
                    base_knowledge,
                    board,
                    temperature=0.9
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
            _ = self._refine_own_knowledge_from_hint(
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
        # Compared with full/SelfIntentional, this part uses the inferred teammate style.
        if hints > 0 and not result:
            best_hint = None
            best_score = -1e9

            for target in range(len(hands)):
                if target == nr or not hands[target]:
                    continue
                if target >= len(knowledge) or not knowledge[target]:
                    continue

                act = self._choose_hint_for_target(
                    target, nr, hands, knowledge, board, hints)
                if act is None:
                    continue

                # Score the selected hint using both immediate usefulness and learned belief confidence.
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

                self._ensure_model(target)
                confidence = float(np.max(self.teammate_models[target].beliefs))
                score = immediate + 0.5 * confidence

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
        if action.type in [HINT_COLOR, HINT_NUMBER]:
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
                self._ensure_model(player)
                model = self.teammate_models[player]

                # 重建当时的 intentions（被 hint 玩家的牌）
                target = action.pnr
                if target < len(game.hands) and game.hands[target]:
                    intentions = []
                    for col, n in game.hands[target]:
                        if game.board[col][1] + 1 == n:
                            intentions.append(PLAY)
                        elif game.board[col][1] >= n:
                            intentions.append(DISCARD)
                        else:
                            intentions.append(CANDISCARD)

                    model.update(
                        action,
                        game.hands,
                        game.knowledge[target],
                        game.board,
                        intentions
                    )

            # 若是对方 hint 给自己，记录以备 self-recognition
            if action.pnr == self.pnr:
                self.gothint      = (action, player)
                self.last_knowledge = copy.deepcopy(game.knowledge)
                self.last_board     = copy.deepcopy(game.board)
                self.last_trash     = list(game.trash)
                self.last_played    = list(game.played)

    def get_explanation(self):
        return self.explanation

    def print_beliefs(self):
        print(f"\n=== {self.name} (player {self.pnr}) Giving Models ===")
        for pid, model in self.teammate_models.items():
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
        return random.choice([Action(DISCARD, cnr=i) for i in range(len(knowledge[0]))])


class IntentionalPlayer(Player):
    def __init__(self, name, pnr):
        self.name = name
        self.pnr  = pnr
        self.explanation = []

    def get_action(self, nr, hands, knowledge, trash, played, board,
                   valid_actions, hints):
        handsize = len(knowledge[0])
        possible = [get_possible(k) for k in knowledge[nr]]
        result = None
        self.explanation = []

        for i, p in enumerate(possible):
            if playable(p, board):
                result = Action(PLAY, cnr=i)
        discards = [i for i, p in enumerate(possible) if discardable(p, board)]
        if discards and hints < 8 and not result:
            result = Action(DISCARD, cnr=random.choice(discards))

        intentions = [None] * handsize
        for i, h in enumerate(hands):
            if i == nr: continue
            for j, (col, n) in enumerate(h):
                if   board[col][1] + 1 == n: intentions[j] = PLAY
                elif board[col][1]     >= n: intentions[j] = DISCARD
                elif n < 5:                  intentions[j] = CANDISCARD

        if hints > 0 and not result:
            valid = []
            for c in ALL_COLORS:
                act = (HINT_COLOR, c)
                ok, sc, _ = pretend(act, knowledge[1 - nr], intentions,
                                    hands[1 - nr], board)
                if ok: valid.append((act, sc))
            for r in range(1, 6):
                act = (HINT_NUMBER, r)
                ok, sc, _ = pretend(act, knowledge[1 - nr], intentions,
                                    hands[1 - nr], board)
                if ok: valid.append((act, sc))
            if valid:
                valid.sort(key=lambda x: -x[1])
                a, _ = valid[0]
                result = (Action(HINT_COLOR,  pnr=1-nr, col=a[1])
                          if a[0] == HINT_COLOR
                          else Action(HINT_NUMBER, pnr=1-nr, num=a[1]))

        if result: return result
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
    "style":       HintStylePlayer,    # 参数化风格玩家（随机风格）
    "adaptive":    AdaptivePlayer,     # ★ 自适应推断玩家
}

STYLE_MAP = {s.name: s for s in HINT_STYLES}
names = ["Shangdi", "Yu Di", "Tian", "Nu Wa", "Pangu"]


def make_player(player_str, i):
    """
    支持格式：
      adaptive
      style(PLAY_NUM)      → 用指定风格的 HintStylePlayer
      intentional
      full
      outer
      inner
      random
    """
    if player_str in playertypes:
        return playertypes[player_str](names[i], i)
    if player_str.startswith("style("):
        style_name = player_str[6:-1].strip()
        style = STYLE_MAP.get(style_name, random.choice(HINT_STYLES))
        return HintStylePlayer(names[i], i, style)
    return Player(names[i], i)


# ─── 实验1：单次对局演示 ────────────────────────────────────────────
def run_demo(player_strs=None, seed=42, verbose=True):
    if player_strs is None:
        player_strs = ["adaptive", "style(PLAY_NUM)"]
    random.seed(seed)
    players = [make_player(p, i) for i, p in enumerate(player_strs)]
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
    reset_beliefs=False → 跨游戏保留信念（模拟 repeated interaction）
    reset_beliefs=True  → 每局重置（对照组）
    """
    players = [make_player(p, i) for i, p in enumerate(player_strs)]
    log     = NullStream()
    scores  = []

    for game_idx in range(n_games):
        if (game_idx + 1) % 100 == 0:
            print(f"  game {game_idx+1}/{n_games}")

        if reset_beliefs:
            # 重置信念：重新创建 AdaptivePlayer（保留其他玩家）
            for i, p in enumerate(players):
                if isinstance(p, AdaptivePlayer):
                    new_p = AdaptivePlayer(p.name, p.pnr, len(players))
                    players[i] = new_p

        random.seed(game_idx + 1)
        g = Game(players, log=log)
        sc = g.run()
        scores.append(sc)

    arr = np.array(scores)
    print(f"  avg={arr.mean():.2f}  std={arr.std(ddof=1):.2f}  "
          f"min={arr.min()}  max={arr.max()}")
    return scores


# ─── 实验3：信念收敛分析 ────────────────────────────────────────────
def run_belief_convergence(true_style_name="PLAY_NUM", n_games=30, seed=0):
    """
    固定队友风格为 true_style_name，观察 AdaptivePlayer 的信念随对局收敛。
    """
    print(f"\n=== Belief Convergence: true style = {true_style_name} ===")
    random.seed(seed)
    adaptive = AdaptivePlayer(names[0], 0, 2)
    teammate = HintStylePlayer(names[1], 1, STYLE_MAP[true_style_name])
    players  = [adaptive, teammate]

    for game_idx in range(n_games):
        random.seed(game_idx + 1)
        g = Game(players, log=NullStream())
        g.run()
        # 打印每局后的信念
        if 1 in adaptive.teammate_models:
            m = adaptive.teammate_models[1]
            belief_str = "  ".join(
                f"{HINT_STYLES[k].name}:{m.beliefs[k]:.2f}"
                for k in range(N_STYLES))
            print(f"  Game {game_idx+1:2d} | obs={m.observation_count:3d} | "
                  f"best={m.most_likely_style().name:12s} | {belief_str}")


# ─── 实验4：跨风格对照表 ────────────────────────────────────────────
def run_comparison_table(n_games=300):
    """
    Adaptive vs 各种风格 HintStylePlayer，同时对比「无推断」基线
    """
    print("\n=== Comparison Table ===")
    print(f"{'Teammate Style':15s}  {'Adaptive':>8s}  {'Intentional':>12s}  {'Difference':>10s}")
    print("-" * 55)

    for style in HINT_STYLES:
        # Adaptive 配对
        scores_a = run_batch(
            ["adaptive", f"style({style.name})"],
            n_games=n_games, reset_beliefs=False, verbose=False)

        # 基线：IntentionalPlayer 配对（无推断）
        scores_b = run_batch(
            ["intentional", f"style({style.name})"],
            n_games=n_games, reset_beliefs=False, verbose=False)

        avg_a = np.mean(scores_a)
        avg_b = np.mean(scores_b)
        print(f"{style.name:15s}  {avg_a:8.2f}  {avg_b:12.2f}  {avg_a-avg_b:+10.2f}")


# ─── 实验5：学习曲线（得分随对局数的变化）────────────────────────────
def run_learning_curve(style_name="PLAY_NUM", n_games=200, window=20):
    """
    展示 AdaptivePlayer 随对局数的得分变化（应该逐渐上升）
    """
    print(f"\n=== Learning Curve vs {style_name} ===")

    adaptive_keep = AdaptivePlayer(names[0], 0, 2)
    teammate_keep = HintStylePlayer(names[1], 1, STYLE_MAP[style_name])
    players_keep  = [adaptive_keep, teammate_keep]

    scores_keep  = []
    scores_reset = []

    for game_idx in range(n_games):
        # Persistent-belief condition: same adaptive player keeps its teammate model.
        random.seed(game_idx + 1)
        g_keep = Game(players_keep, log=NullStream())
        scores_keep.append(g_keep.run())

        # Reset-belief condition: same deck seed, but adaptive belief is reset every game.
        random.seed(game_idx + 1)
        players_reset = [
            AdaptivePlayer(names[0], 0, 2),
            HintStylePlayer(names[1], 1, STYLE_MAP[style_name])
        ]
        g_reset = Game(players_reset, log=NullStream())
        scores_reset.append(g_reset.run())

    # 滑动窗口均值：直接对比「跨局保留信念」和「每局重置」
    print(f"  (window={window})")
    print(f"  {'games':>13s} | {'persistent':>10s} | {'reset':>10s} | {'diff':>8s}")
    print("  " + "-" * 51)
    for i in range(0, n_games, window):
        keep_chunk  = scores_keep[i:i+window]
        reset_chunk = scores_reset[i:i+window]
        keep_avg  = np.mean(keep_chunk)
        reset_avg = np.mean(reset_chunk)
        print(f"  {i+1:3d}-{i+len(keep_chunk):3d}       | "
              f"{keep_avg:10.2f} | {reset_avg:10.2f} | {keep_avg-reset_avg:+8.2f}")


# ─── 实验6：消融实验表（persistent vs reset vs baseline）────────────────────────────

def run_ablation_table(n_games=300):
    """
    更适合写进报告的对照实验：
      1. Adaptive persistent : 跨游戏保留 teammate belief
      2. Adaptive reset      : 每局重置信念
      3. Intentional         : 原版 intentional baseline

    使用相同的 game seeds，减少随机牌堆带来的噪声。
    """
    print("\n=== Ablation Table: persistent belief vs reset belief vs baseline ===")
    print(f"{'Style':12s}  {'Persistent':>10s}  {'Reset':>10s}  {'Intentional':>12s}  {'P-Reset':>8s}  {'P-Base':>8s}  {'Belief':>10s}")
    print("-" * 92)

    for style in HINT_STYLES:
        scores_persistent = []
        scores_reset = []
        scores_base = []

        adaptive_keep = AdaptivePlayer(names[0], 0, 2)
        teammate_keep = HintStylePlayer(names[1], 1, STYLE_MAP[style.name])
        players_keep = [adaptive_keep, teammate_keep]

        for game_idx in range(n_games):
            seed = game_idx + 1

            random.seed(seed)
            g = Game(players_keep, log=NullStream())
            scores_persistent.append(g.run())

            random.seed(seed)
            players_reset = [
                AdaptivePlayer(names[0], 0, 2),
                HintStylePlayer(names[1], 1, STYLE_MAP[style.name])
            ]
            g = Game(players_reset, log=NullStream())
            scores_reset.append(g.run())

            random.seed(seed)
            players_base = [
                IntentionalPlayer(names[0], 0),
                HintStylePlayer(names[1], 1, STYLE_MAP[style.name])
            ]
            g = Game(players_base, log=NullStream())
            scores_base.append(g.run())

        p = np.mean(scores_persistent)
        r = np.mean(scores_reset)
        b = np.mean(scores_base)
        if 1 in adaptive_keep.teammate_models:
            inferred = adaptive_keep.teammate_models[1].most_likely_style().name
        else:
            inferred = "NONE"
        print(f"{style.name:12s}  {p:10.2f}  {r:10.2f}  {b:12.2f}  {p-r:+8.2f}  {p-b:+8.2f}  {inferred:>10s}")


def run_baseline_table(n_games=300):
    """
    Compare AdaptivePlayer with original source-code baselines.
    The teammate is fixed to each synthetic HintStylePlayer type.
    """
    baselines = ["random", "inner", "outer", "intentional", "full"]

    print("\n=== Baseline Table: Adaptive vs Original Source Baselines ===")
    header = f"{'Style':12s}  {'Adaptive':>9s}"
    for b in baselines:
        header += f"  {b:>11s}"
    print(header)
    print("-" * len(header))

    for style in HINT_STYLES:
        scores_adaptive = []
        adaptive = AdaptivePlayer(names[0], 0, 2)
        teammate = HintStylePlayer(names[1], 1, STYLE_MAP[style.name])
        players_adaptive = [adaptive, teammate]
        for game_idx in range(n_games):
            random.seed(game_idx + 1)
            g = Game(players_adaptive, log=NullStream())
            scores_adaptive.append(g.run())

        row = f"{style.name:12s}  {np.mean(scores_adaptive):9.2f}"

        for baseline_name in baselines:
            scores_baseline = []
            for game_idx in range(n_games):
                random.seed(game_idx + 1)
                players_baseline = [
                    make_player(baseline_name, 0),
                    HintStylePlayer(names[1], 1, STYLE_MAP[style.name])
                ]
                g = Game(players_baseline, log=NullStream())
                scores_baseline.append(g.run())
            row += f"  {np.mean(scores_baseline):11.2f}"

        print(row)


# ═══════════════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════════════

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    if not args or args[0] == "demo":
        # 快速演示：adaptive vs style(PLAY_NUM)
        run_demo(["adaptive", "style(PLAY_NUM)"], seed=7, verbose=True)

    elif args[0] == "converge":
        style = args[1] if len(args) > 1 else "PLAY_NUM"
        n     = int(args[2]) if len(args) > 2 else 30
        run_belief_convergence(style, n)

    elif args[0] == "curve":
        style = args[1] if len(args) > 1 else "PLAY_NUM"
        n     = int(args[2]) if len(args) > 2 else 100
        run_learning_curve(style, n)

    elif args[0] == "compare":
        n = int(args[1]) if len(args) > 1 else 300
        run_comparison_table(n)

    elif args[0] == "ablation":
        n = int(args[1]) if len(args) > 1 else 300
        run_ablation_table(n)

    elif args[0] == "baselines":
        n = int(args[1]) if len(args) > 1 else 300
        run_baseline_table(n)

    elif args[0] == "batch":
        p1 = args[1] if len(args) > 1 else "adaptive"
        p2 = args[2] if len(args) > 2 else "style(PLAY_NUM)"
        n  = int(args[3]) if len(args) > 3 else 500
        print(f"Batch: {p1} vs {p2}  ({n} games)")
        run_batch([p1, p2], n_games=n, reset_beliefs=False)

    else:
        print("Usage:")
        print("  python hanabi_adaptive.py demo")
        print("  python hanabi_adaptive.py converge [STYLE] [N_GAMES]")
        print("  python hanabi_adaptive.py curve    [STYLE] [N_GAMES]")
        print("  python hanabi_adaptive.py compare  [N_GAMES]")
        print("  python hanabi_adaptive.py ablation [N_GAMES]")
        print("  python hanabi_adaptive.py baselines [N_GAMES]")
        print("  python hanabi_adaptive.py batch    [P1] [P2] [N_GAMES]")
        print()
        print("Styles:", [s.name for s in HINT_STYLES])
        print("Players: adaptive, intentional, full, outer, inner, random, style(NAME)")


if __name__ == "__main__":
    main()