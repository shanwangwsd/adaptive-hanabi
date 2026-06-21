import copy

from hanabi_core import *
from style_models import *
from adaptive_models import *

class AdaptivePlayer:
    """
    Continuous Bounded-Rational Teammate Model for ad-hoc Hanabi coordination.

    The agent maintains persistent teammate models during repeated play:

    1. Response model:
       Estimates how a teammate reacts after receiving our hint. It is used for
       posterior-predictive response modeling.

    Safety-aware adaptive hint selection:
       Chooses among valid safe hints using posterior-predictive teammate response,
       ambiguity, and uncertainty.

    The decision rule is deliberately not framed as RL. It is a Bayesian cognitive
    utility model: candidate hints are evaluated using posterior-predictive teammate
    responses, ambiguity, posterior uncertainty, and the original intentional
    safety filter.
    """
    def __init__(self, name, pnr, n_players=2):
        self.name      = name
        self.pnr       = pnr
        self.n_players = n_players

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
        self.explanation = []
        self.pending_meta_hints = {}

    def _ensure_response_model(self, player_id):
        if player_id not in self.response_models:
            self.response_models[player_id] = TeammateResponseModel(player_id)

    def _hint_points_card(self, hint_action, card):
        col, num = card
        if hint_action.type == HINT_COLOR:
            return hint_action.col == col
        if hint_action.type == HINT_NUMBER:
            return hint_action.num == num
        return False

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

    def _received_hint_actions(self, nr, knowledge, board):
        """
        Interpret a received hint using the Full/SelfIntentional baseline rule.
        """
        hinted_actions = []
        if not self.gothint:
            return hinted_actions

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
        return hinted_actions

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
        response_model = self.response_models[target]

        response_strength = response_model.posterior_strength(hint_action.type)
        response_conf = min(1.0, max(0.0, (response_strength - 3.0) / 60.0))
        response_uncertainty = response_model.posterior_uncertainty(hint_action.type)
        bounded_response = self._bounded_response_confidence(
            target, hint_action, target_hand, board
        )
        ambiguity = self._hint_ambiguity(hint_action, target_hand, board)

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
            "ambiguity": float(ambiguity),
            "response_uncertainty": float(response_uncertainty),
            "hint_type": hint_type_name,
            "hint_bucket": hint_bucket,
            "ambiguity_bucket": ambiguity_bucket,
        }
        return self.meta_calibrator.predict(features), features

    def _immediate_hint_value(self, hint_action, target_hand, board):
        """
        Public-state usefulness of the cards touched by a safe hint.
        This is separate from learned response utility so ablations can isolate it.
        """
        immediate = 0.0
        for col, n in target_hand:
            if not self._hint_points_card(hint_action, (col, n)):
                continue
            if board[col][1] + 1 == n:
                immediate += 4.0
            elif board[col][1] >= n:
                immediate += 1.5
            elif n == 5:
                immediate += 1.2
            else:
                immediate += 0.35
        return immediate

    def _choose_hint_for_target(self, target, nr, hands, knowledge,
                                board, hints, trash=None, played=None):
        """
        Bounded-rational adaptive hint selection.

        The original intentional `pretend()` safety filter is preserved. Among safe
        candidate hints, choose the hint with the highest Bayesian cognitive
        utility under response posteriors, ambiguity, and uncertainty.
        """
        if not hands[target] or target >= len(knowledge) or not knowledge[target]:
            return None

        self._ensure_response_model(target)

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

    # ── 主决策函数 ────────────────────────────────────────────────────
    def get_action(self, nr, hands, knowledge, trash, played, board,
                   valid_actions, hints):
        """
        Strong adaptive policy:
        1. Keep the strong SelfIntentional-style action rule as the safe backbone.
        2. Use response beliefs and meta-calibrated utility for adaptive hint selection.
        """
        self.explanation = []
        handsize = len(knowledge[nr])
        result = None

        # Step 1: Interpret received hints. The default is the strong
        # SelfIntentional-style rule used by the full baseline.
        hinted_actions = self._received_hint_actions(nr, knowledge, board)

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

        # Step 3: Adaptive hint selection under bounded-rational teammate modeling.
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
                    target, nr, hands, knowledge, board, hints, trash=trash, played=played)
                if choice is None:
                    continue
                act, cognitive_score, meta_feats = choice

                immediate = self._immediate_hint_value(act, hands[target], board)
                score = immediate + cognitive_score

                if score > best_score:
                    best_score = score
                    best_hint = act
                    meta_features = meta_feats

            # Do not require a strictly positive learned score early in a game:
            # the meta model is initially uncertain and may under-score safe hints.
            # The candidate has already passed the intentional safety filter, so
            # allow mildly negative values while rejecting clearly bad hints.
            if best_hint is not None and best_score > -0.25:
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

        # Step 4: Token management after hint selection.  At 7+ tokens, a
        # known-safe discard is valuable because it creates room for future hint
        # information, but only after useful adaptive hints have been evaluated.
        if discards and hints >= 7:
            return Action(DISCARD, cnr=random.choice(discards))

        # Step 5: Same safe discard fallback as Intentional/SelfIntentional.
        scores = [pretend_discard(Action(DISCARD, cnr=i), knowledge[nr], board, trash)
                  for i in range(handsize)]
        if not scores:
            return valid_actions[0] if valid_actions else Action(DISCARD, cnr=0)
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
            # 若是对方 hint 给自己，记录下来供 whattodo() 解释。
            if action.pnr == self.pnr:
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
        print(f"\n=== {self.name} (player {self.pnr}) Response Models ===")
        for pid, model in self.response_models.items():
            print(model)

        print(f"\n=== {self.name} (player {self.pnr}) Meta Utility Calibrator ===")
        print(self.meta_calibrator)
