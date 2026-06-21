from hanabi_core import *
from adaptive_models import FrozenMetaUtilityCalibrator
from adaptive_player import AdaptivePlayer


class FixedUniformResponseModel:
    """No-op response model with a uniform play/discard/no-response prior."""
    RESPONSE_TYPES = [PLAY, DISCARD, None]

    def __init__(self, player_id):
        self.player_id = player_id
        self.observation_count = 0

    @property
    def count(self):
        return self.observation_count

    def update_from_response(self, *args, **kwargs):
        return None

    def update_no_response(self, *args, **kwargs):
        return None

    def prob(self, hint_type, response_type):
        return 1.0 / 3.0

    def posterior_strength(self, hint_type=None):
        return 3.0

    def posterior_uncertainty(self, hint_type):
        return 1.0

    def score_hint(self, hint_action, target_hand, board):
        return 0.0

    def __repr__(self):
        return f"FixedUniformResponseModel(player={self.player_id})"


class AdaptiveNoBeliefPlayer(AdaptivePlayer):
    """
    Adaptive policy shell with neutral fixed response beliefs.

    This keeps the safety filter, immediate utility, and meta calibrator, but
    removes teammate belief learning.
    """
    def _ensure_response_model(self, player_id):
        if player_id not in self.response_models:
            self.response_models[player_id] = FixedUniformResponseModel(player_id)


class AdaptiveNoMetaPlayer(AdaptivePlayer):
    """
    AdaptivePlayer without online meta-calibration.

    It still learns teammate response beliefs, but the feature-to-utility mapping
    stays fixed.
    """
    def __init__(self, name, pnr, n_players=2):
        super().__init__(name, pnr, n_players)
        self.meta_calibrator = FrozenMetaUtilityCalibrator()


class AdaptiveFullHintRankingPlayer(AdaptivePlayer):
    """
    Adaptive shell whose safe hints are ranked exactly by the Full baseline's
    `pretend()` score, without adaptive utility, immediate value, response
    belief, ambiguity, or meta calibration.
    """
    def _immediate_hint_value(self, hint_action, target_hand, board):
        return 0.0

    def _choose_hint_for_target(self, target, nr, hands, knowledge,
                                board, hints, trash=None, played=None):
        if not hands[target] or target >= len(knowledge) or not knowledge[target]:
            return None

        othercards = list(trash or []) + list(played or [])
        intentions = [None] * len(hands[target])
        for j, (col, num) in enumerate(hands[target]):
            if board[col][1] + 1 == num:
                intentions[j] = PLAY
            elif board[col][1] >= num:
                intentions[j] = DISCARD
            elif num < 5 and (col, num) not in othercards:
                intentions[j] = CANDISCARD

        best_score = -1e9
        best_action = None
        for c in ALL_COLORS:
            try:
                ok, safe_score, _ = pretend(
                    (HINT_COLOR, c), knowledge[target], intentions,
                    hands[target], board
                )
            except Exception:
                continue
            if ok and safe_score > best_score:
                best_score = safe_score
                best_action = Action(HINT_COLOR, pnr=target, col=c)

        for r in range(1, 6):
            try:
                ok, safe_score, _ = pretend(
                    (HINT_NUMBER, r), knowledge[target], intentions,
                    hands[target], board
                )
            except Exception:
                continue
            if ok and safe_score > best_score:
                best_score = safe_score
                best_action = Action(HINT_NUMBER, pnr=target, num=r)

        if best_action is None:
            return None
        return best_action, best_score, None


class AdaptiveNoImmediatePlayer(AdaptivePlayer):
    """AdaptivePlayer without the immediate public-state usefulness bonus."""
    def _immediate_hint_value(self, hint_action, target_hand, board):
        return 0.0


class AdaptiveNoAmbiguityPlayer(AdaptivePlayer):
    """AdaptivePlayer with ambiguity treated as always zero."""
    def _hint_ambiguity(self, hint_action, target_hand, board):
        return 0.0


ABLATION_PLAYER_CLASSES = {
    "adaptive_no_belief": AdaptiveNoBeliefPlayer,
    "adaptive_nobelief": AdaptiveNoBeliefPlayer,
    "adaptive_no_meta": AdaptiveNoMetaPlayer,
    "adaptive_nometa": AdaptiveNoMetaPlayer,
    "adaptive_full_hint_ranking": AdaptiveFullHintRankingPlayer,
    "adaptive_fullhintranking": AdaptiveFullHintRankingPlayer,
    "adaptive_no_immediate": AdaptiveNoImmediatePlayer,
    "adaptive_noimmediate": AdaptiveNoImmediatePlayer,
    "adaptive_no_ambiguity": AdaptiveNoAmbiguityPlayer,
    "adaptive_noambiguity": AdaptiveNoAmbiguityPlayer,
}
