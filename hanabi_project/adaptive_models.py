import numpy as np

from hanabi_core import *


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
        "ambiguity",
        "response_uncertainty",
    ]

    BASE_WEIGHTS = {
        "safe_score": 1.00,
        "bounded_response": 1.35,
        "response_conf": 0.35,
        "ambiguity": -0.85,
        "response_uncertainty": -0.25,
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
        ambiguity = float(features.get("ambiguity", 0.0))
        uncertainty = max(0.0, min(1.0, 0.70 * response_unc + 0.30 * ambiguity))
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
