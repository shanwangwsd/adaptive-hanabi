import os
import csv
import random
import sys
import numpy as np
from scipy.stats import t as student_t

from hanabi_core import *
from style_models import *
from adaptive_player import *
from ablation_players import ABLATION_PLAYER_CLASSES
from players_baseline import *
from game import *

names = ["Shangdi", "Yu Di", "Tian", "Nu Wa", "Pangu"]
ADAPTIVE_METHOD_KEYS = {"adaptive", *ABLATION_PLAYER_CLASSES.keys()}


# ═══════════════════════════════════════════════════════════════════════
#  Basic statistics
# ═══════════════════════════════════════════════════════════════════════

def make_base_seed(base_seed=None):
    """
    If base_seed is None, create a fresh run seed.
    If base_seed is provided, use it for reproducibility.
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
    """Newey-West style standard error for paired repeated-game differences."""
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
    elif n > 1 and abs(mean_diff) > 1e-12:
        p_value = 0.0
        effect_dz = float("inf") if mean_diff > 0 else float("-inf")
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


def cumulative_average(values):
    result = []
    running_sum = 0.0
    for i, v in enumerate(values):
        running_sum += float(v)
        result.append(running_sum / (i + 1))
    return result


def rolling_average(values, window=20):
    result = []
    for i in range(len(values)):
        if i < window - 1:
            result.append(np.nan)
        else:
            start = i - window + 1
            result.append(float(np.mean(values[start:i + 1])))
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Player factory and teammate settings
# ═══════════════════════════════════════════════════════════════════════

def make_player(player_str, i, n_players=4):
    """Convert a compact player string into a concrete Player object."""
    if player_str == "adaptive":
        return AdaptivePlayer(names[i], i, n_players)
    if player_str in ABLATION_PLAYER_CLASSES:
        return ABLATION_PLAYER_CLASSES[player_str](names[i], i, n_players)
    if player_str == "full":
        return SelfIntentionalPlayer(names[i], i)
    if player_str == "intentional":
        return IntentionalPlayer(names[i], i)
    if player_str == "outer":
        return OuterStatePlayer(names[i], i)
    if player_str == "inner":
        return InnerStatePlayer(names[i], i)
    if player_str == "random":
        return Player(names[i], i)
    if player_str.startswith("manual_style("):
        style_name = player_str[len("manual_style("):-1].strip()
        return RealisticStylePlayer(names[i], i, style_pair=get_experiment_style(style_name))
    if player_str.startswith("realistic_style("):
        raw = player_str[len("realistic_style("):-1].strip()
        seed = int(raw) if raw else None
        teammate_seed = None if seed is None else normalize_numpy_seed(seed + 1009 * int(i))
        return RealisticStylePlayer(names[i], i, style_pair=sample_teammate_style(teammate_seed))
    return Player(names[i], i)


def normalize_teammate_mode(mode):
    mode = str(mode).lower().strip()
    if mode == "hydric":
        mode = "hybrid"
    if mode not in {"manual", "realistic", "hybrid"}:
        raise ValueError("mode must be one of: manual, realistic, hybrid")
    return mode


def make_teammate_sets(base_seed=0, mode="hybrid", manual_ratio=0.5, n_sets=5):
    """
    Return n_sets teammate sets. Each set contains P1-P3 style-player strings.

    mode='manual':    all three teammates use predefined manual styles.
    mode='realistic': all three teammates use continuous sampled styles.
    mode='hybrid':    each teammate is manual with probability manual_ratio.
    """
    mode = normalize_teammate_mode(mode)
    manual_ratio = max(0.0, min(1.0, float(manual_ratio)))
    rng = np.random.default_rng(normalize_numpy_seed(int(base_seed)))
    style_names = list(EXPERIMENT_STYLE_SETS.keys())

    def random_short_seed():
        return int(rng.integers(10_000, 999_999_999, dtype=np.int64))

    def realistic_entry():
        return f"realistic_style({random_short_seed()})"

    def manual_entry():
        return f"manual_style({str(rng.choice(style_names))})"

    teams = []
    seen = set()
    attempts = 0
    while len(teams) < n_sets:
        attempts += 1
        if mode == "manual":
            team = [manual_entry() for _ in range(3)]
        elif mode == "realistic":
            team = [realistic_entry() for _ in range(3)]
        else:
            team = [manual_entry() if rng.random() < manual_ratio else realistic_entry() for _ in range(3)]

        signature = tuple(team)
        if signature not in seen or attempts > 1000:
            teams.append(team)
            seen.add(signature)
    return teams


def print_teammate_set_summary(teammate_sets, mode):
    print(f"Teammate mode: {mode}")
    print("Teammate sets shared by compared P0 methods:")
    for idx, team in enumerate(teammate_sets, start=1):
        print(f"  TeamSet{idx}: P1={team[0]}, P2={team[1]}, P3={team[2]}")


def build_team(first_player_str, teammates):
    players = [make_player(first_player_str, 0, 4)]
    for pid, pstr in enumerate(teammates, start=1):
        players.append(make_player(pstr, pid, 4))
    return players


# ═══════════════════════════════════════════════════════════════════════
#  Main comparison: P0 method under manual / realistic / hybrid teammates
# ═══════════════════════════════════════════════════════════════════════

def run_compare_table(n_games=300, base_seed=None, mode="hybrid", manual_ratio=0.5):
    """
    Main table:
      P0 = Outer / Intentional / Full / Adaptive
      P1-P3 = teammate style sets selected by mode.
    """
    base_seed = make_base_seed(base_seed)
    mode = normalize_teammate_mode(mode)
    teammate_sets = make_teammate_sets(base_seed, mode=mode, manual_ratio=manual_ratio, n_sets=5)

    print(f"Base seed: {base_seed}")
    print_teammate_set_summary(teammate_sets, mode)
    print("\n=== Hanabi Communication Comparison ===")
    print(
        f"{'TeamSet':8s}  {'Outer':>8s}  {'Intentional':>12s}  "
        f"{'Full':>8s}  {'Adaptive':>10s}  {'A-Full':>8s}  {'SE':>7s}  "
        f"{'95% CI':>19s}  {'p':>11s}  {'Win%':>7s}"
    )
    print("-" * 112)

    methods = [
        ("outer", "Outer"),
        ("intentional", "Intentional"),
        ("full", "Full"),
        ("adaptive", "Adaptive"),
    ]
    all_scores = {name: [] for _, name in methods}

    for set_idx, teammates in enumerate(teammate_sets, start=1):
        scores_by_method = {}

        for method_key, method_name in methods:
            scores = []
            persistent_adaptive = make_player(method_key, 0, 4) if method_key in ADAPTIVE_METHOD_KEYS else None

            for game_idx in range(n_games):
                seed = base_seed + set_idx * 100000 + game_idx + 1
                random.seed(seed)

                if method_key in ADAPTIVE_METHOD_KEYS:
                    players = [persistent_adaptive]
                    for pid, pstr in enumerate(teammates, start=1):
                        players.append(make_player(pstr, pid, 4))
                else:
                    players = build_team(method_key, teammates)

                scores.append(Game(players, log=NullStream()).run())

            scores_by_method[method_name] = scores
            all_scores[method_name].extend(scores)

        st = paired_stats(scores_by_method["Adaptive"], scores_by_method["Full"])
        ci = f"[{st['ci_low']:+.2f},{st['ci_high']:+.2f}]"
        print(
            f"{'set'+str(set_idx):8s}  "
            f"{np.mean(scores_by_method['Outer']):8.2f}  "
            f"{np.mean(scores_by_method['Intentional']):12.2f}  "
            f"{np.mean(scores_by_method['Full']):8.2f}  "
            f"{np.mean(scores_by_method['Adaptive']):10.2f}  "
            f"{st['mean_diff']:+8.2f}  {st['se']:7.3f}  {ci:>19s}  "
            f"p={st['p_value']:.4g} {st['sig']:>3s}  {100.0 * st['win_rate']:6.1f}%"
        )

    st_all = paired_stats(all_scores["Adaptive"], all_scores["Full"])
    ci_all = f"[{st_all['ci_low']:+.2f},{st_all['ci_high']:+.2f}]"
    print("-" * 112)
    print(
        f"{'ALL':8s}  "
        f"{np.mean(all_scores['Outer']):8.2f}  "
        f"{np.mean(all_scores['Intentional']):12.2f}  "
        f"{np.mean(all_scores['Full']):8.2f}  "
        f"{np.mean(all_scores['Adaptive']):10.2f}  "
        f"{st_all['mean_diff']:+8.2f}  {st_all['se']:7.3f}  {ci_all:>19s}  "
        f"p={st_all['p_value']:.4g} {st_all['sig']:>3s}  {100.0 * st_all['win_rate']:6.1f}%"
    )
    return all_scores


# ═══════════════════════════════════════════════════════════════════════
#  Core ablation comparison
# ═══════════════════════════════════════════════════════════════════════

def run_ablation_table(n_games=300, base_seed=None, mode="hybrid", manual_ratio=0.5):
    """
    Core ablation table:
      Full, FullHintRanking, NoImmediate, NoAmbiguity, NoBelief, NoMeta, Adaptive.

    Each adaptive-style P0 is persistent within a teammate set, so online belief
    learning carries across repeated games with the same teammates.
    """
    base_seed = make_base_seed(base_seed)
    mode = normalize_teammate_mode(mode)
    teammate_sets = make_teammate_sets(base_seed, mode=mode, manual_ratio=manual_ratio, n_sets=5)

    methods = [
        ("full", "Full"),
        ("adaptive_full_hint_ranking", "FullHintRanking"),
        ("adaptive_no_immediate", "NoImmediate"),
        ("adaptive_no_ambiguity", "NoAmbiguity"),
        ("adaptive_no_belief", "NoBelief"),
        ("adaptive_no_meta", "NoMeta"),
        ("adaptive", "Adaptive"),
    ]

    print(f"Base seed: {base_seed}")
    print_teammate_set_summary(teammate_sets, mode)
    print("\n=== Adaptive Ablation Comparison ===")

    header = f"{'TeamSet':8s}" + "".join(f"  {label:>15s}" for _, label in methods)
    header += f"  {'Adaptive-Full':>15s}  {'p':>11s}  {'sig':>3s}"
    print(header)
    print("-" * len(header))

    all_scores = {label: [] for _, label in methods}

    for set_idx, teammates in enumerate(teammate_sets, start=1):
        scores_by_method = {}
        for method_key, method_label in methods:
            scores = []
            persistent_p0 = make_player(method_key, 0, 4) if method_key in ADAPTIVE_METHOD_KEYS else None

            for game_idx in range(n_games):
                seed = base_seed + set_idx * 100000 + game_idx + 1
                random.seed(seed)

                if method_key in ADAPTIVE_METHOD_KEYS:
                    players = [persistent_p0]
                    for pid, pstr in enumerate(teammates, start=1):
                        players.append(make_player(pstr, pid, 4))
                else:
                    players = build_team(method_key, teammates)

                scores.append(Game(players, log=NullStream()).run())

            scores_by_method[method_label] = scores
            all_scores[method_label].extend(scores)

        st = paired_stats(scores_by_method["Adaptive"], scores_by_method["Full"])
        row = f"{'set'+str(set_idx):8s}" + "".join(
            f"  {np.mean(scores_by_method[label]):15.2f}" for _, label in methods
        )
        row += f"  {st['mean_diff']:+15.2f}  {st['p_value']:11.4g}  {st['sig']:>3s}"
        print(row)

    st_all = paired_stats(all_scores["Adaptive"], all_scores["Full"])
    print("-" * len(header))
    row = f"{'ALL':8s}" + "".join(
        f"  {np.mean(all_scores[label]):15.2f}" for _, label in methods
    )
    row += f"  {st_all['mean_diff']:+15.2f}  {st_all['p_value']:11.4g}  {st_all['sig']:>3s}"
    print(row)

    print("\nPairwise vs Full on pooled games:")
    for _, label in methods:
        if label == "Full":
            continue
        st = paired_stats(all_scores[label], all_scores["Full"])
        ci = f"[{st['ci_low']:+.2f},{st['ci_high']:+.2f}]"
        print(
            f"  {label:15s} diff={st['mean_diff']:+.2f}, SE={st['se']:.3f}, "
            f"95%CI={ci}, p={st['p_value']:.4g} {st['sig']}, "
            f"Win%={100.0 * st['win_rate']:.1f}%"
        )
    return all_scores


# ═══════════════════════════════════════════════════════════════════════
#  Composition comparison: all-adaptive / all-full / mixed team
# ═══════════════════════════════════════════════════════════════════════

def _composition_players(label):
    if label == "4Adaptive":
        return [AdaptivePlayer(names[i], i, 4) for i in range(4)]
    if label == "4Full":
        return [SelfIntentionalPlayer(names[i], i) for i in range(4)]
    if label == "1Adaptive+3Full":
        return [AdaptivePlayer(names[0], 0, 4)] + [SelfIntentionalPlayer(names[i], i) for i in range(1, 4)]
    raise ValueError(f"Unknown composition label: {label}")


def run_composition_table(n_games=300, base_seed=None):
    """Compare 4Adaptive, 4Full, and 1Adaptive+3Full."""
    base_seed = make_base_seed(base_seed)
    labels = ["4Adaptive", "4Full", "1Adaptive+3Full"]
    all_scores = {}

    print(f"Base seed: {base_seed}")
    print("\n=== Agent Composition Comparison ===")
    print(f"{'Setting':18s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>5s}  {'Max':>5s}")
    print("-" * 56)

    persistent = {
        "4Adaptive": _composition_players("4Adaptive"),
        "1Adaptive+3Full": _composition_players("1Adaptive+3Full"),
    }

    for label in labels:
        scores = []
        for game_idx in range(n_games):
            seed = base_seed + game_idx + 1
            random.seed(seed)
            if label in persistent:
                players = persistent[label]
            else:
                players = _composition_players(label)
            scores.append(Game(players, log=NullStream()).run())

        arr = np.asarray(scores, dtype=float)
        all_scores[label] = scores
        print(f"{label:18s}  {arr.mean():8.2f}  {arr.std(ddof=1):8.2f}  {int(arr.min()):5d}  {int(arr.max()):5d}")

    print("-" * 56)
    for a, b in [("4Adaptive", "4Full"), ("1Adaptive+3Full", "4Full"), ("4Adaptive", "1Adaptive+3Full")]:
        st = paired_stats(all_scores[a], all_scores[b])
        ci = f"[{st['ci_low']:+.2f},{st['ci_high']:+.2f}]"
        print(
            f"{a + ' - ' + b:30s} diff={st['mean_diff']:+.2f}, SE={st['se']:.3f}, "
            f"95%CI={ci}, p={st['p_value']:.4g} {st['sig']}, "
            f"dz={st['effect_dz']:.3f}, Win%={100.0 * st['win_rate']:.1f}%"
        )
    return all_scores


# ═══════════════════════════════════════════════════════════════════════
#  Score curves
# ═══════════════════════════════════════════════════════════════════════

def _run_p0_method_scores(method_key, teammate_sets, n_games, base_seed):
    """For each game index, average this P0 method over all teammate sets."""
    scores = []
    persistent_by_set = {
        set_idx: make_player(method_key, 0, 4)
        for set_idx in range(1, len(teammate_sets) + 1)
    } if method_key in ADAPTIVE_METHOD_KEYS else {}

    for game_idx in range(n_games):
        set_scores = []
        for set_idx, teammates in enumerate(teammate_sets, start=1):
            seed = base_seed + set_idx * 100000 + game_idx + 1
            random.seed(seed)
            if method_key in ADAPTIVE_METHOD_KEYS:
                players = [persistent_by_set[set_idx]]
                for pid, pstr in enumerate(teammates, start=1):
                    players.append(make_player(pstr, pid, 4))
            else:
                players = build_team(method_key, teammates)
            set_scores.append(Game(players, log=NullStream()).run())
        scores.append(float(np.mean(set_scores)))
    return scores


def _run_composition_scores(label, n_games, base_seed):
    scores = []
    persistent = _composition_players(label) if label in {"4Adaptive", "1Adaptive+3Full"} else None
    for game_idx in range(n_games):
        seed = base_seed + game_idx + 1
        random.seed(seed)
        players = persistent if persistent is not None else _composition_players(label)
        scores.append(Game(players, log=NullStream()).run())
    return scores


def save_score_curves(curves, base_seed, mode, window=20, out_dir="results_simplified"):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    labels = list(curves.keys())
    n_games = len(next(iter(curves.values())))
    xs = list(range(1, n_games + 1))

    csv_path = os.path.join(out_dir, f"score_curves_{mode}_seed{base_seed}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["game"] + labels + [f"cum_{label}" for label in labels] + [f"roll{window}_{label}" for label in labels])
        for i in range(n_games):
            writer.writerow(
                [i + 1]
                + [curves[label][i] for label in labels]
                + [cumulative_average(curves[label])[i] for label in labels]
                + [rolling_average(curves[label], window)[i] for label in labels]
            )

    plt.figure(figsize=(12, 7))
    for label in labels:
        plt.plot(xs, curves[label], alpha=0.25, linewidth=0.9, label=f"{label} per-game")
        plt.plot(xs, cumulative_average(curves[label]), linewidth=2.0, label=f"{label} cumulative")
    plt.xlabel("Game")
    plt.ylabel("Score")
    plt.title(f"Score Curves: per-game and cumulative ({mode}, seed={base_seed})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"score_curves_{mode}_seed{base_seed}.png")
    plt.savefig(plot_path, dpi=180)
    plt.close()

    plt.figure(figsize=(12, 7))
    for label in labels:
        plt.plot(xs, rolling_average(curves[label], window), linewidth=2.0, label=label)
    plt.xlabel("Game")
    plt.ylabel(f"Rolling-{window} average score")
    plt.title(f"Rolling Score Curves ({mode}, seed={base_seed})")
    plt.legend(fontsize=9)
    plt.tight_layout()
    roll_path = os.path.join(out_dir, f"score_curves_rolling_{mode}_seed{base_seed}_w{window}.png")
    plt.savefig(roll_path, dpi=180)
    plt.close()

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved rolling plot: {roll_path}")


def run_score_curves(n_games=300, base_seed=None, mode="hybrid", window=20, manual_ratio=0.5, out_dir="results_simplified"):
    """
    Draw all requested methods/settings on the same figure:
      Outer, Intentional, Full, Adaptive, 4Adaptive, 4Full, 1Adaptive+3Full.
    """
    base_seed = make_base_seed(base_seed)
    mode = normalize_teammate_mode(mode)
    teammate_sets = make_teammate_sets(base_seed, mode=mode, manual_ratio=manual_ratio, n_sets=5)

    print(f"Base seed: {base_seed}")
    print(f"Curve teammate mode: {mode}")

    curves = {}
    for method_key, label in [("outer", "Outer"), ("intentional", "Intentional"), ("full", "Full"), ("adaptive", "Adaptive")]:
        print(f"Running curve: {label}")
        curves[label] = _run_p0_method_scores(method_key, teammate_sets, n_games, base_seed)

    for label in ["4Adaptive", "4Full", "1Adaptive+3Full"]:
        print(f"Running curve: {label}")
        curves[label] = _run_composition_scores(label, n_games, base_seed)

    save_score_curves(curves, base_seed, mode, window=window, out_dir=out_dir)
    return curves


# ═══════════════════════════════════════════════════════════════════════
#  Command-line entry
# ═══════════════════════════════════════════════════════════════════════

def _normalize_cli_args(args):
    """
    Accept both the documented flag form and the older positional form.

    Positional examples:
      compare 300 12345 hybrid 0.5
      curve 300 12345 hybrid 20 0.5

    Flag examples:
      compare 300 12345 --style-mode hybrid --manual-ratio 0.5
      curve 300 12345 --style-mode hybrid --window 20 --manual-ratio 0.5
    """
    if not args:
        return args

    cleaned = []
    flag_values = {}
    value_flags = {
        "--style-mode": "mode",
        "--mode": "mode",
        "--manual-ratio": "manual_ratio",
        "--window": "window",
        "--record-every": "record_every",
    }

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in value_flags:
            if i + 1 >= len(args):
                raise ValueError(f"{arg} requires a value")
            flag_values[value_flags[arg]] = args[i + 1]
            i += 2
            continue
        cleaned.append(arg)
        i += 1

    if not cleaned:
        return cleaned

    def ensure_len_with_defaults(values, defaults):
        while len(values) < len(defaults):
            values.append(defaults[len(values)])

    cmd = cleaned[0]
    if cmd in {"compare", "ablation"}:
        if "mode" in flag_values:
            ensure_len_with_defaults(cleaned, [cmd, "300", "none", "hybrid"])
            cleaned[3] = flag_values["mode"]
        if "manual_ratio" in flag_values:
            ensure_len_with_defaults(cleaned, [cmd, "300", "none", "hybrid", "0.5"])
            cleaned[4] = flag_values["manual_ratio"]
    elif cmd == "curve":
        if "mode" in flag_values:
            ensure_len_with_defaults(cleaned, ["curve", "300", "none", "hybrid"])
            cleaned[3] = flag_values["mode"]
        if "window" in flag_values:
            ensure_len_with_defaults(cleaned, ["curve", "300", "none", "hybrid", "20"])
            cleaned[4] = flag_values["window"]
        if "manual_ratio" in flag_values:
            ensure_len_with_defaults(cleaned, ["curve", "300", "none", "hybrid", "20", "0.5"])
            cleaned[5] = flag_values["manual_ratio"]
    return cleaned


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    args = _normalize_cli_args(list(args))

    if not args or args[0] in {"help", "-h", "--help"}:
        print("Usage:")
        print("  python main.py compare [N_GAMES] [BASE_SEED] [MODE] [MANUAL_RATIO]")
        print("  python main.py ablation [N_GAMES] [BASE_SEED] [MODE] [MANUAL_RATIO]")
        print("  python main.py composition [N_GAMES] [BASE_SEED]")
        print("  python main.py curve [N_GAMES] [BASE_SEED] [MODE] [WINDOW] [MANUAL_RATIO]")
        print("")
        print("MODE: manual, realistic, hybrid")
        print("")
        print("Examples:")
        print("  python main.py compare 1000 20030405 manual")
        print("  python main.py compare 1000 20030405 realistic")
        print("  python main.py compare 1000 20030405 hybrid")
        print("  python main.py ablation 1000 20030405 hybrid")
        print("  python main.py composition 1000 20030405")
        print("  python main.py curve 1000 20030405 hybrid 50")
        return

    cmd = args[0]

    if cmd == "compare":
        n = int(args[1]) if len(args) > 1 else 300
        base_seed = int(args[2]) if len(args) > 2 and args[2].lower() != "none" else None
        mode = args[3] if len(args) > 3 else "hybrid"
        manual_ratio = float(args[4]) if len(args) > 4 else 0.5
        run_compare_table(n_games=n, base_seed=base_seed, mode=mode, manual_ratio=manual_ratio)
    elif cmd == "ablation":
        n = int(args[1]) if len(args) > 1 else 300
        base_seed = int(args[2]) if len(args) > 2 and args[2].lower() != "none" else None
        mode = args[3] if len(args) > 3 else "hybrid"
        manual_ratio = float(args[4]) if len(args) > 4 else 0.5
        run_ablation_table(n_games=n, base_seed=base_seed, mode=mode, manual_ratio=manual_ratio)
    elif cmd == "composition":
        n = int(args[1]) if len(args) > 1 else 300
        base_seed = int(args[2]) if len(args) > 2 and args[2].lower() != "none" else None
        run_composition_table(n_games=n, base_seed=base_seed)
    elif cmd == "curve":
        n = int(args[1]) if len(args) > 1 else 300
        base_seed = int(args[2]) if len(args) > 2 and args[2].lower() != "none" else None
        mode = args[3] if len(args) > 3 else "hybrid"
        window = int(args[4]) if len(args) > 4 else 20
        manual_ratio = float(args[5]) if len(args) > 5 else 0.5
        run_score_curves(n_games=n, base_seed=base_seed, mode=mode, window=window, manual_ratio=manual_ratio)
    else:
        print("Unknown command. Use 'help' for usage.")
