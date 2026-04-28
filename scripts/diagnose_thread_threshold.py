"""Diagnose threshold-based onboarding thread initialization.

This script uses existing local artifacts only.  It builds the golden
onboarding cases, sweeps merge thresholds, prints seed geometry, and ranks
thresholds by agreement with expected thread-count ranges.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.concept_tags import load_concept_embedding_artifacts
from user.profile import (
    CORE_SPLIT_POWER,
    MAX_THREADS,
    MERGE_THRESHOLD,
    SeedSignal,
    initialize_user_centroids_threshold,
)
from tests.test_thread_grouping_golden import (
    GOLDEN_CASES,
    GoldenCase,
    build_golden_seeds,
    effective_weight,
)


THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35]


def _distance_matrix(seeds: list[SeedSignal]) -> np.ndarray:
    X = np.stack([seed.vector for seed in seeds])
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return 1.0 - X @ X.T


def _short(label: str, width: int = 24) -> str:
    clean = " ".join(label.split())
    if len(clean) <= width:
        return clean
    return clean[: width - 1] + "."


def print_seed_table(seeds: list[SeedSignal]) -> None:
    print("idx | source | label | weight | reliability | specificity | split_power | effective_weight")
    for idx, seed in enumerate(seeds):
        print(
            f"{idx:>3} | {seed.source} | {seed.label} | {seed.weight:.3f} | "
            f"{seed.reliability:.3f} | {seed.specificity:.3f} | "
            f"{seed.split_power:.3f} | {effective_weight(seed):.4f}"
        )


def print_distance_matrix(seeds: list[SeedSignal]) -> None:
    D = _distance_matrix(seeds)
    print("pairwise cosine-distance matrix")
    header = "      " + " ".join(f"{i:>7}" for i in range(len(seeds)))
    print(header)
    for i, row in enumerate(D):
        print(f"{i:>3} {''.join(f'{float(v):>8.3f}' for v in row)}  {_short(seeds[i].label)}")
    off_diag = D[~np.eye(D.shape[0], dtype=bool)] if D.shape[0] > 1 else np.array([])
    if off_diag.size:
        below_default = np.mean(off_diag <= MERGE_THRESHOLD + 1e-9)
        print(
            f"distance summary: min={off_diag.min():.3f} median={np.median(off_diag):.3f} "
            f"max={off_diag.max():.3f} share<=default({MERGE_THRESHOLD:.2f})={below_default:.2f}"
        )


def print_core_support(seeds: list[SeedSignal]) -> None:
    core = [seed.label for seed in seeds if seed.split_power >= CORE_SPLIT_POWER]
    support = [seed.label for seed in seeds if seed.split_power < CORE_SPLIT_POWER]
    if not core:
        core = [seed.label for seed in seeds]
        support = []
    print(f"core seeds ({len(core)}): {core}")
    print(f"support seeds ({len(support)}): {support}")
    if len(core) <= 1 and support:
        print("diagnostic hint: only one core seed exists, so support seeds cannot create separate threads.")


def run_case(
    case: GoldenCase,
    category_centroids: dict[str, np.ndarray],
    concept_embeddings: dict[str, np.ndarray],
    *,
    merge_history_threshold: float | None,
) -> dict[float, int]:
    seeds = build_golden_seeds(case, category_centroids, concept_embeddings)
    print("\n" + "=" * 96)
    print(f"case={case.name} expected_k_range={case.expected_k_range}")
    print_seed_table(seeds)
    print_core_support(seeds)
    print_distance_matrix(seeds)

    inferred_by_threshold: dict[float, int] = {}
    print("threshold sweep")
    print("threshold | k_u | phase1_merges | forced_max_thread_merges | thread_weights")
    for threshold in THRESHOLDS:
        debug = merge_history_threshold is not None and abs(threshold - merge_history_threshold) < 1e-9
        result = initialize_user_centroids_threshold(
            seeds,
            max_threads=MAX_THREADS,
            merge_threshold=threshold,
            core_split_power=CORE_SPLIT_POWER,
            debug=True,
        )
        inferred_by_threshold[threshold] = int(result.centroids.shape[0])
        info = result.debug or {}
        print(
            f"{threshold:>9.2f} | {result.centroids.shape[0]:>3} | "
            f"{info.get('phase1_threshold_merges', 0):>13} | "
            f"{info.get('phase2_forced_max_threads_merges', 0):>25} | "
            f"{np.asarray(result.thread_weights).round(3).tolist()}"
        )
        if debug:
            print(f"merge history at threshold={threshold:.2f}")
            for merge in info.get("merge_history", []):
                print(
                    f"  {merge['phase']} d={merge['distance']:.4f} "
                    f"threshold={merge['threshold']:.4f} "
                    f"{merge['left_group_labels']} + {merge['right_group_labels']} "
                    f"-> {merge['merged_group_labels']}"
                )
    return inferred_by_threshold


def score_thresholds(
    all_results: dict[str, dict[float, int]],
    cases: list[GoldenCase],
) -> list[tuple[float, int, int, float]]:
    ranked = []
    for threshold in THRESHOLDS:
        in_range = 0
        absolute_error = 0
        for case in cases:
            k_u = all_results[case.name][threshold]
            lo, hi = case.expected_k_range
            if lo <= k_u <= hi:
                in_range += 1
            elif k_u < lo:
                absolute_error += lo - k_u
            else:
                absolute_error += k_u - hi
        score = in_range * 10 - absolute_error
        mean_abs_error = absolute_error / max(len(cases), 1)
        ranked.append((threshold, score, in_range, mean_abs_error))
    return sorted(ranked, key=lambda row: (row[1], row[2], -row[3]), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Directory containing existing artifacts.")
    parser.add_argument("--case", action="append", help="Limit to one or more golden case names.")
    parser.add_argument(
        "--merge-history-threshold",
        type=float,
        default=MERGE_THRESHOLD,
        help="Print merge history for this threshold. Use -1 to disable.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    category_path = data_dir / "category_centroids.npy"
    if not category_path.exists():
        raise SystemExit(f"Missing category centroid artifact: {category_path}")
    category_centroids = np.load(category_path, allow_pickle=True).item()
    concept_embeddings = load_concept_embedding_artifacts(data_dir)

    case_names = set(args.case or [])
    cases = [case for case in GOLDEN_CASES if not case_names or case.name in case_names]
    missing = case_names - {case.name for case in cases}
    if missing:
        raise SystemExit(f"Unknown case name(s): {sorted(missing)}")

    merge_history_threshold = (
        None if args.merge_history_threshold < 0 else float(args.merge_history_threshold)
    )
    all_results: dict[str, dict[float, int]] = {}
    for case in cases:
        all_results[case.name] = run_case(
            case,
            category_centroids,
            concept_embeddings,
            merge_history_threshold=merge_history_threshold,
        )

    print("\n" + "=" * 96)
    print("threshold ranking")
    for threshold, score, in_range, mean_abs_error in score_thresholds(all_results, cases):
        print(
            f"threshold={threshold:.2f} score={score} "
            f"in_range={in_range}/{len(cases)} mean_abs_error={mean_abs_error:.3f}"
        )

    print("\ninterpretation hints")
    print("- If low thresholds still collapse, the selected artifact vectors are geometrically close.")
    print("- If phase-1 merges jump at the default threshold, the default is likely too permissive.")
    print("- If a case has only one core seed, core/support separation is preventing extra threads.")
    print("- Forced max-thread merges should appear only when inferred groups exceed max_threads.")


if __name__ == "__main__":
    main()
