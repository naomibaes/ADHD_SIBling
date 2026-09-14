#!/usr/bin/env python3
"""
Calculate frequency-balanced annual Breadth scores with nested resampling.

Point estimate for each year:
1. Sample N sentence embeddings without replacement.
2. Calculate one Breadth score.
3. Restore the complete annual pool.
4. Repeat this R times.
5. Average the R scores.

Uncertainty interval for each year:
1. Repeat the complete R-repetition estimator CI_REPS times.
2. Use the requested percentiles of those CI_REPS averaged estimates.

Years containing fewer than N valid, non-zero embeddings are excluded.
"""

from __future__ import annotations

import argparse
import math
import re
import zlib
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# Expects filenames ending in, for example:
# narcissism_raw-sentences_us_Newsstream_1985_XLL.txt
YEAR_RE = re.compile(r"(?P<year>\d{4})_XLL\.txt$", re.IGNORECASE)


def find_files_by_year(
    root: Path,
    start_year: Optional[int],
    end_year: Optional[int],
) -> dict[int, list[Path]]:
    """Find all *_XLL.txt files and group them by year."""

    groups: dict[int, list[Path]] = {}

    for path in sorted(root.rglob("*_XLL.txt")):
        if not path.is_file():
            continue

        match = YEAR_RE.search(path.name)

        if match is None:
            print(f"Skipping unrecognised filename: {path.name}")
            continue

        year = int(match.group("year"))

        if start_year is not None and year < start_year:
            continue

        if end_year is not None and year > end_year:
            continue

        groups.setdefault(year, []).append(path)

    return groups


def parse_vector(
    line: str,
    expected_dimension: Optional[int],
) -> Optional[np.ndarray]:
    """Parse one whitespace-separated embedding vector."""

    line = line.strip()

    if not line:
        return None

    vector = np.fromstring(line, sep=" ", dtype=np.float32)

    if vector.size == 0:
        return None

    if expected_dimension is not None and vector.size != expected_dimension:
        return None

    if not np.all(np.isfinite(vector)):
        return None

    # Zero vectors do not have a defined cosine similarity.
    if float(np.linalg.norm(vector)) == 0.0:
        return None

    return vector


def load_annual_pool(files: Iterable[Path]) -> np.ndarray:
    """
    Load all valid embeddings for one year and L2-normalise them once.

    The annual pools in this application are small enough to keep in memory.
    Loading once is much faster than rereading the files for every repetition.
    """

    vectors: list[np.ndarray] = []
    expected_dimension: Optional[int] = None

    for file_path in files:
        with file_path.open("r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                vector = parse_vector(
                    line,
                    expected_dimension=expected_dimension,
                )

                if vector is None:
                    continue

                if expected_dimension is None:
                    expected_dimension = int(vector.size)

                vectors.append(vector)

    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)

    pool = np.vstack(vectors).astype(np.float32, copy=False)
    norms = np.linalg.norm(pool, axis=1, keepdims=True)
    return (pool / norms).astype(np.float32, copy=False)


def draw_breadth_scores(
    normalised_pool: np.ndarray,
    sample_size: int,
    n_draws: int,
    rng: np.random.Generator,
    batch_size: int = 100,
) -> np.ndarray:
    """
    Draw independent samples without replacement and calculate Breadth.

    Each draw restores the complete annual pool. Samples are generated in
    batches for speed.

    For unit-normalised vectors x_i, mean pairwise cosine similarity is:

        (||sum_i x_i||^2 - n) / (n * (n - 1))

    Breadth is mean pairwise cosine dissimilarity:

        (1 - mean_similarity) / 2
    """

    n_total = int(normalised_pool.shape[0])

    if n_total < sample_size:
        raise ValueError(
            f"Pool contains {n_total} embeddings but {sample_size} are required."
        )

    if n_draws < 1:
        return np.empty(0, dtype=np.float64)

    scores = np.empty(n_draws, dtype=np.float64)
    denominator = float(sample_size * (sample_size - 1))

    for start in range(0, n_draws, batch_size):
        stop = min(start + batch_size, n_draws)
        current_batch = stop - start

        # Random ranks give a uniform sample without replacement for each row.
        random_ranks = rng.random((current_batch, n_total))
        indices = np.argpartition(
            random_ranks,
            kth=sample_size - 1,
            axis=1,
        )[:, :sample_size]

        sampled_sums = normalised_pool[indices].sum(
            axis=1,
            dtype=np.float64,
        )

        squared_norms = np.einsum(
            "ij,ij->i",
            sampled_sums,
            sampled_sums,
        )

        mean_similarities = (
            squared_norms - float(sample_size)
        ) / denominator

        mean_similarities = np.clip(
            mean_similarities,
            -1.0,
            1.0,
        )

        scores[start:stop] = (
            1.0 - mean_similarities
        ) / 2.0

    return scores


def annual_seed(
    base_seed: int,
    term: str,
    year: int,
) -> int:
    """Create a deterministic seed for each term-year."""

    group_code = zlib.crc32(
        f"{term}:{year}".encode("utf-8")
    )

    return int((base_seed + group_code) % (2**32))


def summarise_year(
    term: str,
    year: int,
    files: list[Path],
    sample_size: int,
    repetitions: int,
    ci_repetitions: int,
    ci_level: float,
    seed: int,
    batch_size: int,
) -> Optional[dict[str, object]]:
    """
    Calculate one annual Breadth estimate and its nested resampling interval.
    """

    pool = load_annual_pool(files)
    n_total = int(pool.shape[0])

    if n_total < sample_size:
        print(
            f"Skipping {term} {year}: only {n_total} valid embeddings; "
            f"{sample_size} are required."
        )
        return None

    group_seed = annual_seed(seed, term, year)

    # Main annual point estimate: mean of R fresh samples of N sentences.
    point_rng = np.random.default_rng(group_seed)

    point_scores = draw_breadth_scores(
        normalised_pool=pool,
        sample_size=sample_size,
        n_draws=repetitions,
        rng=point_rng,
        batch_size=batch_size,
    )

    breadth_score = float(np.mean(point_scores))
    repetition_sd = (
        float(np.std(point_scores, ddof=1))
        if point_scores.size >= 2
        else float("nan")
    )
    repetition_se = (
        float(repetition_sd / math.sqrt(point_scores.size))
        if point_scores.size >= 2
        else float("nan")
    )

    # CI distribution: repeat the complete mean-of-R estimator CI_REPS times.
    ci_rng = np.random.default_rng(
        (group_seed + 2_000_033) % (2**32)
    )

    ci_draw_scores = draw_breadth_scores(
        normalised_pool=pool,
        sample_size=sample_size,
        n_draws=ci_repetitions * repetitions,
        rng=ci_rng,
        batch_size=batch_size,
    )

    ci_estimates = ci_draw_scores.reshape(
        ci_repetitions,
        repetitions,
    ).mean(axis=1)

    alpha = 1.0 - ci_level

    breadth_ci_lo = float(
        np.quantile(ci_estimates, alpha / 2.0)
    )
    breadth_ci_hi = float(
        np.quantile(ci_estimates, 1.0 - alpha / 2.0)
    )

    resampling_se = (
        float(np.std(ci_estimates, ddof=1))
        if ci_estimates.size >= 2
        else float("nan")
    )

    return {
        "term": term,
        "year": year,
        "year_start": year,
        "year_end": year,
        "n_total": n_total,
        "n_used_per_draw": sample_size,
        "n_reps": repetitions,
        "n_ci_reps": ci_repetitions,

        # Main outcome: mean of the original R frequency-balanced draws.
        "breadth_score": breadth_score,

        # Variability among the R individual scores used for the point estimate.
        "breadth_rep_sd": repetition_sd,
        "breadth_rep_se": repetition_se,

        # Uncertainty for the complete mean-of-R annual estimator.
        "breadth_resampling_se": resampling_se,
        "breadth_ci_lo": breadth_ci_lo,
        "breadth_ci_hi": breadth_ci_hi,

        # Compatibility columns for older plotting code.
        "breadth_boot_mean": breadth_score,
        "breadth_boot_sd": resampling_se,
        "breadth_boot_se": resampling_se,
        "n_boot": ci_repetitions,
    }


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Calculate frequency-balanced annual Breadth scores "
            "with nested resampling intervals."
        )
    )

    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Directory containing the *_XLL.txt embedding files.",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Output CSV path.",
    )

    parser.add_argument(
        "--term_name",
        type=str,
        default=None,
        help=(
            "Term label written to the CSV. "
            "Defaults to the input directory name."
        ),
    )

    parser.add_argument(
        "--start_year",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--end_year",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--n_per_year",
        type=int,
        default=50,
        help="Number of sentences sampled in each draw.",
    )

    parser.add_argument(
        "--n_reps",
        type=int,
        default=10,
        help=(
            "Number of fresh N-sentence draws averaged to produce "
            "the annual point estimate."
        ),
    )

    parser.add_argument(
        "--n_ci_reps",
        type=int,
        default=500,
        help=(
            "Number of times to repeat the complete n_reps estimator "
            "for the resampling interval."
        ),
    )

    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Percentile resampling interval level.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of samples processed together for speed.",
    )

    args = parser.parse_args()

    if not args.dir.exists():
        parser.error(
            f"Input directory does not exist: {args.dir}"
        )

    if args.n_per_year < 2:
        parser.error("--n_per_year must be at least 2.")

    if args.n_reps < 1:
        parser.error("--n_reps must be at least 1.")

    if args.n_ci_reps < 2:
        parser.error("--n_ci_reps must be at least 2.")

    if not 0.0 < args.ci < 1.0:
        parser.error("--ci must be between 0 and 1.")

    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1.")

    term = (
        args.term_name.lower()
        if args.term_name
        else args.dir.name.lower()
    )

    groups = find_files_by_year(
        root=args.dir,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    if not groups:
        print("No matching *_XLL.txt files were found.")
        return

    rows: list[dict[str, object]] = []

    for year, files in sorted(groups.items()):

        result = summarise_year(
            term=term,
            year=year,
            files=files,
            sample_size=args.n_per_year,
            repetitions=args.n_reps,
            ci_repetitions=args.n_ci_reps,
            ci_level=args.ci,
            seed=args.seed,
            batch_size=args.batch_size,
        )

        if result is not None:
            rows.append(result)

    if not rows:
        print(
            "No annual Breadth scores were produced. "
            "All years may contain fewer than the required "
            "number of embeddings."
        )
        return

    dataframe = (
        pd.DataFrame(rows)
        .sort_values(["term", "year"])
        .reset_index(drop=True)
    )

    args.csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        args.csv,
        index=False,
        float_format="%.8f",
    )

    print(
        f"Wrote {len(dataframe)} rows -> {args.csv}"
    )


if __name__ == "__main__":
    main()
