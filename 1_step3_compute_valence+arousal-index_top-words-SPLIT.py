#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute annual ADHD Valence/Arousal indices from per-year co-occurrence CSVs
and estimate per-collocate contributions to statistically supported trend phases.

Input layout
------------
<input_root>/<target>/<target>_<YEAR>_cooccurrences.csv

Each co-occurrence CSV must contain:
    cooccurrence, count

Warriner ratings CSV must contain:
    word, V.Mean.Sum, A.Mean.Sum

Contribution logic
------------------
1. Annual Valence and Arousal indices are token-weighted means among
   Warriner-covered neighbouring lemmas.

2. For each user-defined phase and each collocate, annual relative frequency
   among Warriner-covered neighbour tokens is calculated for every year in the
   phase. Missing collocate-years are treated as relative frequency 0.

3. Relative frequency is regressed on year:
       rel_freq ~ year

4. Estimated frequency change over the phase is:
       delta_rel_freq_hat = beta_year * (phase_end - phase_start)

5. Contribution is:
       (rating - reference_mean) * delta_rel_freq_hat

   Because relative frequencies sum to 1 within each year, using a fixed
   reference mean does not change the summed decomposition; it makes individual
   contributions interpretable as shifts toward/away from above- or below-
   average affective ratings.

Outputs
-------
Annual index files:
    <target>_annual_valence_indices.csv
    <target>_annual_arousal_indices.csv

Contribution files:
    <target>_global_valence_arousal_contributions.csv
    <target>_phase_valence_arousal_contributions.csv
    <target>_phase_contribution_summary.csv

The phase summary and top contributors are also printed concisely.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm


# ============================================================
# USER-DEFINED PHASES
# ============================================================
# These ADHD phases come from the separate quadratic-slope analysis:
#
# Valence:
#   significant decrease: 1990-2017
#   no significant slope: 2018-2025
#
# Arousal:
#   significant increase: 1990-2011
#   no significant slope: 2012-2022
#   significant decrease: 2023-2025
#
# Only statistically directional phases are analysed for contributors.
# 2023-2025 arousal should be interpreted cautiously because it is short
# and 2025 is a partial year.

PHASES: Dict[str, Dict[str, List[Tuple[str, int, int]]]] = {
    "adhd": {
        "valence": [
            ("declining_valence", 1990, 2017),
        ],
        "arousal": [
            ("increasing_arousal", 1990, 2011),
            ("decreasing_arousal", 2023, 2025),
        ],
    },
}


# ============================================================
# HELPERS
# ============================================================

def load_warriner_ratings(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)

    required = {"word", "V.Mean.Sum", "A.Mean.Sum"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Warriner ratings missing columns: {sorted(missing)}"
        )

    df = df[["word", "V.Mean.Sum", "A.Mean.Sum"]].copy()
    df["word"] = df["word"].astype(str).str.lower().str.strip()

    df["V.Mean.Sum"] = pd.to_numeric(df["V.Mean.Sum"], errors="coerce")
    df["A.Mean.Sum"] = pd.to_numeric(df["A.Mean.Sum"], errors="coerce")

    df = df.dropna(subset=["word", "V.Mean.Sum", "A.Mean.Sum"])
    df = df.drop_duplicates(subset=["word"], keep="first")

    return df


def extract_file_details(filename: str) -> Tuple[str, int]:
    """
    From <target>_<YEAR>_cooccurrences.csv -> (<target>, YEAR).

    Uses the final 4-digit token before '_cooccurrences' as the year,
    so targets containing underscores are supported.
    """
    base = os.path.basename(filename).replace(".csv", "")
    m = re.search(r"^(.*)_(\d{4})_cooccurrences$", base)

    if not m:
        raise ValueError(
            f"Could not parse target/year from filename: {filename}"
        )

    target = m.group(1)
    year = int(m.group(2))

    return target, year


def target_matches(parsed_target: str, requested_target: str) -> bool:
    """
    Match a requested target against a filename stem target.

    Supports both simple filenames such as:
        ADHD_1990_cooccurrences.csv

    and longer pipeline filenames such as:
        ADHD_US_newsstream_normalized-ADHD_sentences_cleaned_lemmatized_1990_cooccurrences.csv

    The requested target must occur as its own alphanumeric token, so e.g.
    'ADHD' matches before '_' or after '-', but not inside a longer word.
    """
    if parsed_target.lower() == requested_target.lower():
        return True

    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(requested_target)}(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )
    return bool(pattern.search(parsed_target))


def normalize(value: float, min_val: float = 1.0, max_val: float = 9.0) -> float:
    """Normalize a Warriner rating (1-9) to 0-1."""
    return (value - min_val) / (max_val - min_val)


def compute_weighted_index(
    cooccurrences_df: pd.DataFrame,
    warriner_df: pd.DataFrame,
    rating_col: str,
) -> Dict[str, float]:
    """
    Compute a token-weighted annual affective index and coverage statistics.

    Returns:
        mean
        se
        normalized_mean
        normalized_se
        eligible_context_tokens
        covered_tokens
        coverage_proportion
    """
    x = cooccurrences_df.copy()

    required = {"cooccurrence", "count"}
    missing = required.difference(x.columns)
    if missing:
        raise ValueError(
            f"Co-occurrence file missing columns: {sorted(missing)}"
        )

    x["cooccurrence"] = (
        x["cooccurrence"].astype(str).str.lower().str.strip()
    )
    x["count"] = pd.to_numeric(x["count"], errors="coerce").fillna(0.0)

    eligible_context_tokens = float(x["count"].sum())

    merged = pd.merge(
        x,
        warriner_df[["word", rating_col]],
        left_on="cooccurrence",
        right_on="word",
        how="inner",
    )

    covered_tokens = float(merged["count"].sum())

    coverage_proportion = (
        covered_tokens / eligible_context_tokens
        if eligible_context_tokens > 0
        else np.nan
    )

    if covered_tokens <= 0:
        return {
            "mean": np.nan,
            "se": np.nan,
            "normalized_mean": np.nan,
            "normalized_se": np.nan,
            "eligible_context_tokens": eligible_context_tokens,
            "covered_tokens": covered_tokens,
            "coverage_proportion": coverage_proportion,
        }

    ratings = merged[rating_col].astype(float).to_numpy()
    counts = merged["count"].astype(float).to_numpy()

    mean = float(np.sum(ratings * counts) / covered_tokens)

    # Token-weighted variance around the weighted mean.
    variance = float(
        np.sum(((ratings - mean) ** 2) * counts) / covered_tokens
    )
    se = float(np.sqrt(variance / covered_tokens))

    return {
        "mean": mean,
        "se": se,
        "normalized_mean": normalize(mean),
        "normalized_se": se / 8.0,
        "eligible_context_tokens": eligible_context_tokens,
        "covered_tokens": covered_tokens,
        "coverage_proportion": coverage_proportion,
    }


def collect_target_locations(
    input_root: str,
    only_target: Optional[str],
) -> List[Tuple[str, str]]:
    """
    Find target data in either of these layouts:

      Nested:
        <input_root>/<target>/<target>_<YEAR>_cooccurrences.csv

      Flat:
        <input_root>/<target>_<YEAR>_cooccurrences.csv

    Returns a list of (directory_containing_files, target_name) tuples.
    """
    if not os.path.isdir(input_root):
        return []

    locations: List[Tuple[str, str]] = []

    if only_target:
        # 1) Preferred nested layout: <input_root>/<target>/...
        direct = os.path.join(input_root, only_target)
        if os.path.isdir(direct):
            return [(direct, only_target)]

        # Case-insensitive nested fallback.
        for d in os.listdir(input_root):
            candidate = os.path.join(input_root, d)
            if os.path.isdir(candidate) and d.lower() == only_target.lower():
                return [(candidate, d)]

        # 2) Flat layout: files live directly under input_root.
        matched_target_name = None
        for f in os.listdir(input_root):
            if not f.endswith("_cooccurrences.csv"):
                continue
            fp = os.path.join(input_root, f)
            try:
                parsed_target, _ = extract_file_details(fp)
            except ValueError:
                continue
            if target_matches(parsed_target, only_target):
                matched_target_name = only_target
                break

        if matched_target_name is not None:
            return [(input_root, matched_target_name)]

        return []

    # No target supplied: collect nested targets first.
    for d in sorted(os.listdir(input_root)):
        candidate = os.path.join(input_root, d)
        if not os.path.isdir(candidate):
            continue
        if any(
            f.endswith("_cooccurrences.csv")
            for f in os.listdir(candidate)
        ):
            locations.append((candidate, d))

    # Also infer any targets stored directly in a flat input_root.
    flat_targets = set()
    for f in os.listdir(input_root):
        if not f.endswith("_cooccurrences.csv"):
            continue
        fp = os.path.join(input_root, f)
        try:
            parsed_target, _ = extract_file_details(fp)
        except ValueError:
            continue
        flat_targets.add(parsed_target)

    for target_name in sorted(flat_targets, key=str.lower):
        locations.append((input_root, target_name))

    # Remove exact duplicates while preserving order.
    seen = set()
    unique_locations = []
    for directory, target_name in locations:
        key = (os.path.abspath(directory), target_name.lower())
        if key not in seen:
            seen.add(key)
            unique_locations.append((directory, target_name))

    return unique_locations


def get_output_dir(
    output_root: str,
    target_name: str,
    flat_output: bool,
) -> str:
    out_dir = (
        output_root
        if flat_output
        else os.path.join(output_root, target_name)
    )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def fit_linear_change(
    years: List[int],
    values: List[float],
) -> Dict[str, float]:
    """
    Fit OLS: value ~ year.

    Returns slope, SE, p, fitted start/end values, and estimated change.
    """
    if len(years) < 2:
        return {
            "slope": np.nan,
            "slope_se": np.nan,
            "slope_p": np.nan,
            "fitted_start": np.nan,
            "fitted_end": np.nan,
            "estimated_change": np.nan,
            "r_squared": np.nan,
        }

    tmp = pd.DataFrame({
        "year": pd.to_numeric(pd.Series(years), errors="coerce"),
        "value": pd.to_numeric(pd.Series(values), errors="coerce"),
    }).dropna()

    if len(tmp) < 2 or tmp["year"].nunique() < 2:
        return {
            "slope": np.nan,
            "slope_se": np.nan,
            "slope_p": np.nan,
            "fitted_start": np.nan,
            "fitted_end": np.nan,
            "estimated_change": np.nan,
            "r_squared": np.nan,
        }

    # Centre year for numerical stability.
    year0 = float(tmp["year"].mean())
    tmp["year_c"] = tmp["year"] - year0

    X = sm.add_constant(tmp["year_c"])
    model = sm.OLS(tmp["value"], X).fit()

    slope = float(model.params["year_c"])
    slope_se = float(model.bse["year_c"])
    slope_p = float(model.pvalues["year_c"])

    start_year = float(tmp["year"].min())
    end_year = float(tmp["year"].max())

    fitted_start = float(
        model.params["const"] + slope * (start_year - year0)
    )
    fitted_end = float(
        model.params["const"] + slope * (end_year - year0)
    )

    return {
        "slope": slope,
        "slope_se": slope_se,
        "slope_p": slope_p,
        "fitted_start": fitted_start,
        "fitted_end": fitted_end,
        "estimated_change": fitted_end - fitted_start,
        "r_squared": float(model.rsquared),
    }


def add_share_columns(
    df: pd.DataFrame,
    contribution_col: str = "contribution",
) -> pd.DataFrame:
    """
    Add shares based on the FULL unfiltered contribution set.

    share_abs_pct:
        share of all absolute contribution magnitude.

    share_negative_pct:
        among negative contributions only, share of the total negative
        contribution magnitude.

    share_positive_pct:
        among positive contributions only, share of the total positive
        contribution magnitude.
    """
    out = df.copy()

    abs_total = float(out[contribution_col].abs().sum())
    neg_total = float(
        out.loc[out[contribution_col] < 0, contribution_col].abs().sum()
    )
    pos_total = float(
        out.loc[out[contribution_col] > 0, contribution_col].abs().sum()
    )

    out["share_abs_pct"] = np.where(
        abs_total > 0,
        100.0 * out[contribution_col].abs() / abs_total,
        0.0,
    )

    out["share_negative_pct"] = np.where(
        (out[contribution_col] < 0) & (neg_total > 0),
        100.0 * out[contribution_col].abs() / neg_total,
        0.0,
    )

    out["share_positive_pct"] = np.where(
        (out[contribution_col] > 0) & (pos_total > 0),
        100.0 * out[contribution_col].abs() / pos_total,
        0.0,
    )

    return out


def apply_display_thresholds(
    df: pd.DataFrame,
    min_dev: float,
    min_total_count: float,
    min_abs_estimated_change: float,
) -> pd.DataFrame:
    """
    Optional display/export filtering.

    IMPORTANT: shares are calculated BEFORE filtering so reported percentages
    always refer to the full contribution decomposition.
    """
    if df.empty:
        return df

    mask = np.ones(len(df), dtype=bool)

    if min_dev > 0:
        mask &= df["deviation"].abs() >= min_dev

    if min_total_count > 0:
        mask &= df["total_count"] >= min_total_count

    if min_abs_estimated_change > 0:
        mask &= (
            df["estimated_delta_rel_freq"].abs()
            >= min_abs_estimated_change
        )

    out = df.loc[mask].copy()
    out["abs_contribution"] = out["contribution"].abs()

    return out.sort_values(
        "abs_contribution",
        ascending=False,
    )


def make_phase_contributions(
    target_name: str,
    dimension: str,
    phase_label: str,
    phase_start: int,
    phase_end: int,
    yearly_word_stats: Dict[int, Dict[str, Dict[str, float]]],
    annual_index_df: pd.DataFrame,
    reference_mean: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Estimate per-word contribution within one phase using ALL annual data.

    For every word:
        rel_freq ~ year

    Missing word-years are assigned rel_freq = 0.

    estimated_delta_rel_freq =
        fitted yearly slope * (observed_end_year - observed_start_year)

    contribution =
        (rating - reference_mean) * estimated_delta_rel_freq
    """
    available_years = sorted(
        y
        for y in yearly_word_stats.keys()
        if phase_start <= y <= phase_end
    )

    if len(available_years) < 2:
        return pd.DataFrame(), {}

    observed_start = min(available_years)
    observed_end = max(available_years)

    all_words = set()
    for year in available_years:
        all_words.update(yearly_word_stats.get(year, {}).keys())

    rows = []

    for word in sorted(all_words):
        rel_freqs = []
        total_count = 0.0
        rating = None

        for year in available_years:
            stats = yearly_word_stats.get(year, {}).get(word)

            if stats is None:
                rel_freqs.append(0.0)
                continue

            rel_freqs.append(float(stats.get("rel_freq", 0.0)))
            total_count += float(stats.get("count", 0.0))

            if rating is None:
                rating = float(stats[dimension])

        if rating is None:
            continue

        freq_fit = fit_linear_change(
            years=available_years,
            values=rel_freqs,
        )

        estimated_delta = float(freq_fit["estimated_change"])
        deviation = float(rating - reference_mean)
        contribution = float(deviation * estimated_delta)

        rows.append({
            "target": target_name,
            "dimension": dimension,
            "phase": phase_label,
            "phase_start": int(phase_start),
            "phase_end": int(phase_end),
            "observed_start_year": int(observed_start),
            "observed_end_year": int(observed_end),
            "n_years": len(available_years),
            "word": word,
            "rating": rating,
            "reference_mean": float(reference_mean),
            "deviation": deviation,
            "total_count": total_count,
            "rel_freq_slope_per_year": float(freq_fit["slope"]),
            "rel_freq_slope_se": float(freq_fit["slope_se"]),
            "rel_freq_slope_p": float(freq_fit["slope_p"]),
            "fitted_rel_freq_start": float(freq_fit["fitted_start"]),
            "fitted_rel_freq_end": float(freq_fit["fitted_end"]),
            "estimated_delta_rel_freq": estimated_delta,
            "contribution": contribution,
        })

    phase_df = pd.DataFrame(rows)

    if phase_df.empty:
        return phase_df, {}

    phase_df = add_share_columns(phase_df)

    # --------------------------------------------------------
    # Fit the aggregate annual index within the same phase.
    # This is a transparent check on the decomposition.
    # --------------------------------------------------------
    index_col = (
        "avg_valence_index"
        if dimension == "valence"
        else "avg_arousal_index"
    )

    phase_index = annual_index_df[
        annual_index_df["year"].between(observed_start, observed_end)
    ][["year", index_col]].dropna()

    index_fit = fit_linear_change(
        phase_index["year"].astype(int).tolist(),
        phase_index[index_col].astype(float).tolist(),
    )

    contribution_sum = float(phase_df["contribution"].sum())

    # Since the decomposition is based on linear frequency trends,
    # this should closely match the linear fitted aggregate index change.
    decomposition_gap = (
        contribution_sum - float(index_fit["estimated_change"])
    )

    # Directionally relevant contributions for reviewer-facing summary.
    aggregate_change = float(index_fit["estimated_change"])

    if aggregate_change < 0:
        directional = phase_df[phase_df["contribution"] < 0].copy()
        directional_share_col = "share_negative_pct"
    elif aggregate_change > 0:
        directional = phase_df[phase_df["contribution"] > 0].copy()
        directional_share_col = "share_positive_pct"
    else:
        directional = phase_df.iloc[0:0].copy()
        directional_share_col = "share_abs_pct"

    directional = directional.sort_values(
        "contribution",
        ascending=(aggregate_change > 0),
    )

    # The sort above puts most negative first when aggregate_change < 0,
    # but for positive aggregate change we want largest positive first.
    if aggregate_change > 0:
        directional = directional.sort_values(
            "contribution",
            ascending=False,
        )

    top5 = directional.head(5)
    top10 = directional.head(10)

    top5_direction_share = float(
        top5[directional_share_col].sum()
    ) if not top5.empty else 0.0

    top10_direction_share = float(
        top10[directional_share_col].sum()
    ) if not top10.empty else 0.0

    summary = {
        "target": target_name,
        "dimension": dimension,
        "phase": phase_label,
        "phase_start": int(phase_start),
        "phase_end": int(phase_end),
        "observed_start_year": int(observed_start),
        "observed_end_year": int(observed_end),
        "n_years": len(available_years),
        "aggregate_index_slope_per_year": float(index_fit["slope"]),
        "aggregate_index_slope_p": float(index_fit["slope_p"]),
        "aggregate_fitted_change": aggregate_change,
        "sum_word_contributions": contribution_sum,
        "decomposition_gap": decomposition_gap,
        "top5_same_direction_share_pct": top5_direction_share,
        "top10_same_direction_share_pct": top10_direction_share,
        "top5_words": ", ".join(top5["word"].tolist()),
        "top10_words": ", ".join(top10["word"].tolist()),
    }

    return phase_df, summary


def make_global_endpoint_contributions(
    target_name: str,
    yearly_word_stats: Dict[int, Dict[str, Dict[str, float]]],
    reference_valence: float,
    reference_arousal: float,
) -> pd.DataFrame:
    """
    Retain the old whole-span endpoint decomposition as an auxiliary output.

    This is NOT the preferred reviewer-facing analysis for nonlinear trends.
    """
    years = sorted(yearly_word_stats.keys())
    if len(years) < 2:
        return pd.DataFrame()

    early = min(years)
    late = max(years)

    all_words = set()
    for year in years:
        all_words.update(yearly_word_stats[year].keys())

    rows = []

    for word in sorted(all_words):
        early_stats = yearly_word_stats.get(early, {}).get(word, {})
        late_stats = yearly_word_stats.get(late, {}).get(word, {})

        p_early = float(early_stats.get("rel_freq", 0.0))
        p_late = float(late_stats.get("rel_freq", 0.0))
        delta = p_late - p_early

        rating_val = None
        rating_aro = None
        total_count = 0.0

        for year in years:
            stats = yearly_word_stats.get(year, {}).get(word)
            if stats is not None:
                total_count += float(stats.get("count", 0.0))
                if rating_val is None:
                    rating_val = float(stats["valence"])
                    rating_aro = float(stats["arousal"])

        if rating_val is None or rating_aro is None:
            continue

        rows.append({
            "target": target_name,
            "word": word,
            "valence": rating_val,
            "arousal": rating_aro,
            "valence_deviation": rating_val - reference_valence,
            "arousal_deviation": rating_aro - reference_arousal,
            "rel_freq_early": p_early,
            "rel_freq_late": p_late,
            "delta_rel_freq": delta,
            "total_count": total_count,
            "valence_contribution": (
                (rating_val - reference_valence) * delta
            ),
            "arousal_contribution": (
                (rating_aro - reference_arousal) * delta
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN TARGET PIPELINE
# ============================================================

def process_one_target(
    target_dir: str,
    target_name: str,
    output_root: str,
    warriner_df: pd.DataFrame,
    flat_output: bool = False,
    min_valence_dev: float = 0.0,
    min_arousal_dev: float = 0.0,
    min_total_count: float = 0.0,
    min_delta_rel_freq: float = 0.0,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> Tuple[
    Optional[str],
    Optional[str],
    Tuple[Optional[str], Optional[str], Optional[str]],
]:
    """
    Process a single target folder:
      - annual Valence/Arousal indices
      - old whole-span endpoint contributions (auxiliary)
      - regression-based contributions within predefined phases
      - concise phase summary
    """
    # target_name is supplied explicitly so this works for both
    # nested target folders and a flat directory containing many targets.
    all_csvs = []
    for f in os.listdir(target_dir):
        if not f.endswith("_cooccurrences.csv"):
            continue

        fp = os.path.join(target_dir, f)
        try:
            parsed_target, _ = extract_file_details(fp)
        except ValueError:
            continue

        if target_matches(parsed_target, target_name):
            all_csvs.append(fp)

    if not all_csvs:
        print(f"[warn] No co-occurrence files in {target_dir}")
        return None, None, (None, None, None)

    file_years: List[Tuple[str, int]] = []

    for fp in all_csvs:
        try:
            _, year = extract_file_details(fp)
        except ValueError as e:
            print(f"[warn] {e}")
            continue

        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue

        file_years.append((fp, year))

    if not file_years:
        print(f"[warn] No valid co-occurrence files in selected year range for {target_name}")
        return None, None, (None, None, None)

    file_years = sorted(file_years, key=lambda x: x[1])
    years = [y for _, y in file_years]

    val_rows = []
    aro_rows = []

    # year -> word -> stats
    yearly_word_stats: Dict[int, Dict[str, Dict[str, float]]] = {}

    for fp, year in tqdm(
        file_years,
        desc=f"Processing {target_name}",
    ):
        try:
            cooc = pd.read_csv(fp, dtype={"cooccurrence": str})

            required = {"cooccurrence", "count"}
            missing = required.difference(cooc.columns)
            if missing:
                raise ValueError(
                    f"missing columns {sorted(missing)}"
                )

            cooc["cooccurrence"] = (
                cooc["cooccurrence"].astype(str).str.lower().str.strip()
            )
            cooc["count"] = pd.to_numeric(
                cooc["count"],
                errors="coerce",
            ).fillna(0.0)

            target_from_file, _ = extract_file_details(fp)

            v = compute_weighted_index(
                cooc,
                warriner_df,
                "V.Mean.Sum",
            )
            a = compute_weighted_index(
                cooc,
                warriner_df,
                "A.Mean.Sum",
            )

            val_rows.append({
                "target": target_name,
                "year": int(year),
                "avg_valence_index": v["mean"],
                "se_valence_index": v["se"],
                "normalized_valence_index": v["normalized_mean"],
                "normalized_se_valence_index": v["normalized_se"],
                "eligible_context_tokens": v["eligible_context_tokens"],
                "warriner_valence_covered_tokens": v["covered_tokens"],
                "warriner_valence_coverage_proportion": v["coverage_proportion"],
            })

            aro_rows.append({
                "target": target_name,
                "year": int(year),
                "avg_arousal_index": a["mean"],
                "se_arousal_index": a["se"],
                "normalized_arousal_index": a["normalized_mean"],
                "normalized_se_arousal_index": a["normalized_se"],
                "eligible_context_tokens": a["eligible_context_tokens"],
                "warriner_arousal_covered_tokens": a["covered_tokens"],
                "warriner_arousal_coverage_proportion": a["coverage_proportion"],
            })

            # ------------------------------------------------
            # Build annual word-level relative frequencies
            # among Warriner-covered tokens.
            # ------------------------------------------------
            merged = pd.merge(
                cooc,
                warriner_df,
                left_on="cooccurrence",
                right_on="word",
                how="inner",
            )

            covered_total = float(merged["count"].sum())

            yearly_word_stats[year] = {}

            if covered_total > 0:
                merged["rel_freq"] = (
                    merged["count"] / covered_total
                )

                for _, row in merged.iterrows():
                    word = str(row["cooccurrence"])

                    yearly_word_stats[year][word] = {
                        "valence": float(row["V.Mean.Sum"]),
                        "arousal": float(row["A.Mean.Sum"]),
                        "count": float(row["count"]),
                        "rel_freq": float(row["rel_freq"]),
                    }

        except Exception as e:
            print(f"[warn] Error processing {fp}: {e}")

    if not val_rows or not aro_rows:
        print(f"[warn] No valid annual rows for {target_name}")
        return None, None, (None, None, None)

    out_dir = get_output_dir(
        output_root,
        target_name,
        flat_output,
    )

    # --------------------------------------------------------
    # Annual index files
    # --------------------------------------------------------
    val_df = pd.DataFrame(val_rows).sort_values(
        ["target", "year"]
    )
    aro_df = pd.DataFrame(aro_rows).sort_values(
        ["target", "year"]
    )

    val_path = os.path.join(
        out_dir,
        f"{target_name}_annual_valence_indices.csv",
    )
    aro_path = os.path.join(
        out_dir,
        f"{target_name}_annual_arousal_indices.csv",
    )

    val_df.to_csv(val_path, index=False)
    aro_df.to_csv(aro_path, index=False)

    # --------------------------------------------------------
    # Fixed reference means.
    #
    # Retain the same concept as the previous script:
    # mean rating across the Warriner lexicon.
    #
    # Because annual relative frequencies sum to 1, any fixed
    # reference constant cancels in the summed decomposition.
    # --------------------------------------------------------
    mean_val = float(warriner_df["V.Mean.Sum"].mean())
    mean_aro = float(warriner_df["A.Mean.Sum"].mean())

    # --------------------------------------------------------
    # Auxiliary global endpoint contributions
    # --------------------------------------------------------
    global_df = make_global_endpoint_contributions(
        target_name=target_name,
        yearly_word_stats=yearly_word_stats,
        reference_valence=mean_val,
        reference_arousal=mean_aro,
    )

    global_path = os.path.join(
        out_dir,
        f"{target_name}_global_valence_arousal_contributions.csv",
    )
    global_df.to_csv(global_path, index=False)

    # --------------------------------------------------------
    # Preferred phase-specific regression contributions
    # --------------------------------------------------------
    target_phase_config = PHASES.get(
        target_name.lower(),
        {},
    )

    phase_chunks = []
    phase_summaries = []

    for dimension in ("valence", "arousal"):
        phase_specs = target_phase_config.get(
            dimension,
            [],
        )

        for phase_label, phase_start, phase_end in phase_specs:
            annual_df = (
                val_df
                if dimension == "valence"
                else aro_df
            )
            reference_mean = (
                mean_val
                if dimension == "valence"
                else mean_aro
            )

            phase_df, summary = make_phase_contributions(
                target_name=target_name,
                dimension=dimension,
                phase_label=phase_label,
                phase_start=phase_start,
                phase_end=phase_end,
                yearly_word_stats=yearly_word_stats,
                annual_index_df=annual_df,
                reference_mean=reference_mean,
            )

            if phase_df.empty:
                print(
                    f"[warn] No phase results for "
                    f"{target_name} / {dimension} / {phase_label}"
                )
                continue

            # Shares are already calculated from ALL words.
            # Threshold only what gets displayed/exported.
            min_dev = (
                min_valence_dev
                if dimension == "valence"
                else min_arousal_dev
            )

            display_df = apply_display_thresholds(
                phase_df,
                min_dev=min_dev,
                min_total_count=min_total_count,
                min_abs_estimated_change=min_delta_rel_freq,
            )

            phase_chunks.append(display_df)

            if summary:
                phase_summaries.append(summary)

    phase_df_all = (
        pd.concat(phase_chunks, ignore_index=True)
        if phase_chunks
        else pd.DataFrame()
    )

    phase_summary_df = pd.DataFrame(phase_summaries)

    phase_path = os.path.join(
        out_dir,
        f"{target_name}_phase_valence_arousal_contributions.csv",
    )
    phase_summary_path = os.path.join(
        out_dir,
        f"{target_name}_phase_contribution_summary.csv",
    )

    phase_df_all.to_csv(phase_path, index=False)
    phase_summary_df.to_csv(
        phase_summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Concise reviewer-facing printout
    # --------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"CONTRIBUTOR SUMMARY: {target_name}")
    print("=" * 78)

    if phase_summary_df.empty:
        print("No predefined phases were found for this target.")
    else:
        for _, s in phase_summary_df.iterrows():
            print(
                f"\n{s['dimension'].upper()} — {s['phase']} "
                f"({int(s['observed_start_year'])}-"
                f"{int(s['observed_end_year'])})"
            )

            print(
                f"  Aggregate fitted change: "
                f"{s['aggregate_fitted_change']:.6f}"
            )
            print(
                f"  Aggregate linear slope: "
                f"{s['aggregate_index_slope_per_year']:.6f} "
                f"(p={s['aggregate_index_slope_p']:.4g})"
            )
            print(
                f"  Sum of word contributions: "
                f"{s['sum_word_contributions']:.6f}"
            )
            print(
                f"  Decomposition check gap: "
                f"{s['decomposition_gap']:.3e}"
            )
            print(
                f"  Top 5 same-direction contributors: "
                f"{s['top5_words']}"
            )
            print(
                f"  Top 5 account for "
                f"{s['top5_same_direction_share_pct']:.2f}% "
                f"of same-direction contribution"
            )

            # Print top 10 terms with values.
            sub = phase_df_all[
                (phase_df_all["dimension"] == s["dimension"])
                & (phase_df_all["phase"] == s["phase"])
            ].copy()

            agg_change = float(s["aggregate_fitted_change"])

            if agg_change < 0:
                sub = sub[sub["contribution"] < 0].sort_values(
                    "contribution",
                    ascending=True,
                )
            elif agg_change > 0:
                sub = sub[sub["contribution"] > 0].sort_values(
                    "contribution",
                    ascending=False,
                )
            else:
                sub = sub.sort_values(
                    "abs_contribution",
                    ascending=False,
                )

            print("  Top directional contributors:")
            for _, r in sub.head(10).iterrows():
                share_col = (
                    "share_negative_pct"
                    if agg_change < 0
                    else "share_positive_pct"
                )

                print(
                    f"    {r['word']:<18} "
                    f"contribution={r['contribution']:+.6f} | "
                    f"Δfreq_hat={r['estimated_delta_rel_freq']:+.6f} | "
                    f"rating={r['rating']:.2f} | "
                    f"share={r[share_col]:.2f}%"
                )

    print("\nSaved:")
    print(f"  Annual valence:     {val_path}")
    print(f"  Annual arousal:     {aro_path}")
    print(f"  Global auxiliary:   {global_path}")
    print(f"  Phase details:      {phase_path}")
    print(f"  Phase summary:      {phase_summary_path}")

    return (
        val_path,
        aro_path,
        (global_path, phase_path, phase_summary_path),
    )


# ============================================================
# CLI
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compute annual Valence/Arousal indices and "
            "regression-based collocate contributions within "
            "predefined statistically supported phases."
        )
    )

    ap.add_argument(
        "--input-root",
        default="output/cooccurrences_by_year",
        help=(
            "Directory containing either per-target subfolders or "
            "flat per-year co-occurrence CSVs."
        ),
    )

    ap.add_argument(
        "--target",
        default=None,
        help=(
            "Single target name (e.g., ADHD). Works with either nested "
            "or flat input layouts. If omitted, infer all targets."
        ),
    )

    ap.add_argument(
        "--ratings-file",
        default="../resources/warriner_rat.csv",
        help=(
            "CSV containing word, V.Mean.Sum, A.Mean.Sum."
        ),
    )

    ap.add_argument(
        "--output-root",
        default="./output/valence_arousal_contributions",
        help="Output root folder.",
    )

    ap.add_argument(
        "--flat-output",
        action="store_true",
        help=(
            "Write outputs directly under output-root rather "
            "than a per-target subfolder."
        ),
    )

    # Optional DISPLAY/EXPORT thresholds only.
    # They do not affect the contribution decomposition or shares.
    ap.add_argument(
        "--min-valence-dev",
        type=float,
        default=0.0,
        help="Minimum absolute Valence deviation for exported phase rows.",
    )

    ap.add_argument(
        "--min-arousal-dev",
        type=float,
        default=0.0,
        help="Minimum absolute Arousal deviation for exported phase rows.",
    )

    ap.add_argument(
        "--min-total-count",
        type=float,
        default=0.0,
        help="Minimum total raw count across a phase for exported rows.",
    )

    ap.add_argument(
        "--min-delta-rel-freq",
        type=float,
        default=0.0,
        help=(
            "Minimum absolute regression-estimated relative-frequency "
            "change across a phase for exported rows."
        ),
    )

    ap.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Earliest year to include.",
    )

    ap.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Latest year to include.",
    )

    args = ap.parse_args()

    output_root = args.output_root or args.input_root

    warriner = load_warriner_ratings(
        args.ratings_file
    )

    target_locations = collect_target_locations(
        args.input_root,
        args.target,
    )

    if not target_locations:
        raise SystemExit(
            f"No matching co-occurrence files found under "
            f"{args.input_root}"
        )

    for target_dir, target_name in target_locations:
        process_one_target(
            target_dir=target_dir,
            target_name=target_name,
            output_root=output_root,
            warriner_df=warriner,
            flat_output=args.flat_output,
            min_valence_dev=args.min_valence_dev,
            min_arousal_dev=args.min_arousal_dev,
            min_total_count=args.min_total_count,
            min_delta_rel_freq=args.min_delta_rel_freq,
            start_year=args.start_year,
            end_year=args.end_year,
        )


if __name__ == "__main__":
    main()
