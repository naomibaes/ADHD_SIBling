#!/usr/bin/env python3
# Authors: Naomi Baes and ChatGPT
"""
Compute annual Valence and Arousal indices from per-year cooccurrence CSVs,
and derive per-collocate contributions to overall change in Valence/Arousal
at both the global level and within decades.

Input layout (produced by Step 2):

Nested layout:
    <input_root>/<target>/<target>_<YEAR>_cooccurrences.csv

Flat layout:
    <input_root>/<anything>_<YEAR>_cooccurrences.csv
    (all CSVs live directly under <input_root>)

Each cooccurrence file has columns:
    'cooccurrence', 'count'

Warriner ratings CSV must have columns:
    'word', 'V.Mean.Sum', 'A.Mean.Sum'

Outputs (default nested: per-target subfolder):

    <output_root>/<target>/<target>_annual_valence_indices.csv
    <output_root>/<target>/<target>_annual_arousal_indices.csv
    <output_root>/<target>/<target>_global_valence_arousal_contributions.csv
    <output_root>/<target>/<target>_decade_valence_arousal_contributions.csv

Outputs (with --flat-output: no per-target subfolder):

    <output_root>/<target>_annual_valence_indices.csv
    <output_root>/<target>_annual_arousal_indices.csv
    <output_root>/<target>_global_valence_arousal_contributions.csv
    <output_root>/<target>_decade_valence_arousal_contributions.csv

The “target” name comes from either:
    - the subfolder name (nested layout), or
    - the basename of <input_root> (flat layout).
"""

import os
import re
import argparse
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


# ------------ helpers -------------

def load_warriner_ratings(filename: str) -> pd.DataFrame:
    """
    Load Warriner ratings and keep only the necessary columns.
    Ensures column names are: 'word', 'V.Mean.Sum', 'A.Mean.Sum'
    and lowercases the 'word' column.
    """
    df = pd.read_csv(filename)
    required = {"word", "V.Mean.Sum", "A.Mean.Sum"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Warriner ratings missing columns: {sorted(missing)}")
    df["word"] = df["word"].astype(str).str.lower()
    return df[["word", "V.Mean.Sum", "A.Mean.Sum"]]


def extract_file_details(filename: str) -> Tuple[str, int]:
    """
    From <something>_<YEAR>_cooccurrences.csv -> (<target>, YEAR).

    Robust to extra underscores in <something> by taking the last
    4-digit token as the year.
    """
    base = os.path.basename(filename).replace(".csv", "")
    parts = base.split("_")

    year = None
    for p in reversed(parts):
        if re.fullmatch(r"\d{4}", p):
            year = int(p)
            break
    if year is None:
        raise ValueError(f"Year not found in filename: {filename}")

    # everything before the year token is treated as the "target" label
    year_str = str(year)
    year_idx = parts.index(year_str)
    target = "_".join(parts[:year_idx])
    return target, year


def normalize(val: float, min_val: float = 1.0, max_val: float = 9.0) -> float:
    """
    Normalize a Warriner rating (1–9) to 0–1.
    """
    return (val - min_val) / (max_val - min_val + min_val - min_val) if False else (val - min_val) / (max_val - min_val)


def compute_weighted_index(
    cooccurrences_df: pd.DataFrame,
    warriner_df: pd.DataFrame,
    rating_col: str,
) -> Tuple[float, float, float, float]:
    """
    Generic weighted mean + SE for a Warriner column (e.g., 'V.Mean.Sum' or 'A.Mean.Sum').
    Returns: (mean, se, normalized_mean_0to1, normalized_se_0to1)
    """
    merged = pd.merge(
        cooccurrences_df.astype({"cooccurrence": str}),
        warriner_df[["word", rating_col]],
        left_on="cooccurrence",
        right_on="word",
        how="inner",
    )

    if merged.empty:
        return 0.0, 0.0, 0.0, 0.0

    merged["weighted"] = merged["count"] * merged[rating_col]
    total_count = merged["count"].sum()
    mean = merged["weighted"].sum() / total_count if total_count > 0 else 0.0

    # SE of weighted mean (population variance approx)
    merged["sqdiff"] = (merged[rating_col] - mean) ** 2 * merged["count"]
    variance = merged["sqdiff"].sum() / total_count if total_count > 0 else 0.0
    se = np.sqrt(variance / total_count) if total_count > 0 else 0.0

    # Normalize to 0–1 (Warriner uses 1–9)
    norm_mean = normalize(mean)
    norm_se = se / (9 - 1)
    return mean, se, norm_mean, norm_se


def collect_target_dirs(input_root: str, only_target: Optional[str]) -> List[Tuple[str, str]]:
    """
    Discover which directories to treat as 'targets'.

    Layouts supported:

    1) Nested:
        <input_root>/<target>/*_cooccurrences.csv

    2) Flat:
        <input_root>/*_cooccurrences.csv

    If --target is given, we assume the nested layout and just
    return (<target>, <input_root>/<target>).
    """
    # Explicit target: assume nested layout
    if only_target:
        td = os.path.join(input_root, only_target)
        return [(only_target, td)]

    target_dirs: List[Tuple[str, str]] = []

    # Case 1: nested layout – subfolders under input_root containing cooccurrence CSVs
    for d in os.listdir(input_root):
        p = os.path.join(input_root, d)
        if not os.path.isdir(p):
            continue
        if any(f.endswith("_cooccurrences.csv") for f in os.listdir(p)):
            target_dirs.append((d, p))

    # Case 2: flat layout – cooccurrence CSVs directly under input_root
    if any(f.endswith("_cooccurrences.csv") for f in os.listdir(input_root)):
        base = os.path.basename(os.path.normpath(input_root))
        target_dirs.append((base, input_root))

    return target_dirs


# ------------ main pipeline -------------

def process_one_target(
    target_name: str,
    target_dir: str,
    output_root: str,
    warriner_df: pd.DataFrame,
    flat_output: bool,
    min_valence_dev: float,
    min_arousal_dev: float,
    min_total_count: float,
    min_delta_rel_freq: float,
    start_year: Optional[int],
    end_year: Optional[int],
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Process a single target directory:

      - Compute annual valence/arousal indices
      - Compute collocate contributions to overall change (global)
      - Compute collocate contributions by decade

    Thresholds are applied to the contribution files only.
    """

    # All cooccurrence files for this target
    all_csvs = [
        os.path.join(target_dir, f)
        for f in os.listdir(target_dir)
        if f.endswith("_cooccurrences.csv")
    ]
    if not all_csvs:
        print(f"⚠ No cooccurrence files in {target_dir}")
        return None, None, None, None

    # Filter by year range
    file_years: List[Tuple[str, int]] = []
    for fp in all_csvs:
        _, y = extract_file_details(fp)
        if start_year is not None and y < start_year:
            continue
        if end_year is not None and y > end_year:
            continue
        file_years.append((fp, y))

    if not file_years:
        print(f"⚠ No cooccurrence files in range for {target_dir}")
        return None, None, None, None

    years = [y for _, y in file_years]
    earliest_year = min(years)
    latest_year = max(years)

    # Precompute available years per decade within the filtered range
    decade_years: Dict[int, List[int]] = {}
    for _, y in file_years:
        decade = y - (y % 10)
        decade_years.setdefault(decade, []).append(y)
    for d in decade_years:
        decade_years[d] = sorted(set(decade_years[d]))

    val_rows: List[Dict] = []
    aro_rows: List[Dict] = []

    # For global contributions: word -> stats across entire (filtered) span
    collocate_stats: Dict[str, Dict[str, float]] = {}

    # For decade contributions: decade -> word -> stats within that decade
    decade_collocate_stats: Dict[int, Dict[str, Dict[str, float]]] = {}

    for fp, year in tqdm(
        sorted(file_years, key=lambda t: t[1]),
        desc=f"Processing {target_name}"
    ):
        try:
            df = pd.read_csv(fp, dtype={"cooccurrence": str})

            # Compute yearly indices
            v_mean, v_se, v_norm, v_norm_se = compute_weighted_index(
                df, warriner_df, "V.Mean.Sum"
            )
            a_mean, a_se, a_norm, a_norm_se = compute_weighted_index(
                df, warriner_df, "A.Mean.Sum"
            )

            val_rows.append(
                {
                    "target": target_name,
                    "year": int(year),
                    "avg_valence_index": v_mean,
                    "se_valence_index": v_se,
                    "normalized_valence_index": v_norm,
                    "normalized_se_valence_index": v_norm_se,
                }
            )
            aro_rows.append(
                {
                    "target": target_name,
                    "year": int(year),
                    "avg_arousal_index": a_mean,
                    "se_arousal_index": a_se,
                    "normalized_arousal_index": a_norm,
                    "normalized_se_arousal_index": a_norm_se,
                }
            )

            # ---- extra work for contribution files ----
            merged = pd.merge(
                df.astype({"cooccurrence": str}),
                warriner_df,
                left_on="cooccurrence",
                right_on="word",
                how="inner",
            )
            if merged.empty:
                continue

            total_count_year = merged["count"].sum()
            if total_count_year == 0:
                continue

            merged["rel_freq"] = merged["count"] / total_count_year

            # Global + decade stats
            decade = year - (year % 10)
            dec_earliest = min(decade_years[decade])
            dec_latest = max(decade_years[decade])

            for _, row in merged.iterrows():
                w = row["cooccurrence"]
                v = float(row["V.Mean.Sum"])
                a = float(row["A.Mean.Sum"])
                c = float(row["count"])
                p = float(row["rel_freq"])

                # Global stats
                gstats = collocate_stats.setdefault(
                    w,
                    {
                        "valence": v,
                        "arousal": a,
                        "total_count": 0.0,
                        "p_early": 0.0,
                        "p_late": 0.0,
                    },
                )
                gstats["valence"] = v
                gstats["arousal"] = a
                gstats["total_count"] += c
                if year == earliest_year:
                    gstats["p_early"] = p
                if year == latest_year:
                    gstats["p_late"] = p

                # Decade stats
                dwords = decade_collocate_stats.setdefault(decade, {})
                dstats = dwords.setdefault(
                    w,
                    {
                        "valence": v,
                        "arousal": a,
                        "total_count": 0.0,
                        "p_early": 0.0,
                        "p_late": 0.0,
                    },
                )
                dstats["valence"] = v
                dstats["arousal"] = a
                dstats["total_count"] += c
                if year == dec_earliest:
                    dstats["p_early"] = p
                if year == dec_latest:
                    dstats["p_late"] = p

        except Exception as e:
            print(f"Error processing {fp}: {e}")

    if not val_rows:
        print(f"⚠ No valid rows for {target_dir}")
        return None, None, None, None

    # ----- write annual index files -----
    out_dir = output_root if flat_output else os.path.join(output_root, target_name)
    os.makedirs(out_dir, exist_ok=True)

    val_df = pd.DataFrame(val_rows).sort_values(["target", "year"])
    val_path = os.path.join(out_dir, f"{target_name}_annual_valence_indices.csv")
    val_df.to_csv(val_path, index=False)

    aro_df = pd.DataFrame(aro_rows).sort_values(["target", "year"])
    aro_path = os.path.join(out_dir, f"{target_name}_annual_arousal_indices.csv")
    aro_df.to_csv(aro_path, index=False)

    # ----- Contribution files -----
    mean_val = warriner_df["V.Mean.Sum"].mean()
    mean_aro = warriner_df["A.Mean.Sum"].mean()

    # Global contributions
    global_rows: List[Dict] = []
    for word, s in collocate_stats.items():
        p_early = float(s.get("p_early", 0.0))
        p_late = float(s.get("p_late", 0.0))
        delta_p = p_late - p_early

        val = float(s["valence"])
        aro = float(s["arousal"])
        val_dev = val - mean_val
        aro_dev = aro - mean_aro

        val_contrib = val_dev * delta_p
        aro_contrib = aro_dev * delta_p

        global_rows.append(
            {
                "target": target_name,
                "word": word,
                "valence": val,
                "arousal": aro,
                "valence_dev": val_dev,
                "arousal_dev": aro_dev,
                "rel_freq_early": p_early,
                "rel_freq_late": p_late,
                "delta_rel_freq": delta_p,
                "total_count": s["total_count"],
                "valence_contribution": val_contrib,
                "arousal_contribution": aro_contrib,
                "abs_valence_contribution": abs(val_contrib),
                "abs_arousal_contribution": abs(aro_contrib),
            }
        )

    global_df = pd.DataFrame(global_rows)

    # Apply thresholds to global contributions
    if not global_df.empty:
        mask = np.ones(len(global_df), dtype=bool)
        if min_valence_dev > 0.0:
            mask &= global_df["valence_dev"].abs() >= min_valence_dev
        if min_arousal_dev > 0.0:
            mask &= global_df["arousal_dev"].abs() >= min_arousal_dev
        if min_total_count > 0.0:
            mask &= global_df["total_count"] >= min_total_count
        if min_delta_rel_freq > 0.0:
            mask &= global_df["delta_rel_freq"].abs() >= min_delta_rel_freq
        global_df = global_df[mask]
        global_df = global_df.sort_values(
            ["abs_valence_contribution", "abs_arousal_contribution"],
            ascending=False,
        )

    global_path = os.path.join(out_dir, f"{target_name}_global_valence_arousal_contributions.csv")
    global_df.to_csv(global_path, index=False)

    # Decade contributions
    decade_rows: List[Dict] = []
    for decade_start, dwords in decade_collocate_stats.items():
        for word, s in dwords.items():
            p_early = float(s.get("p_early", 0.0))
            p_late = float(s.get("p_late", 0.0))
            delta_p = p_late - p_early

            val = float(s["valence"])
            aro = float(s["arousal"])
            val_dev = val - mean_val
            aro_dev = aro - mean_aro

            val_contrib = val_dev * delta_p
            aro_contrib = aro_dev * delta_p

            decade_rows.append(
                {
                    "target": target_name,
                    "decade_start": decade_start,
                    "word": word,
                    "valence": val,
                    "arousal": aro,
                    "valence_dev": val_dev,
                    "arousal_dev": aro_dev,
                    "rel_freq_early": p_early,
                    "rel_freq_late": p_late,
                    "delta_rel_freq": delta_p,
                    "total_count": s["total_count"],
                    "valence_contribution": val_contrib,
                    "arousal_contribution": aro_contrib,
                    "abs_valence_contribution": abs(val_contrib),
                    "abs_arousal_contribution": abs(aro_contrib),
                }
            )

    decade_df = pd.DataFrame(decade_rows)

    if not decade_df.empty:
        mask = np.ones(len(decade_df), dtype=bool)
        if min_valence_dev > 0.0:
            mask &= decade_df["valence_dev"].abs() >= min_valence_dev
        if min_arousal_dev > 0.0:
            mask &= decade_df["arousal_dev"].abs() >= min_arousal_dev
        if min_total_count > 0.0:
            mask &= decade_df["total_count"] >= min_total_count
        if min_delta_rel_freq > 0.0:
            mask &= decade_df["delta_rel_freq"].abs() >= min_delta_rel_freq
        decade_df = decade_df[mask]
        decade_df = decade_df.sort_values(
            ["decade_start", "abs_valence_contribution", "abs_arousal_contribution"],
            ascending=[True, False, False],
        )

    decade_path = os.path.join(out_dir, f"{target_name}_decade_valence_arousal_contributions.csv")
    decade_df.to_csv(decade_path, index=False)

    print(f"✓ Saved: {val_path}")
    print(f"✓ Saved: {aro_path}")
    print(f"✓ Saved (global): {global_path}")
    print(f"✓ Saved (decade): {decade_path}")

    return val_path, aro_path, global_path, decade_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compute annual Valence and Arousal indices per target, "
            "and collocate contributions to overall change (global and by decade)."
        )
    )

    ap.add_argument(
        "--input-root",
        type=str,
        default="output/cooccurrences_by_year",
        help="Root containing per-target subfolders of cooccurrence CSVs, "
             "or a flat folder of cooccurrence CSVs.",
    )
    ap.add_argument(
        "--target",
        type=str,
        default=None,
        help="Optional target name (subfolder under --input-root). "
             "If omitted, all suitable targets are processed.",
    )
    ap.add_argument(
        "--ratings-file",
        type=str,
        required=True,
        help="Path to Warriner ratings CSV.",
    )
    ap.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Where to write results. Default: <input-root>.",
    )
    ap.add_argument(
        "--flat-output",
        action="store_true",
        help="Write summary CSVs directly into --output-root (no per-target subfolder).",
    )

    # Thresholds for contribution files
    ap.add_argument(
        "--min-valence-dev",
        type=float,
        default=0.0,
        help="Minimum |valence_dev| for a collocate to be retained "
             "in the contribution files (default: 0, no filter).",
    )
    ap.add_argument(
        "--min-arousal-dev",
        type=float,
        default=0.0,
        help="Minimum |arousal_dev| for a collocate to be retained (default: 0).",
    )
    ap.add_argument(
        "--min-total-count",
        type=float,
        default=0.0,
        help="Minimum total raw count across all years for a collocate to be "
             "retained (default: 0).",
    )
    ap.add_argument(
        "--min-delta-rel-freq",
        type=float,
        default=0.0,
        help="Minimum |delta_rel_freq| (change in relative frequency between "
             "earliest and latest year) for a collocate to be retained "
             "(default: 0).",
    )

    # Year range filter
    ap.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Earliest year to include (optional).",
    )
    ap.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Latest year to include (optional).",
    )

    args = ap.parse_args()

    output_root = args.output_root or args.input_root
    warriner = load_warriner_ratings(args.ratings_file)

    target_dirs = collect_target_dirs(args.input_root, args.target)
    if not target_dirs:
        raise SystemExit(f"No target directories or files found under {args.input_root}")

    for target_name, td in target_dirs:
        print(f"\n=== Processing target: {target_name} ===")
        process_one_target(
            target_name=target_name,
            target_dir=td,
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
