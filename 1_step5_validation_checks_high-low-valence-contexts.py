#!/usr/bin/env python3
# Authors: Naomi Baes & ChatGPT
"""
Validate **valence** scores by exporting:
  • all context windows (±k words) with per-window **valence** + SE & lexicon coverage
  • global TOP-N and BOTTOM-N **valence** windows
  • coverage summaries (**global** / **by year** / **by decade**)

INPUT
  --input-file     Path to lemmatized sentences CSV/TSV for ONE target
                   (we'll infer target from filename unless --target given)
  --ratings-file   Warriner ratings CSV with columns: word, V.Mean.Sum

DEFAULT OUTPUT LAYOUT
  <output-root>/context_window_validation/<target>/
    all_context_window_valence_scores.csv
    global_top<N>_valence_windows.csv
    global_bottom<N>_valence_windows.csv
    coverage_global_valence.csv
    coverage_by_year_valence.csv
    coverage_by_decade_valence.csv

With --flat-output, files are written directly in <output-root>/context_window_validation/
and are prefixed with <target>_ (no per-target subfolder).
"""

from __future__ import annotations

import os
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# ---------------- helpers ---------------- #

def load_warriner(path: str) -> pd.DataFrame:
    """Return df with 'word', 'V.Mean.Sum' (force lower-case on word)."""
    df = pd.read_csv(path, usecols=["word", "V.Mean.Sum"])
    df["word"] = df["word"].astype(str).str.lower()
    return df


def infer_target_from_filename(fname: str) -> str:
    """Infer target from filename e.g., 'schizophrenia_US_News_...tsv' -> 'schizophrenia'."""
    return os.path.basename(fname).split("_")[0].lower()


def extract_context(words: List[str], idx: int, win: int) -> List[str]:
    """Return context words (*excluding* the target word)."""
    start = max(0, idx - win)
    end = min(len(words), idx + win + 1)
    return [words[i] for i in range(start, end) if i != idx]


def window_valence_and_coverage(
    ctx_words: List[str],
    warriner: pd.DataFrame
) -> Tuple[float, float, int, int]:
    """
    Return (avg_valence, se_valence, matched_tokens, total_tokens) for a window.
    Valence is based on Warriner **V.Mean.Sum** (1–9). SE uses sample stdev / sqrt(n).
    """
    total = len(ctx_words)
    if total == 0:
        return None, None, 0, 0

    ctx_df = pd.DataFrame({"word": [w.lower() for w in ctx_words]})
    merged = ctx_df.merge(warriner, on="word", how="inner")
    matched = len(merged)
    if matched == 0:
        return None, None, 0, total

    avg = merged["V.Mean.Sum"].mean()
    se = merged["V.Mean.Sum"].std(ddof=1) / np.sqrt(matched) if matched > 1 else 0.0
    return avg, se, matched, total


def normalize_1to9(x: float) -> float | None:
    """Normalize 1–9 to 0–1 (optional convenience output)."""
    if x is None:
        return None
    return (x - 1.0) / 8.0


# ---------------- core ---------------- #

def process_file(
    file_path: str,
    warriner: pd.DataFrame,
    output_root: str,
    target_override: str | None,
    window: int,
    top_n: int,
    flat_output: bool,
    sep: str,
) -> str:
    """
    Main driver: reads the lemmatized sentences file, computes per-window valence & coverage,
    writes outputs, and returns path to the *all windows* CSV.
    """
    target = (target_override or infer_target_from_filename(file_path)).lower()

    # Output folders
    base_out = os.path.join(output_root, "context_window_validation")
    out_dir = base_out if flat_output else os.path.join(base_out, target)
    os.makedirs(out_dir, exist_ok=True)

    # Load input (detect one of the usual lemmatized columns)
    df = pd.read_csv(file_path, sep=sep)
    sent_col = next((c for c in ["lemmatized_sent", "lemmatized_sentence", "lemmas", "sent"] if c in df.columns), None)
    if sent_col is None:
        raise ValueError(
            f"Could not find a lemmatized sentence column in {file_path}. "
            "Tried: lemmatized_sent, lemmatized_sentence, lemmas, sent"
        )

    # Useful metadata columns we’ll carry over if present
    meta_cols = [c for c in ["year", "id", "title", "publisher", "source_type", "pub_province", "raw_sent"] if c in df.columns]

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing sentences"):
        sentence = str(row.get(sent_col) or "")
        words = sentence.split()

        # Whole-word match? Replace `in` with `==` if you want strict equality.
        targets_idx = [i for i, w in enumerate(words) if target in w.lower()]

        for idx in targets_idx:
            ctx = extract_context(words, idx, window)
            avg, se, m, total = window_valence_and_coverage(ctx, warriner)
            rows.append({
                "target": target,
                "year": int(row["year"]) if "year" in row and pd.notna(row["year"]) else "",
                "avg_valence_index": avg,
                "se_valence_index": se,
                "normalized_valence_index": normalize_1to9(avg),
                "normalized_se_valence_index": (se / 8.0) if se is not None else None,
                "matched_tokens": m,
                "total_ctx_tokens": total,
                "coverage_ratio": (m / total) if total > 0 else None,
                "context_window": " ".join(ctx),
                "lemmatized_sentence": sentence,
                **{k: row.get(k) for k in meta_cols},
            })

    ctx_df = pd.DataFrame(rows)
    prefix = f"{target}_" if flat_output else ""

    # 1) All windows
    all_path = os.path.join(out_dir, f"{prefix}all_context_window_valence_scores.csv")
    ctx_df.to_csv(all_path, index=False)
    print(f"✓ Saved all windows ➜ {all_path}")

    # 2) Top/Bottom N by valence
    if "avg_valence_index" in ctx_df.columns and not ctx_df.empty:
        top_path = os.path.join(out_dir, f"{prefix}global_top{top_n}_valence_windows.csv")
        bottom_path = os.path.join(out_dir, f"{prefix}global_bottom{top_n}_valence_windows.csv")

        (ctx_df.sort_values("avg_valence_index", ascending=False)
              .head(top_n).to_csv(top_path, index=False))
        (ctx_df.sort_values("avg_valence_index", ascending=True)
              .head(top_n).to_csv(bottom_path, index=False))

        print(f"✓ Saved top/bottom {top_n} ➜ {top_path} / {bottom_path}")
    else:
        print("⚠️ Column 'avg_valence_index' missing or no rows; skipping top/bottom exports.")

    # 3) Coverage summaries (global / by year / by decade) with **valence**-specific names
    def summarize(df_part: pd.DataFrame) -> pd.DataFrame:
        agg = df_part.agg({"matched_tokens": "sum", "total_ctx_tokens": "sum"})
        total = agg["total_ctx_tokens"] if "total_ctx_tokens" in agg else 0
        matched = agg["matched_tokens"] if "matched_tokens" in agg else 0
        coverage = (matched / total) if total > 0 else np.nan
        return pd.DataFrame({
            "windows": [len(df_part)],
            "matched_tokens": [int(matched)],
            "total_ctx_tokens": [int(total)],
            "coverage_ratio": [coverage],
            "percent_matched": [(coverage * 100.0) if pd.notna(coverage) else np.nan],
            "percent_unmatched": [(100.0 - coverage * 100.0) if pd.notna(coverage) else np.nan],
        })

    # Keep only rows with a defined coverage
    cov_cols = ["matched_tokens", "total_ctx_tokens"]
    valid_cov = ctx_df.dropna(subset=["coverage_ratio"]) if "coverage_ratio" in ctx_df.columns else ctx_df[cov_cols]

    # Global
    global_summary = summarize(valid_cov)
    global_path = os.path.join(out_dir, f"{prefix}coverage_global_valence.csv")
    global_summary.to_csv(global_path, index=False)
    print(f"✓ Saved coverage (global) ➜ {global_path}")

    # By year
    if "year" in valid_cov.columns and not valid_cov.empty:
        by_year = (valid_cov.groupby("year", dropna=False)[["matched_tokens", "total_ctx_tokens"]]
                             .sum().reset_index())
        by_year["coverage_ratio"] = by_year["matched_tokens"] / by_year["total_ctx_tokens"]
        by_year["percent_matched"] = by_year["coverage_ratio"] * 100.0
        by_year["percent_unmatched"] = 100.0 - by_year["percent_matched"]
        year_path = os.path.join(out_dir, f"{prefix}coverage_by_year_valence.csv")
        by_year.to_csv(year_path, index=False)
        print(f"✓ Saved coverage (by year) ➜ {year_path}")

        # By decade (coerce year to decade bins like 1980s, 1990s, etc.)
        tmp = valid_cov.copy()
        tmp["decade"] = tmp["year"].apply(lambda y: f"{(int(y)//10)*10}s" if f"{y}".isdigit() else "unknown")
        by_decade = (tmp.groupby("decade", dropna=False)[["matched_tokens", "total_ctx_tokens"]]
                        .sum().reset_index())
        by_decade["coverage_ratio"] = by_decade["matched_tokens"] / by_decade["total_ctx_tokens"]
        by_decade["percent_matched"] = by_decade["coverage_ratio"] * 100.0
        by_decade["percent_unmatched"] = 100.0 - by_decade["percent_matched"]
        decade_path = os.path.join(out_dir, f"{prefix}coverage_by_decade_valence.csv")
        by_decade.to_csv(decade_path, index=False)
        print(f"✓ Saved coverage (by decade) ➜ {decade_path}")

    return all_path


# ---------------- CLI ---------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Export **valence** validation windows (all + top/bottom N) and lexicon coverage summaries for a single target file."
    )
    ap.add_argument("--input-file", required=True)
    ap.add_argument("--ratings-file", default="resources/warriner_rat.csv")
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--target", default=None, help="Override inferred target (optional).")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--flat-output", action="store_true")
    ap.add_argument("--sep", default=",")
    args = ap.parse_args()

    # Print a transparent config header
    print("\n--- Valence Validation Config ---")
    for k, v in vars(args).items():
        print(f"{k:>14}: {v}")
    print("-" * 32 + "\n")

    warriner = load_warriner(args.ratings_file)

    _ = process_file(
        file_path=args.input_file,
        warriner=warriner,
        output_root=args.output_root,
        target_override=args.target,
        window=args.window,
        top_n=args.top_n,
        flat_output=args.flat_output,
        sep=args.sep,
    )


if __name__ == "__main__":
    main()
