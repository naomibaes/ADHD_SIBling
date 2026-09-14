#!/usr/bin/env python3
# Authors: Naomi Baes & ChatGPT
"""
Summarise global and phase-level word contributions to Valence and Arousal
for a single target, and write a single plain-text report with tables.

Expected inputs:

If --flat-output was used earlier:
  <output_root>/<target>_global_valence_arousal_contributions.csv
  <output_root>/<target>_phase_valence_arousal_contributions.csv

Otherwise:
  <output_root>/<target>/<target>_global_valence_arousal_contributions.csv
  <output_root>/<target>/<target>_phase_valence_arousal_contributions.csv

Output:
  <output_root>/valence_arousal_contributions/<target>_valence_arousal_summary.txt
"""

import os
import argparse
import pandas as pd


def resolve_input_paths(output_root: str, target: str, flat_output: bool):
    """
    Find the global + phase CSVs either directly under output_root (flat)
    or under output_root/target/ (nested).
    """
    if flat_output:
        global_path = os.path.join(
            output_root,
            f"{target}_global_valence_arousal_contributions.csv"
        )
        phase_path = os.path.join(
            output_root,
            f"{target}_phase_valence_arousal_contributions.csv"
        )
    else:
        tdir = os.path.join(output_root, target)
        global_path = os.path.join(
            tdir,
            f"{target}_global_valence_arousal_contributions.csv"
        )
        phase_path = os.path.join(
            tdir,
            f"{target}_phase_valence_arousal_contributions.csv"
        )

    if not os.path.exists(global_path):
        raise FileNotFoundError(f"Missing global contributions file: {global_path}")
    if not os.path.exists(phase_path):
        raise FileNotFoundError(f"Missing phase contributions file: {phase_path}")

    return global_path, phase_path


def ensure_output_dir(output_root: str) -> str:
    """
    Make a folder <output_root>/valence_arousal_contributions.
    """
    out = os.path.join(output_root, "valence_arousal_contributions")
    os.makedirs(out, exist_ok=True)
    return out


def format_table(df: pd.DataFrame, cols, max_rows: int):
    """
    Convert df[cols] into a readable table string for the TXT report.
    """
    if df.empty:
        return "(no data)\n"

    df = df[cols].head(max_rows).copy()

    for col in df.columns:
        if col.endswith("_pct"):
            df[col] = df[col].map(lambda x: f"{x:.2f}")
        elif col in {
            "valence_dev", "arousal_dev", "delta_rel_freq",
            "valence_contribution", "arousal_contribution",
            "rating", "deviation", "contribution"
        }:
            df[col] = df[col].map(lambda x: f"{x:.6f}")
        elif col == "total_count":
            df[col] = df[col].map(lambda x: f"{x:.0f}")

    return df.to_string(index=False)


def sort_global_tables(global_df: pd.DataFrame):
    """
    Prepare sorted global tables for valence and arousal.
    """
    global_val = global_df.copy()
    global_aro = global_df.copy()

    if "abs_valence_contribution" in global_val.columns:
        global_val = global_val.sort_values("abs_valence_contribution", ascending=False)

    if "abs_arousal_contribution" in global_aro.columns:
        global_aro = global_aro.sort_values("abs_arousal_contribution", ascending=False)

    return global_val, global_aro


def sort_phase_table(phase_df: pd.DataFrame, dimension: str, sign: str):
    """
    Sort a phase subset according to dimension and desired sign.
    sign in {"positive", "negative", "absolute"}.
    """
    sub = phase_df[phase_df["dimension"] == dimension].copy()

    if sub.empty:
        return sub

    if sign == "positive":
        sub = sub[sub["contribution"] > 0].sort_values("contribution", ascending=False)
    elif sign == "negative":
        sub = sub[sub["contribution"] < 0].sort_values("contribution", ascending=True)
    else:
        if "abs_contribution" in sub.columns:
            sub = sub.sort_values("abs_contribution", ascending=False)
        else:
            sub = sub.assign(abs_contribution=sub["contribution"].abs()).sort_values(
                "abs_contribution", ascending=False
            )

    return sub


def choose_default_direction(phase_name: str, dimension: str) -> str:
    """
    Infer which direction is most natural to report for a phase.
    """
    phase_name = phase_name.lower()

    if dimension == "valence":
        if "declin" in phase_name or "lower" in phase_name or "down" in phase_name:
            return "negative"
        if "rebound" in phase_name or "higher" in phase_name or "up" in phase_name:
            return "positive"

    if dimension == "arousal":
        if "lower" in phase_name or "declin" in phase_name or "down" in phase_name:
            return "negative"
        if "higher" in phase_name or "rebound" in phase_name or "up" in phase_name:
            return "positive"

    return "absolute"


def pretty_phase_label(row: pd.Series) -> str:
    """
    Make a human-readable phase header.
    """
    return f"{row['phase']} ({int(row['phase_start'])}–{int(row['phase_end'])})"


def main():
    ap = argparse.ArgumentParser(
        description="Summarise valence/arousal global + phase contributions "
                    "for a single target into a TXT report."
    )
    ap.add_argument(
        "--target", required=True,
        help="Target term name, e.g. 'schizophrenia'"
    )
    ap.add_argument(
        "--output-root", default="output",
        help="Directory where contribution CSVs live."
    )
    ap.add_argument(
        "--flat-output", action="store_true",
        help="Set if previous script placed CSVs directly in output-root."
    )
    ap.add_argument(
        "--top-n", type=int, default=20,
        help="Number of top contributors to show in each section."
    )

    args = ap.parse_args()

    global_path, phase_path = resolve_input_paths(
        args.output_root, args.target, args.flat_output
    )

    global_df = pd.read_csv(global_path)
    phase_df = pd.read_csv(phase_path)

    if global_df.empty:
        raise SystemExit("Global contributions file is empty.")
    if phase_df.empty:
        print("Warning: phase-level contributions file is empty.")

    out_dir = ensure_output_dir(args.output_root)
    txt_path = os.path.join(
        out_dir, f"{args.target}_valence_arousal_summary_phase.txt"
    )

    global_val, global_aro = sort_global_tables(global_df)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Valence & Arousal contributions for target: {args.target}\n")
        f.write("=" * 90 + "\n\n")

        # -----------------------------------------------------------------
        # GLOBAL CONTRIBUTIONS
        # -----------------------------------------------------------------
        f.write("GLOBAL CONTRIBUTIONS\n")
        f.write("-" * 90 + "\n\n")

        if not global_val.empty:
            f.write(f"Top {args.top_n} global contributors to VALENCE\n")
            f.write("-" * 90 + "\n")
            f.write(
                format_table(
                    global_val,
                    [
                        "word",
                        "valence_dev",
                        "delta_rel_freq",
                        "valence_contribution",
                        "abs_valence_contribution",
                        "valence_share_abs_pct",
                        "valence_share_neg_pct",
                        "valence_share_pos_pct",
                        "total_count",
                    ],
                    args.top_n,
                )
            )
            f.write("\n\n")

        if not global_aro.empty:
            f.write(f"Top {args.top_n} global contributors to AROUSAL\n")
            f.write("-" * 90 + "\n")
            f.write(
                format_table(
                    global_aro,
                    [
                        "word",
                        "arousal_dev",
                        "delta_rel_freq",
                        "arousal_contribution",
                        "abs_arousal_contribution",
                        "arousal_share_abs_pct",
                        "arousal_share_neg_pct",
                        "arousal_share_pos_pct",
                        "total_count",
                    ],
                    args.top_n,
                )
            )
            f.write("\n\n")

        # -----------------------------------------------------------------
        # PHASE-LEVEL CONTRIBUTIONS
        # -----------------------------------------------------------------
        f.write("PHASE-LEVEL CONTRIBUTIONS\n")
        f.write("=" * 90 + "\n\n")

        if phase_df.empty:
            f.write("(No phase-level data.)\n")
        else:
            phase_keys = (
                phase_df[["dimension", "phase", "phase_start", "phase_end"]]
                .drop_duplicates()
                .sort_values(["dimension", "phase_start", "phase_end", "phase"])
            )

            for _, phase_row in phase_keys.iterrows():
                dimension = phase_row["dimension"]
                phase = phase_row["phase"]

                df_phase = phase_df[
                    (phase_df["dimension"] == dimension) &
                    (phase_df["phase"] == phase)
                ].copy()

                if df_phase.empty:
                    continue

                direction = choose_default_direction(phase, dimension)
                phase_header = pretty_phase_label(phase_row)

                f.write(f"{dimension.upper()} — {phase_header}\n")
                f.write("-" * 90 + "\n")

                if direction == "negative":
                    title = f"Top {args.top_n} negative contributors"
                elif direction == "positive":
                    title = f"Top {args.top_n} positive contributors"
                else:
                    title = f"Top {args.top_n} contributors by absolute magnitude"

                f.write(title + "\n")
                f.write("-" * 60 + "\n")

                ranked = sort_phase_table(df_phase, dimension, direction)

                share_col = f"{dimension}_share_abs_pct"
                dir_share_col = (
                    f"{dimension}_share_neg_pct"
                    if direction == "negative"
                    else f"{dimension}_share_pos_pct"
                    if direction == "positive"
                    else f"{dimension}_share_abs_pct"
                )

                f.write(
                    format_table(
                        ranked,
                        [
                            "word",
                            "deviation",
                            "delta_rel_freq",
                            "contribution",
                            "abs_contribution",
                            share_col,
                            dir_share_col,
                            "total_count",
                        ],
                        args.top_n,
                    )
                )
                f.write("\n\n")

    print(f"\nTXT summary written to:\n{txt_path}\n")


if __name__ == "__main__":
    main()