#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarise phase-specific Valence/Arousal neighbour contributions.

Designed for the outputs of the revised contributor script:
    <target>_phase_valence_arousal_contributions.csv
    <target>_phase_contribution_summary.csv

Optionally reads:
    <target>_global_valence_arousal_contributions.csv

The phase definitions have already been selected upstream from the
quadratic-slope analysis, so this script does NOT redefine or truncate them.

Outputs:
    <target>_phase_contribution_report.txt
    <target>_top_phase_contributors.csv
"""

import os
import argparse
import pandas as pd


def resolve_input_dir(output_root: str, target: str, flat_output: bool) -> str:
    """Return the directory containing contribution CSVs."""
    return output_root if flat_output else os.path.join(output_root, target)


def format_p(p):
    if pd.isna(p):
        return "NA"
    p = float(p)
    if p < 0.001:
        return "< .001"
    return f"= {p:.3f}"


def fmt_num(x, digits=6):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Summarise phase-specific Valence/Arousal neighbour contributions "
            "from the revised regression-based contributor analysis."
        )
    )

    ap.add_argument(
        "--target",
        required=True,
        help="Target name used in the contribution files, e.g. ADHD.",
    )
    ap.add_argument(
        "--output-root",
        default="output/valence_arousal_contributions",
        help="Directory containing the contribution CSVs.",
    )
    ap.add_argument(
        "--flat-output",
        action="store_true",
        help="Set if contribution CSVs were written directly under --output-root.",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of same-direction contributors to show per phase.",
    )
    ap.add_argument(
        "--opposing-n",
        type=int,
        default=5,
        help="Number of strongest opposing contributors to show per phase.",
    )
    ap.add_argument(
        "--include-global",
        action="store_true",
        help="Also mention the auxiliary whole-span endpoint file if present.",
    )

    args = ap.parse_args()

    input_dir = resolve_input_dir(
        args.output_root,
        args.target,
        args.flat_output,
    )

    phase_path = os.path.join(
        input_dir,
        f"{args.target}_phase_valence_arousal_contributions.csv",
    )
    summary_path = os.path.join(
        input_dir,
        f"{args.target}_phase_contribution_summary.csv",
    )
    global_path = os.path.join(
        input_dir,
        f"{args.target}_global_valence_arousal_contributions.csv",
    )

    if not os.path.exists(phase_path):
        raise FileNotFoundError(
            f"Missing phase contribution file:\n{phase_path}\n"
            "Run the revised contributor script successfully first."
        )

    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Missing phase summary file:\n{summary_path}\n"
            "Run the revised contributor script successfully first."
        )

    phase_df = pd.read_csv(phase_path)
    summary_df = pd.read_csv(summary_path)

    required_phase = {
        "dimension", "phase", "word", "rating", "deviation",
        "rel_freq_slope_per_year", "estimated_delta_rel_freq",
        "contribution", "share_negative_pct", "share_positive_pct",
        "total_count",
    }
    missing_phase = required_phase.difference(phase_df.columns)
    if missing_phase:
        raise ValueError(
            "Phase contribution file is missing required columns: "
            f"{sorted(missing_phase)}"
        )

    required_summary = {
        "dimension", "phase", "observed_start_year", "observed_end_year",
        "aggregate_index_slope_per_year", "aggregate_index_slope_p",
        "aggregate_fitted_change", "sum_word_contributions",
        "decomposition_gap", "top5_same_direction_share_pct",
        "top10_same_direction_share_pct",
    }
    missing_summary = required_summary.difference(summary_df.columns)
    if missing_summary:
        raise ValueError(
            "Phase summary file is missing required columns: "
            f"{sorted(missing_summary)}"
        )

    # Prepare output paths in the same folder as the inputs.
    report_path = os.path.join(
        input_dir,
        f"{args.target}_phase_contribution_report.txt",
    )
    top_csv_path = os.path.join(
        input_dir,
        f"{args.target}_top_phase_contributors.csv",
    )

    report_lines = []
    export_chunks = []

    report_lines.append(
        f"Phase-specific Valence/Arousal neighbour contributions: {args.target}"
    )
    report_lines.append("=" * 86)
    report_lines.append(
        "Phases were defined upstream from statistically directional portions "
        "of the fitted quadratic Valence/Arousal trajectories."
    )
    report_lines.append(
        "Within each phase, each neighbour's annual relative frequency was "
        "regressed on year; its fitted frequency change was multiplied by its "
        "Valence/Arousal deviation to estimate its contribution."
    )
    report_lines.append("")

    print("\n" + "=" * 78)
    print(f"PHASE CONTRIBUTION SUMMARY: {args.target}")
    print("=" * 78)

    for _, s in summary_df.iterrows():
        dimension = str(s["dimension"])
        phase = str(s["phase"])
        start = int(s["observed_start_year"])
        end = int(s["observed_end_year"])
        aggregate_change = float(s["aggregate_fitted_change"])

        sub = phase_df[
            (phase_df["dimension"].astype(str) == dimension)
            & (phase_df["phase"].astype(str) == phase)
        ].copy()

        if sub.empty:
            continue

        if aggregate_change < 0:
            same = sub[sub["contribution"] < 0].sort_values(
                "contribution", ascending=True
            )
            opposing = sub[sub["contribution"] > 0].sort_values(
                "contribution", ascending=False
            )
            share_col = "share_negative_pct"
            direction_label = "decrease"
        elif aggregate_change > 0:
            same = sub[sub["contribution"] > 0].sort_values(
                "contribution", ascending=False
            )
            opposing = sub[sub["contribution"] < 0].sort_values(
                "contribution", ascending=True
            )
            share_col = "share_positive_pct"
            direction_label = "increase"
        else:
            same = sub.reindex(
                sub["contribution"].abs().sort_values(ascending=False).index
            )
            opposing = sub.iloc[0:0].copy()
            share_col = "share_abs_pct"
            direction_label = "change"

        top_same = same.head(args.top_n).copy()
        top_opposing = opposing.head(args.opposing_n).copy()

        top_same["role"] = "same_direction"
        top_same["rank"] = range(1, len(top_same) + 1)

        if not top_opposing.empty:
            top_opposing["role"] = "opposing"
            top_opposing["rank"] = range(1, len(top_opposing) + 1)

        export_chunks.append(top_same)
        if not top_opposing.empty:
            export_chunks.append(top_opposing)

        heading = f"{dimension.upper()} — {phase} ({start}–{end})"
        print("\n" + heading)
        print("-" * len(heading))
        print(
            f"Aggregate fitted change: {aggregate_change:+.6f}; "
            f"slope/year = {float(s['aggregate_index_slope_per_year']):+.6f}, "
            f"p {format_p(s['aggregate_index_slope_p'])}"
        )
        print(
            f"Decomposition: word contributions sum to "
            f"{float(s['sum_word_contributions']):+.6f}; "
            f"gap = {float(s['decomposition_gap']):.3e}"
        )
        print(
            f"Top 5 account for "
            f"{float(s['top5_same_direction_share_pct']):.2f}% and "
            f"top 10 for {float(s['top10_same_direction_share_pct']):.2f}% "
            f"of same-direction contribution."
        )

        print(f"Top {min(args.top_n, len(top_same))} contributors to the {direction_label}:")
        for i, (_, r) in enumerate(top_same.iterrows(), start=1):
            print(
                f"  {i:>2}. {str(r['word']):<18} "
                f"contribution={float(r['contribution']):+.6f} | "
                f"share={float(r[share_col]):5.2f}% | "
                f"Δfreq_hat={float(r['estimated_delta_rel_freq']):+.6f} | "
                f"rating={float(r['rating']):.2f}"
            )

        report_lines.append(heading)
        report_lines.append("-" * len(heading))
        report_lines.append(
            f"Aggregate fitted change: {aggregate_change:+.6f}"
        )
        report_lines.append(
            f"Aggregate linear slope/year: "
            f"{float(s['aggregate_index_slope_per_year']):+.6f} "
            f"(p {format_p(s['aggregate_index_slope_p'])})"
        )
        report_lines.append(
            f"Sum of word contributions: "
            f"{float(s['sum_word_contributions']):+.6f}"
        )
        report_lines.append(
            f"Decomposition gap: {float(s['decomposition_gap']):.3e}"
        )
        report_lines.append(
            f"Top 5 same-direction share: "
            f"{float(s['top5_same_direction_share_pct']):.2f}%"
        )
        report_lines.append(
            f"Top 10 same-direction share: "
            f"{float(s['top10_same_direction_share_pct']):.2f}%"
        )
        report_lines.append("")

        report_lines.append(
            f"Top {min(args.top_n, len(top_same))} same-direction contributors:"
        )
        report_lines.append(
            "rank  word                contribution   share%   "
            "delta_freq_hat   rating   total_count"
        )

        for i, (_, r) in enumerate(top_same.iterrows(), start=1):
            report_lines.append(
                f"{i:>4}  {str(r['word']):<18} "
                f"{float(r['contribution']):>+12.6f} "
                f"{float(r[share_col]):>8.2f} "
                f"{float(r['estimated_delta_rel_freq']):>+14.6f} "
                f"{float(r['rating']):>8.2f} "
                f"{float(r['total_count']):>12.0f}"
            )

        if not top_opposing.empty:
            report_lines.append("")
            report_lines.append(
                f"Top {min(args.opposing_n, len(top_opposing))} opposing contributors:"
            )
            report_lines.append(
                "rank  word                contribution   "
                "delta_freq_hat   rating   total_count"
            )
            for i, (_, r) in enumerate(top_opposing.iterrows(), start=1):
                report_lines.append(
                    f"{i:>4}  {str(r['word']):<18} "
                    f"{float(r['contribution']):>+12.6f} "
                    f"{float(r['estimated_delta_rel_freq']):>+14.6f} "
                    f"{float(r['rating']):>8.2f} "
                    f"{float(r['total_count']):>12.0f}"
                )

        report_lines.append("")
        report_lines.append("")

    if export_chunks:
        top_df = pd.concat(export_chunks, ignore_index=True)
        keep_cols = [
            "target", "dimension", "phase", "phase_start", "phase_end",
            "observed_start_year", "observed_end_year",
            "role", "rank", "word", "rating", "deviation",
            "rel_freq_slope_per_year", "rel_freq_slope_p",
            "estimated_delta_rel_freq", "contribution",
            "share_abs_pct", "share_negative_pct", "share_positive_pct",
            "total_count",
        ]
        keep_cols = [c for c in keep_cols if c in top_df.columns]
        top_df[keep_cols].to_csv(top_csv_path, index=False)
    else:
        pd.DataFrame().to_csv(top_csv_path, index=False)

    if args.include_global:
        report_lines.append("AUXILIARY WHOLE-SPAN ENDPOINT FILE")
        report_lines.append("-" * 38)
        if os.path.exists(global_path):
            report_lines.append(
                f"Available at: {global_path}"
            )
            report_lines.append(
                "This is retained as an auxiliary output only; "
                "the phase-specific regression analysis is preferred for "
                "reviewer-facing interpretation of nonlinear trends."
            )
        else:
            report_lines.append("No auxiliary global file found.")
        report_lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n" + "=" * 78)
    print("FILES WRITTEN")
    print("=" * 78)
    print(f"Readable TXT report:\n  {report_path}")
    print(f"Top-contributors CSV:\n  {top_csv_path}")
    print(f"Source phase details:\n  {phase_path}")
    print(f"Source phase summary:\n  {summary_path}")


if __name__ == "__main__":
    main()
