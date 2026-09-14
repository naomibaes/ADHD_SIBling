# ============================================================
# QUADRATIC TREND + YEAR-SPECIFIC SLOPES
# ADHD Valence (Sentiment) and Arousal (Intensity)
#
# Loads TWO separate annual-index files.
#
# Outputs:
#   1. Full year-by-year slope + CI CSV
#   2. Concise model/phase summary CSV
#   3. Neat printed summary showing only trend periods
# ============================================================

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as t_dist


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

VALENCE_FILE = "output/annual_valence_indices.csv"
AROUSAL_FILE = "output/annual_arousal_indices.csv"

OUTPUT_DIR = "output/descriptives"
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEAR_MIN = 1990
YEAR_MAX = 2025


# ------------------------------------------------------------
# 2. LOAD THE TWO INDEX FILES
# ------------------------------------------------------------

valence_df = pd.read_csv(VALENCE_FILE)
arousal_df = pd.read_csv(AROUSAL_FILE)

# Keep ADHD only, if file contains multiple targets
if "target" in valence_df.columns:
    valence_df = valence_df[
        valence_df["target"].astype(str).str.upper() == "ADHD"
    ].copy()

if "target" in arousal_df.columns:
    arousal_df = arousal_df[
        arousal_df["target"].astype(str).str.upper() == "ADHD"
    ].copy()

# Make year numeric
valence_df["year"] = pd.to_numeric(
    valence_df["year"], errors="coerce"
)

arousal_df["year"] = pd.to_numeric(
    arousal_df["year"], errors="coerce"
)

# IMPORTANT: restrict BOTH analyses to study period
valence_df = (
    valence_df[
        valence_df["year"].between(YEAR_MIN, YEAR_MAX)
    ]
    .dropna(subset=["year", "avg_valence_index"])
    .sort_values("year")
    .copy()
)

arousal_df = (
    arousal_df[
        arousal_df["year"].between(YEAR_MIN, YEAR_MAX)
    ]
    .dropna(subset=["year", "avg_arousal_index"])
    .sort_values("year")
    .copy()
)

valence_df["year"] = valence_df["year"].astype(int)
arousal_df["year"] = arousal_df["year"].astype(int)

print(
    f"Valence data: {valence_df['year'].min()}–"
    f"{valence_df['year'].max()} "
    f"({len(valence_df)} years)"
)

print(
    f"Arousal data: {arousal_df['year'].min()}–"
    f"{arousal_df['year'].max()} "
    f"({len(arousal_df)} years)"
)

print()


# ------------------------------------------------------------
# 3. HELPER: COLLAPSE CONSECUTIVE YEARS
# ------------------------------------------------------------

def collapse_year_ranges(years):

    years = sorted([int(y) for y in years])

    if len(years) == 0:
        return "None"

    ranges = []
    start = previous = years[0]

    for year in years[1:]:

        if year == previous + 1:
            previous = year

        else:
            ranges.append((start, previous))
            start = previous = year

    ranges.append((start, previous))

    output = []

    for start, end in ranges:

        if start == end:
            output.append(str(start))

        else:
            output.append(f"{start}–{end}")

    return ", ".join(output)


# ------------------------------------------------------------
# 4. QUADRATIC MODEL + YEAR-SPECIFIC SLOPES
# ------------------------------------------------------------

def quadratic_slope_analysis(
    data,
    outcome_col,
    measure_name
):

    d = data[["year", outcome_col]].dropna().copy()

    # Centre year for numerical stability
    year_center = d["year"].mean()

    d["year_c"] = d["year"] - year_center
    d["year_c2"] = d["year_c"] ** 2

    # --------------------------------------------
    # Fit quadratic model
    #
    # Y = b0 + b1(year) + b2(year²)
    # --------------------------------------------

    X = sm.add_constant(
        d[["year_c", "year_c2"]]
    )

    y = d[outcome_col]

    model = sm.OLS(y, X).fit()

    b0 = model.params["const"]
    b1 = model.params["year_c"]
    b2 = model.params["year_c2"]

    cov = model.cov_params()

    df_resid = int(model.df_resid)

    t_crit = t_dist.ppf(
        0.975,
        df=df_resid
    )

    yearly_results = []

    # --------------------------------------------
    # Calculate slope at every observed year
    #
    # slope = b1 + 2*b2*t
    # --------------------------------------------

    for _, row in d.iterrows():

        year = int(row["year"])
        t = row["year_c"]

        slope = b1 + (2 * b2 * t)

        slope_var = (
            cov.loc["year_c", "year_c"]
            +
            (4 * (t ** 2) *
             cov.loc["year_c2", "year_c2"])
            +
            (4 * t *
             cov.loc["year_c", "year_c2"])
        )

        slope_se = np.sqrt(slope_var)

        ci_low = slope - (t_crit * slope_se)
        ci_high = slope + (t_crit * slope_se)

        t_value = slope / slope_se

        p_value = 2 * (
            1 - t_dist.cdf(
                abs(t_value),
                df=df_resid
            )
        )

        # ----------------------------------------
        # Classify direction
        # ----------------------------------------

        if ci_high < 0:

            direction = "Significant decrease"

        elif ci_low > 0:

            direction = "Significant increase"

        else:

            direction = "No significant change"

        yearly_results.append({
            "measure": measure_name,
            "year": year,
            "fitted_slope": slope,
            "slope_se": slope_se,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "t": t_value,
            "p": p_value,
            "direction": direction
        })

    slope_df = pd.DataFrame(yearly_results)

    # --------------------------------------------
    # Quadratic turning point
    # --------------------------------------------

    if b2 != 0:

        turning_point_c = -b1 / (2 * b2)
        turning_point_year = (
            year_center + turning_point_c
        )

    else:

        turning_point_year = np.nan

    # --------------------------------------------
    # Get significant periods
    # --------------------------------------------

    decrease_years = slope_df.loc[
        slope_df["direction"] ==
        "Significant decrease",
        "year"
    ].tolist()

    stable_years = slope_df.loc[
        slope_df["direction"] ==
        "No significant change",
        "year"
    ].tolist()

    increase_years = slope_df.loc[
        slope_df["direction"] ==
        "Significant increase",
        "year"
    ].tolist()

    summary = {

        "measure": measure_name,

        "n_years": len(d),

        "linear_b": b1,
        "linear_p": model.pvalues["year_c"],

        "quadratic_b": b2,
        "quadratic_p": model.pvalues["year_c2"],

        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,

        "F": model.fvalue,
        "F_p": model.f_pvalue,

        "turning_point_year":
            turning_point_year,

        "significant_decrease_years":
            collapse_year_ranges(
                decrease_years
            ),

        "non_significant_years":
            collapse_year_ranges(
                stable_years
            ),

        "significant_increase_years":
            collapse_year_ranges(
                increase_years
            )
    }

    return model, slope_df, summary


# ------------------------------------------------------------
# 5. RUN VALENCE
# ------------------------------------------------------------

valence_model, valence_slopes, valence_summary = (
    quadratic_slope_analysis(
        valence_df,
        outcome_col="avg_valence_index",
        measure_name="Valence / Sentiment"
    )
)


# ------------------------------------------------------------
# 6. RUN AROUSAL
# ------------------------------------------------------------

arousal_model, arousal_slopes, arousal_summary = (
    quadratic_slope_analysis(
        arousal_df,
        outcome_col="avg_arousal_index",
        measure_name="Arousal / Intensity"
    )
)


# ------------------------------------------------------------
# 7. COMBINE + SAVE COMPLETE YEAR-BY-YEAR RESULTS
# ------------------------------------------------------------

all_slopes = pd.concat(
    [
        valence_slopes,
        arousal_slopes
    ],
    ignore_index=True
)

slope_file = os.path.join(
    OUTPUT_DIR,
    "ADHD_valence_arousal_yearly_quadratic_slopes.csv"
)

all_slopes.to_csv(
    slope_file,
    index=False
)


# ------------------------------------------------------------
# 8. SAVE CONCISE MODEL SUMMARY
# ------------------------------------------------------------

summary_df = pd.DataFrame([
    valence_summary,
    arousal_summary
])

summary_file = os.path.join(
    OUTPUT_DIR,
    "ADHD_valence_arousal_quadratic_phase_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ------------------------------------------------------------
# 9. PRINT NEAT SUMMARY
# ------------------------------------------------------------

def print_measure_summary(summary):

    print("=" * 70)
    print(summary["measure"].upper())
    print("=" * 70)

    print(
        f"Quadratic model: "
        f"R² = {summary['r_squared']:.3f}; "
        f"adjusted R² = "
        f"{summary['adj_r_squared']:.3f}"
    )

    print(
        f"Linear term: "
        f"b = {summary['linear_b']:.6f}, "
        f"p = {summary['linear_p']:.4g}"
    )

    print(
        f"Quadratic term: "
        f"b = {summary['quadratic_b']:.8f}, "
        f"p = {summary['quadratic_p']:.4g}"
    )

    print(
        f"Estimated turning point: "
        f"{summary['turning_point_year']:.2f}"
    )

    print()

    print("SIGNIFICANT DIRECTION OF FITTED TREND")

    print(
        "  Decreasing : "
        f"{summary['significant_decrease_years']}"
    )

    print(
        "  No clear slope: "
        f"{summary['non_significant_years']}"
    )

    print(
        "  Increasing : "
        f"{summary['significant_increase_years']}"
    )

    print()


print_measure_summary(
    valence_summary
)

print_measure_summary(
    arousal_summary
)


# ------------------------------------------------------------
# 10. PRINT ONLY POINTS WHERE CLASSIFICATION CHANGES
# ------------------------------------------------------------

print("=" * 70)
print("TREND-PHASE TRANSITIONS")
print("=" * 70)

for measure, measure_df in all_slopes.groupby(
    "measure",
    sort=False
):

    measure_df = (
        measure_df
        .sort_values("year")
        .copy()
    )

    measure_df["previous_direction"] = (
        measure_df["direction"].shift(1)
    )

    transitions = measure_df[
        measure_df["direction"] !=
        measure_df["previous_direction"]
    ]

    print(f"\n{measure}")

    for _, row in transitions.iterrows():

        print(
            f"  {int(row['year'])}: "
            f"{row['direction']} | "
            f"slope = {row['fitted_slope']:.5f}, "
            f"95% CI "
            f"[{row['ci_low']:.5f}, "
            f"{row['ci_high']:.5f}], "
            f"p = {row['p']:.4g}"
        )


# ------------------------------------------------------------
# 11. FILE LOCATIONS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    "\nFull year-by-year results:\n"
    f"{slope_file}"
)

print(
    "\nConcise phase summary:\n"
    f"{summary_file}"
)