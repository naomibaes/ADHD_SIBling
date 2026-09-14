# Authors: Naomi Baes and ChatGPT

import os

import numpy as np
import pandas as pd
from tqdm import tqdm


# =========================================================
# Helper functions
# =========================================================

def load_warriner_ratings(filename):
    """Load Warriner valence and arousal ratings."""
    return pd.read_csv(filename)


def extract_file_details(filename):
    """
    Extract target and year from a filename such as:
    ADHD_US_newsstream_normalized-ADHD_sentences_cleaned_
    lemmatized_1997_cooccurrences.csv
    """
    filename = filename.replace(".csv", "")
    parts = filename.split("_")

    target = parts[0]
    year = int(parts[-2])

    return target, year


def normalize(value, min_value=1, max_value=9):
    """Normalize a Warriner score from the 1–9 scale to 0–1."""
    return (value - min_value) / (max_value - min_value)


# =========================================================
# Compute annual arousal and lexicon coverage
# =========================================================

def compute_arousal_indices(cooccurrences_df, warriner_ratings):
    """
    Calculate the token-weighted average Warriner arousal score
    for one year's ±5-lemma co-occurrences.

    Coverage is calculated as:

        matched neighbour-token instances
        ---------------------------------
        all eligible neighbour-token instances

    Counts are token-weighted rather than based on unique lemmas.
    """

    required_columns = {"cooccurrence", "count"}

    if not required_columns.issubset(cooccurrences_df.columns):
        raise KeyError(
            "Co-occurrence file must contain the columns "
            "'cooccurrence' and 'count'."
        )

    cooccurrences_df = cooccurrences_df.copy()

    cooccurrences_df["cooccurrence"] = (
        cooccurrences_df["cooccurrence"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    cooccurrences_df["count"] = pd.to_numeric(
        cooccurrences_df["count"],
        errors="coerce"
    )

    cooccurrences_df = cooccurrences_df.dropna(
        subset=["count"]
    )

    # Denominator:
    # all eligible token instances in the ±5-lemma context window
    eligible_context_tokens = int(
        cooccurrences_df["count"].sum()
    )

    # Prepare the Warriner arousal lexicon
    arousal_lexicon = (
        warriner_ratings[
            ["word", "A.Mean.Sum"]
        ]
        .copy()
    )

    arousal_lexicon["word"] = (
        arousal_lexicon["word"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    arousal_lexicon["A.Mean.Sum"] = pd.to_numeric(
        arousal_lexicon["A.Mean.Sum"],
        errors="coerce"
    )

    arousal_lexicon = (
        arousal_lexicon
        .dropna(subset=["word", "A.Mean.Sum"])
        .drop_duplicates(subset="word")
    )

    # Match co-occurring lemmas to Warriner arousal ratings
    merged = pd.merge(
        cooccurrences_df,
        arousal_lexicon,
        left_on="cooccurrence",
        right_on="word",
        how="inner"
    )

    # Numerator:
    # token instances successfully assigned an arousal score
    warriner_arousal_covered_tokens = int(
        merged["count"].sum()
    )

    if eligible_context_tokens > 0:
        warriner_arousal_coverage_proportion = (
            warriner_arousal_covered_tokens
            / eligible_context_tokens
        )
    else:
        warriner_arousal_coverage_proportion = np.nan

    if warriner_arousal_covered_tokens > 0:
        # Token-weighted annual arousal index
        merged["weighted_arousal"] = (
            merged["count"]
            * merged["A.Mean.Sum"]
        )

        total_weighted_arousal = (
            merged["weighted_arousal"].sum()
        )

        avg_arousal = (
            total_weighted_arousal
            / warriner_arousal_covered_tokens
        )

        # Token-weighted standard error
        merged["squared_diff"] = (
            (
                merged["A.Mean.Sum"]
                - avg_arousal
            ) ** 2
            * merged["count"]
        )

        variance = (
            merged["squared_diff"].sum()
            / warriner_arousal_covered_tokens
        )

        se_arousal = np.sqrt(
            variance
            / warriner_arousal_covered_tokens
        )

        normalized_arousal = normalize(
            avg_arousal
        )

        normalized_se = (
            se_arousal / (9 - 1)
        )

    else:
        avg_arousal = np.nan
        se_arousal = np.nan
        normalized_arousal = np.nan
        normalized_se = np.nan

    return {
        "avg_arousal_index": avg_arousal,
        "se_arousal_index": se_arousal,
        "normalized_arousal_index": normalized_arousal,
        "normalized_se_arousal_index": normalized_se,
        "eligible_context_tokens": eligible_context_tokens,
        "warriner_arousal_covered_tokens":
            warriner_arousal_covered_tokens,
        "warriner_arousal_coverage_proportion":
            warriner_arousal_coverage_proportion
    }


# =========================================================
# Process all yearly co-occurrence files
# =========================================================

def process_annual_cooccurrences(
    input_dir,
    output_dir,
    warriner_ratings
):
    results = []

    all_files = sorted(
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if filename.endswith("_cooccurrences.csv")
    )

    for file in tqdm(
        all_files,
        desc="Processing co-occurrence files"
    ):
        try:
            cooccurrences_df = pd.read_csv(
                file,
                dtype={"cooccurrence": str}
            )

            target, year = extract_file_details(
                os.path.basename(file)
            )

            annual_result = compute_arousal_indices(
                cooccurrences_df,
                warriner_ratings
            )

            results.append({
                "target": target,
                "year": int(year),
                **annual_result
            })

        except Exception as error:
            print(
                f"Error processing {file}: {error}"
            )

    if not results:
        print("No valid co-occurrence files found.")
        return

    result_df = (
        pd.DataFrame(results)
        .sort_values(["target", "year"])
        .reset_index(drop=True)
    )

    # Restrict to the study period before saving or summarising
    result_df = result_df[
        result_df["year"].between(1990, 2025)
    ].copy()

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_file = os.path.join(
        output_dir,
        "annual_arousal_indices.csv"
    )

    result_df.to_csv(
        output_file,
        index=False
    )

    print(f"\nSaved: {output_file}")

    print("\nOutput columns:")
    print(result_df.columns.tolist())

    print("\nCoverage summary:")
    print(
        result_df[
            "warriner_arousal_coverage_proportion"
        ].describe()
    )


# =========================================================
# Main execution
# =========================================================

warriner_ratings = load_warriner_ratings(
    "/home/ec2-user/SageMaker/resources/warriner_rat.csv"
)

input_dir = "output/cooccurrences_by_year"
output_dir = "output"

process_annual_cooccurrences(
    input_dir,
    output_dir,
    warriner_ratings
)