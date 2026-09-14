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
# Compute annual valence and lexicon coverage
# =========================================================

def compute_valence_indices(cooccurrences_df, warriner_ratings):
    """
    Calculate the token-weighted average Warriner valence score
    for one year's ±5-lemma co-occurrences.

    Coverage is:

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

    # Denominator: every eligible token in the ±5-lemma window
    eligible_context_tokens = int(
        cooccurrences_df["count"].sum()
    )

    valence_lexicon = (
        warriner_ratings[
            ["word", "V.Mean.Sum"]
        ]
        .copy()
    )

    valence_lexicon["word"] = (
        valence_lexicon["word"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    valence_lexicon["V.Mean.Sum"] = pd.to_numeric(
        valence_lexicon["V.Mean.Sum"],
        errors="coerce"
    )

    valence_lexicon = (
        valence_lexicon
        .dropna(subset=["word", "V.Mean.Sum"])
        .drop_duplicates(subset="word")
    )

    merged = pd.merge(
        cooccurrences_df,
        valence_lexicon,
        left_on="cooccurrence",
        right_on="word",
        how="inner"
    )

    # Numerator: token instances successfully assigned a valence score
    warriner_valence_covered_tokens = int(
        merged["count"].sum()
    )

    if eligible_context_tokens > 0:
        warriner_valence_coverage_proportion = (
            warriner_valence_covered_tokens
            / eligible_context_tokens
        )
    else:
        warriner_valence_coverage_proportion = np.nan

    if warriner_valence_covered_tokens > 0:
        merged["weighted_valence"] = (
            merged["count"]
            * merged["V.Mean.Sum"]
        )

        total_weighted_valence = (
            merged["weighted_valence"].sum()
        )

        avg_valence = (
            total_weighted_valence
            / warriner_valence_covered_tokens
        )

        # Token-weighted standard error
        merged["squared_diff"] = (
            (
                merged["V.Mean.Sum"]
                - avg_valence
            ) ** 2
            * merged["count"]
        )

        variance = (
            merged["squared_diff"].sum()
            / warriner_valence_covered_tokens
        )

        se_valence = np.sqrt(
            variance
            / warriner_valence_covered_tokens
        )

        normalized_valence = normalize(
            avg_valence
        )

        normalized_se = (
            se_valence / (9 - 1)
        )

    else:
        avg_valence = np.nan
        se_valence = np.nan
        normalized_valence = np.nan
        normalized_se = np.nan

    return {
        "avg_valence_index": avg_valence,
        "se_valence_index": se_valence,
        "normalized_valence_index": normalized_valence,
        "normalized_se_valence_index": normalized_se,
        "eligible_context_tokens": eligible_context_tokens,
        "warriner_valence_covered_tokens":
            warriner_valence_covered_tokens,
        "warriner_valence_coverage_proportion":
            warriner_valence_coverage_proportion
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

            annual_result = compute_valence_indices(
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

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_file = os.path.join(
        output_dir,
        "annual_valence_indices.csv"
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
            "warriner_valence_coverage_proportion"
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