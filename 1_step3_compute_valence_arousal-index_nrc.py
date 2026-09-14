"""Compute annual NRC-VAD valence and arousal indices.

This is the NRC-VAD v2.1 counterpart of the Warriner-based index script.
It retains the existing lemma-frequency weighting and coverage denominator,
but matches co-occurring lemmas to the combined NRC-VAD unigram file.
"""

import os
import re

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        """Use the iterable unchanged when tqdm is unavailable."""
        return iterable


def load_nrc_vad_lexicon(filename):
    """Load and validate the combined NRC-VAD v2.1 unigram lexicon."""
    # keep_default_na=False preserves the lexicon's literal unigram "null"
    # instead of allowing pandas to interpret it as a missing value.
    lexicon = pd.read_csv(filename, sep="\t", keep_default_na=False)

    required_columns = {"term", "valence", "arousal"}
    if not required_columns.issubset(lexicon.columns):
        raise KeyError(
            "NRC-VAD unigram file must contain the columns "
            "'term', 'valence', and 'arousal'."
        )

    lexicon = lexicon[["term", "valence", "arousal"]].copy()
    lexicon["term"] = (
        lexicon["term"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    for dimension in ("valence", "arousal"):
        lexicon[dimension] = pd.to_numeric(
            lexicon[dimension], errors="coerce"
        )

    lexicon = (
        lexicon
        .dropna(subset=["valence", "arousal"])
        .loc[lambda frame: frame["term"].ne("")]
        .drop_duplicates(subset="term")
        .reset_index(drop=True)
    )

    for dimension in ("valence", "arousal"):
        if not lexicon[dimension].between(-1, 1).all():
            raise ValueError(
                f"NRC-VAD {dimension} scores must fall between -1 and 1."
            )

    return lexicon


def extract_file_details(filename):
    """Extract the target and year from a yearly co-occurrence filename."""
    match = re.search(r"_([12]\d{3})_cooccurrences\.csv$", filename)
    if match is None:
        raise ValueError(
            "Could not extract year from co-occurrence filename: "
            f"{filename}"
        )

    target = filename.split("_")[0]
    year = int(match.group(1))
    return target, year


def normalize_nrc_score(value):
    """Rescale an NRC-VAD score from -1--1 to 0--1."""
    return (value + 1) / 2


def compute_weighted_dimension(merged_df, dimension, eligible_tokens):
    """Compute one token-weighted NRC-VAD index, SE, and coverage."""
    dimension_df = merged_df.dropna(subset=[dimension]).copy()
    covered_tokens = int(dimension_df["count"].sum())

    coverage_proportion = (
        covered_tokens / eligible_tokens
        if eligible_tokens > 0
        else np.nan
    )

    if covered_tokens == 0:
        return {
            "average": np.nan,
            "standard_error": np.nan,
            "normalized_average": np.nan,
            "normalized_standard_error": np.nan,
            "covered_tokens": 0,
            "coverage_proportion": coverage_proportion,
        }

    average = (
        (dimension_df["count"] * dimension_df[dimension]).sum()
        / covered_tokens
    )

    weighted_squared_difference = (
        dimension_df["count"]
        * (dimension_df[dimension] - average) ** 2
    )
    variance = weighted_squared_difference.sum() / covered_tokens
    standard_error = np.sqrt(variance / covered_tokens)

    # The -1--1 NRC scale spans two units, so both the index and its SE
    # are transformed to the 0--1 scale using that range.
    normalized_average = normalize_nrc_score(average)
    normalized_standard_error = standard_error / 2

    return {
        "average": average,
        "standard_error": standard_error,
        "normalized_average": normalized_average,
        "normalized_standard_error": normalized_standard_error,
        "covered_tokens": covered_tokens,
        "coverage_proportion": coverage_proportion,
    }


def compute_valence_arousal_indices(cooccurrences_df, nrc_vad_lexicon):
    """Compute annual token-weighted NRC valence and arousal indices.

    Coverage denominator:
        all eligible neighbour-token instances in the +/-5-lemma window

    Coverage numerator:
        neighbour-token instances matched to an NRC-VAD unigram

    The ``count`` column ensures that calculations are token-weighted rather
    than based on unique co-occurring lemmas.
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
        cooccurrences_df["count"], errors="coerce"
    )
    cooccurrences_df = cooccurrences_df.dropna(subset=["count"])
    cooccurrences_df = cooccurrences_df.loc[
        cooccurrences_df["count"] > 0
    ]

    eligible_context_tokens = int(cooccurrences_df["count"].sum())

    # A left join preserves the full eligible-token denominator. Unmatched
    # lemmas retain missing NRC scores and are excluded only from numerators.
    merged = pd.merge(
        cooccurrences_df,
        nrc_vad_lexicon,
        left_on="cooccurrence",
        right_on="term",
        how="left",
        validate="many_to_one",
    )

    valence = compute_weighted_dimension(
        merged, "valence", eligible_context_tokens
    )
    arousal = compute_weighted_dimension(
        merged, "arousal", eligible_context_tokens
    )

    return {
        "avg_valence_index": valence["average"],
        "se_valence_index": valence["standard_error"],
        "normalized_valence_index": valence["normalized_average"],
        "normalized_se_valence_index": valence[
            "normalized_standard_error"
        ],
        "avg_arousal_index": arousal["average"],
        "se_arousal_index": arousal["standard_error"],
        "normalized_arousal_index": arousal["normalized_average"],
        "normalized_se_arousal_index": arousal[
            "normalized_standard_error"
        ],
        "eligible_context_tokens": eligible_context_tokens,
        "nrc_valence_covered_tokens": valence["covered_tokens"],
        "nrc_valence_coverage_proportion": valence[
            "coverage_proportion"
        ],
        "nrc_arousal_covered_tokens": arousal["covered_tokens"],
        "nrc_arousal_coverage_proportion": arousal[
            "coverage_proportion"
        ],
    }


def process_annual_cooccurrences(
    input_dir,
    output_dir,
    nrc_vad_lexicon,
):
    """Process every yearly co-occurrence file and save two NRC outputs."""
    results = []

    all_files = sorted(
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if filename.endswith("_cooccurrences.csv")
    )

    for file in tqdm(all_files, desc="Processing co-occurrence files"):
        try:
            cooccurrences_df = pd.read_csv(
                file,
                dtype={"cooccurrence": str},
            )
            target, year = extract_file_details(os.path.basename(file))
            annual_result = compute_valence_arousal_indices(
                cooccurrences_df,
                nrc_vad_lexicon,
            )
            results.append(
                {
                    "target": target,
                    "year": int(year),
                    **annual_result,
                }
            )
        except Exception as error:
            print(f"Error processing {file}: {error}")

    if not results:
        print("No valid co-occurrence files found.")
        return

    result_df = (
        pd.DataFrame(results)
        .sort_values(["target", "year"])
        .reset_index(drop=True)
    )

    os.makedirs(output_dir, exist_ok=True)

    valence_columns = [
        "target",
        "year",
        "avg_valence_index",
        "se_valence_index",
        "normalized_valence_index",
        "normalized_se_valence_index",
        "eligible_context_tokens",
        "nrc_valence_covered_tokens",
        "nrc_valence_coverage_proportion",
    ]
    arousal_columns = [
        "target",
        "year",
        "avg_arousal_index",
        "se_arousal_index",
        "normalized_arousal_index",
        "normalized_se_arousal_index",
        "eligible_context_tokens",
        "nrc_arousal_covered_tokens",
        "nrc_arousal_coverage_proportion",
    ]

    valence_output_file = os.path.join(
        output_dir,
        "annual_valence_indices_nrc.csv",
    )
    arousal_output_file = os.path.join(
        output_dir,
        "annual_arousal_indices_nrc.csv",
    )

    result_df[valence_columns].to_csv(valence_output_file, index=False)
    result_df[arousal_columns].to_csv(arousal_output_file, index=False)

    print(f"\nSaved: {valence_output_file}")
    print(f"Saved: {arousal_output_file}")

    print("\nNRC valence coverage summary:")
    print(result_df["nrc_valence_coverage_proportion"].describe())

    print("\nNRC arousal coverage summary:")
    print(result_df["nrc_arousal_coverage_proportion"].describe())


if __name__ == "__main__":
    NRC_VAD_FILE = (
        "/home/ec2-user/SageMaker/resources/NRC-VAD/"
        "unigrams-NRC-VAD-Lexicon-v2.1.txt"
    )
    INPUT_DIR = "output/cooccurrences_by_year"
    OUTPUT_DIR = "output"

    nrc_vad_lexicon = load_nrc_vad_lexicon(NRC_VAD_FILE)
    process_annual_cooccurrences(
        INPUT_DIR,
        OUTPUT_DIR,
        nrc_vad_lexicon,
    )
