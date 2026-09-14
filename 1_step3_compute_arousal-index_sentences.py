# Authors: Naomi Baes and ChatGPT

import ast
import os

import numpy as np
import pandas as pd


# =========================================================
# File paths
# =========================================================

sentence_file = (
    "output/"
    "ADHD_US_newsstream_normalized-ADHD_"
    "sentences_cleaned_lemmatized.csv"
)

warriner_file = (
    "/home/ec2-user/SageMaker/resources/warriner_rat.csv"
)

output_file = (
    "output/"
    "annual_arousal_indices_full_sentence.csv"
)


# =========================================================
# Load data
# =========================================================

df = pd.read_csv(
    sentence_file,
    low_memory=False
)

warriner = pd.read_csv(
    warriner_file,
    low_memory=False
)

print("Sentence corpus shape:", df.shape)
print("Sentence corpus columns:", df.columns.tolist())
print("Warriner columns:", warriner.columns.tolist())


# Automatically identify the lemmatised sentence column
lemma_col = next(
    (
        column
        for column in df.columns
        if column.lower().startswith("lemmatized")
    ),
    None
)

if lemma_col is None:
    raise KeyError(
        "Could not identify the lemmatised sentence column. "
        f"Available columns: {df.columns.tolist()}"
    )

if "year" not in df.columns:
    raise KeyError(
        "The sentence corpus does not contain a 'year' column."
    )

print("Using lemma column:", lemma_col)


# =========================================================
# Parse lemmatised sentences
# =========================================================

def parse_lemmas(value):
    """
    Convert the stored lemmatised sentence into a list.

    Handles:
    - actual Python lists;
    - Python-style lists stored as strings;
    - space-separated strings.
    """

    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [
            str(token).strip().lower()
            for token in value
            if str(token).strip()
        ]

    text = str(value).strip()

    if not text:
        return []

    # Example:
    # "['child', 'adhd', 'school']"
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, list):
                return [
                    str(token).strip().lower()
                    for token in parsed
                    if str(token).strip()
                ]

        except (ValueError, SyntaxError):
            pass

    # Fallback for ordinary space-separated strings
    return [
        token.strip().lower()
        for token in text.split()
        if token.strip()
    ]


# =========================================================
# Create one row per eligible full-sentence token
# =========================================================

tokens = df[
    ["year", lemma_col]
].copy()

tokens["lemma"] = tokens[
    lemma_col
].apply(parse_lemmas)

tokens = tokens.explode(
    "lemma"
)

tokens["lemma"] = (
    tokens["lemma"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)

# Exclude blank tokens and the target itself
tokens = tokens[
    tokens["lemma"].ne("")
    & tokens["lemma"].ne("adhd")
].copy()

# Restrict to the study period
tokens = tokens[
    tokens["year"].between(1990, 2025)
].copy()

print(
    "Eligible full-sentence token instances:",
    len(tokens)
)


# =========================================================
# Prepare Warriner arousal ratings
# =========================================================

required_columns = {
    "word",
    "A.Mean.Sum"
}

if not required_columns.issubset(warriner.columns):
    raise KeyError(
        "Warriner file must contain 'word' and 'A.Mean.Sum'."
    )

arousal_lexicon = warriner[
    ["word", "A.Mean.Sum"]
].copy()

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


# =========================================================
# Compute annual full-sentence arousal indices
# =========================================================

results = []

for year, year_tokens in tokens.groupby("year"):

    # Denominator:
    # all eligible token instances in target-containing sentences
    eligible_full_sentence_tokens = len(year_tokens)

    matched = year_tokens.merge(
        arousal_lexicon,
        left_on="lemma",
        right_on="word",
        how="inner"
    )

    # Numerator:
    # token instances with an available Warriner arousal score
    warriner_arousal_covered_tokens = len(matched)

    if eligible_full_sentence_tokens > 0:
        warriner_arousal_coverage_proportion = (
            warriner_arousal_covered_tokens
            / eligible_full_sentence_tokens
        )
    else:
        warriner_arousal_coverage_proportion = np.nan

    if warriner_arousal_covered_tokens > 0:

        scores = matched[
            "A.Mean.Sum"
        ].astype(float)

        avg_arousal = scores.mean()

        # Match the token-weighted SE logic of the original analysis
        variance = (
            ((scores - avg_arousal) ** 2).sum()
            / warriner_arousal_covered_tokens
        )

        se_arousal = np.sqrt(
            variance
            / warriner_arousal_covered_tokens
        )

        normalized_arousal = (
            (avg_arousal - 1)
            / (9 - 1)
        )

        normalized_se_arousal = (
            se_arousal
            / (9 - 1)
        )

    else:
        avg_arousal = np.nan
        se_arousal = np.nan
        normalized_arousal = np.nan
        normalized_se_arousal = np.nan

    results.append({
        "target": "ADHD",
        "year": int(year),
        "avg_arousal_index": avg_arousal,
        "se_arousal_index": se_arousal,
        "normalized_arousal_index": normalized_arousal,
        "normalized_se_arousal_index": normalized_se_arousal,
        "eligible_full_sentence_tokens":
            eligible_full_sentence_tokens,
        "warriner_arousal_covered_tokens":
            warriner_arousal_covered_tokens,
        "warriner_arousal_coverage_proportion":
            warriner_arousal_coverage_proportion
    })


result_df = (
    pd.DataFrame(results)
    .sort_values("year")
    .reset_index(drop=True)
)


# =========================================================
# Save output
# =========================================================

os.makedirs(
    os.path.dirname(output_file),
    exist_ok=True
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

print("\nAnnual results:")
print(result_df)