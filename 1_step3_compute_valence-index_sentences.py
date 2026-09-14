import ast
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

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
    "annual_valence_indices_full_sentence.csv"
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(sentence_file, low_memory=False)
warriner = pd.read_csv(warriner_file)

# Automatically find the lemmatised-sentence column
lemma_col = next(
    col for col in df.columns
    if col.lower().startswith("lemmatized")
)

print("Using lemma column:", lemma_col)


# ---------------------------------------------------------
# Parse stored lemma lists
# ---------------------------------------------------------

def parse_lemmas(value):
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    text = str(value).strip()

    if not text:
        return []

    # Handles strings such as:
    # "['child', 'hyperactive', 'adhd', 'school']"
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass

    # Fallback for space-separated lemmas
    return text.split()


# ---------------------------------------------------------
# Create one row per eligible token
# ---------------------------------------------------------

tokens = df[["year", lemma_col]].copy()
tokens["lemma"] = tokens[lemma_col].apply(parse_lemmas)
tokens = tokens.explode("lemma")

tokens["lemma"] = (
    tokens["lemma"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)

# Exclude blanks and the target itself
tokens = tokens[
    tokens["lemma"].ne("")
    & tokens["lemma"].ne("adhd")
].copy()


# ---------------------------------------------------------
# Prepare Warriner lexicon
# ---------------------------------------------------------

warriner["word"] = (
    warriner["word"]
    .astype(str)
    .str.lower()
    .str.strip()
)

# Ensure one rating per lemma
warriner = warriner[
    ["word", "V.Mean.Sum"]
].drop_duplicates(subset="word")


# ---------------------------------------------------------
# Calculate annual full-sentence valence indices
# ---------------------------------------------------------

results = []

for year, year_tokens in tokens.groupby("year"):
    eligible_count = len(year_tokens)

    matched = year_tokens.merge(
        warriner,
        left_on="lemma",
        right_on="word",
        how="inner"
    )

    covered_count = len(matched)

    if covered_count > 0:
        scores = matched["V.Mean.Sum"].astype(float)

        avg_valence = scores.mean()

        # Matches the logic of the current script
        se_valence = (
            scores.std(ddof=0)
            / np.sqrt(covered_count)
        )

        normalized_valence = (
            (avg_valence - 1) / (9 - 1)
        )

        normalized_se = se_valence / (9 - 1)

    else:
        avg_valence = np.nan
        se_valence = np.nan
        normalized_valence = np.nan
        normalized_se = np.nan

    coverage = (
        covered_count / eligible_count
        if eligible_count > 0
        else np.nan
    )

    results.append({
        "target": "ADHD",
        "year": int(year),
        "eligible_full_sentence_tokens": eligible_count,
        "warriner_valence_covered_tokens": covered_count,
        "warriner_valence_coverage_proportion": coverage,
        "avg_valence_index": avg_valence,
        "se_valence_index": se_valence,
        "normalized_valence_index": normalized_valence,
        "normalized_se_valence_index": normalized_se
    })


result_df = (
    pd.DataFrame(results)
    .sort_values("year")
)

os.makedirs(
    os.path.dirname(output_file),
    exist_ok=True
)

result_df.to_csv(
    output_file,
    index=False
)

print(result_df)
print(f"\nSaved to: {output_file}")