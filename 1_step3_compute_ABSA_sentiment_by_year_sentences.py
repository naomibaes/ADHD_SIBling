# Authors: Naomi Baes and ChatGPT

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# =========================================================
# Settings
# =========================================================

INPUT_FILE = Path(
    "output/"
    "ADHD_US_newsstream_normalized-ADHD_"
    "sentences_cleaned_lemmatized.csv"
)

MODEL_PATH = Path(
    "/home/ec2-user/SageMaker/models/"
    "deberta-v3-base-absa-v1.1"
)

ASPECT = "ADHD"

MIN_YEAR = 1990
MAX_YEAR = 2025

MAX_LENGTH = 512
BATCH_SIZE = 16

# Run 64 of the longest sentences first to test the
# target-centred truncation.
# Change to None for the complete analysis.
TEST_N = None
TEST_LONGEST_SENTENCES = True

if TEST_N is None:
    OUTPUT_FILE = Path(
        "output/annual_ABSA_sentiment_indices.csv"
    )
else:
    OUTPUT_FILE = Path(
        "output/annual_ABSA_sentiment_indices_TEST.csv"
    )


# =========================================================
# Check local model folder
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model folder not found: {MODEL_PATH}"
    )

required_model_files = [
    "config.json",
    "spm.model",
    "tokenizer_config.json",
]

missing_model_files = [
    filename
    for filename in required_model_files
    if not (MODEL_PATH / filename).exists()
]

has_model_weights = (
    (MODEL_PATH / "model.safetensors").exists()
    or (MODEL_PATH / "pytorch_model.bin").exists()
)

if missing_model_files:
    raise FileNotFoundError(
        f"Missing model files: {missing_model_files}"
    )

if not has_model_weights:
    raise FileNotFoundError(
        "The model folder does not contain either "
        "'model.safetensors' or 'pytorch_model.bin'."
    )

print("Model folder:", MODEL_PATH)

print("\nModel files:")
for file_path in sorted(MODEL_PATH.iterdir()):
    if file_path.is_file():
        file_size_mb = (
            file_path.stat().st_size
            / (1024 ** 2)
        )

        print(
            f"  {file_path.name}: "
            f"{file_size_mb:.2f} MB"
        )


# =========================================================
# Load sentence corpus
# =========================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input corpus not found: {INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("\nInput shape:", df.shape)
print("Input columns:", df.columns.tolist())

required_corpus_columns = {
    "year",
    "raw_sent",
}

missing_corpus_columns = (
    required_corpus_columns
    - set(df.columns)
)

if missing_corpus_columns:
    raise KeyError(
        "The input corpus is missing required columns: "
        f"{sorted(missing_corpus_columns)}"
    )

# Use the original, non-lemmatised sentence text.
sentence_col = "raw_sent"

print(
    "Using sentence column:",
    sentence_col
)


# =========================================================
# Prepare analysis dataframe
# =========================================================

analysis_df = df[
    ["year", sentence_col]
].copy()

analysis_df = analysis_df.rename(
    columns={
        sentence_col: "sentence"
    }
)

analysis_df["year"] = pd.to_numeric(
    analysis_df["year"],
    errors="coerce"
)

analysis_df = analysis_df.dropna(
    subset=[
        "year",
        "sentence",
    ]
)

analysis_df["year"] = (
    analysis_df["year"]
    .astype(int)
)

analysis_df = analysis_df[
    analysis_df["year"].between(
        MIN_YEAR,
        MAX_YEAR
    )
].copy()

analysis_df["sentence"] = (
    analysis_df["sentence"]
    .astype(str)
    .str.replace(
        r"\s+",
        " ",
        regex=True
    )
    .str.strip()
)

analysis_df = analysis_df[
    analysis_df["sentence"].ne("")
].copy()


# =========================================================
# Normalize target terminology
# =========================================================

def normalize_target_mention(text):
    """
    Replace query variants with ADHD so that the aspect supplied
    to the model is visibly present in every analysed sentence.
    """

    normalized_text = str(text)

    target_patterns = [
        (
            r"\battention[\s-]+deficit[\s-]+"
            r"hyperactivity[\s/-]+disorder\b"
        ),
        (
            r"\battention[\s-]+deficit[\s-]+"
            r"disorder\b"
        ),
        (
            r"(?<!\w)A\.\s*D\.\s*H\.\s*D\."
            r"(?!\w)"
        ),
        r"\bADHD\b",
    ]

    for pattern in target_patterns:
        normalized_text = re.sub(
            pattern,
            ASPECT,
            normalized_text,
            flags=re.IGNORECASE
        )

    return normalized_text


analysis_df["sentence"] = (
    analysis_df["sentence"]
    .apply(normalize_target_mention)
)

contains_target = (
    analysis_df["sentence"]
    .str.contains(
        r"\bADHD\b",
        case=False,
        regex=True,
        na=False
    )
)

print(
    "\nSentences containing ADHD "
    "after normalization:",
    int(contains_target.sum())
)

print(
    "Sentences without a visible target "
    "after normalization:",
    int((~contains_target).sum())
)

analysis_df = (
    analysis_df[contains_target]
    .reset_index(drop=True)
)

if analysis_df.empty:
    raise ValueError(
        "No valid ADHD-containing sentences remained."
    )


# =========================================================
# Optional smoke test
# =========================================================

if TEST_N is not None:

    if TEST_LONGEST_SENTENCES:
        analysis_df["_character_length"] = (
            analysis_df["sentence"]
            .str.len()
        )

        analysis_df = (
            analysis_df
            .sort_values(
                "_character_length",
                ascending=False
            )
            .head(TEST_N)
            .drop(
                columns="_character_length"
            )
            .reset_index(drop=True)
        )

        print(
            f"\nTEST MODE: analysing the "
            f"{len(analysis_df)} longest sentences."
        )

    else:
        analysis_df = (
            analysis_df
            .head(TEST_N)
            .copy()
        )

        print(
            f"\nTEST MODE: analysing the first "
            f"{len(analysis_df)} sentences."
        )

print(
    "\nSentences to analyse:",
    len(analysis_df)
)

print(
    "Year range:",
    analysis_df["year"].min(),
    "to",
    analysis_df["year"].max()
)

print("\nExample normalized sentence:")
print(analysis_df["sentence"].iloc[0])


# =========================================================
# Load tokenizer locally
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    use_fast=False,
    local_files_only=True,
)


# =========================================================
# Target-centred truncation
# =========================================================

def get_paired_token_length(text):
    """
    Return the total token length of the sentence-aspect pair,
    including the model's special tokens.
    """

    encoded = tokenizer(
        str(text),
        ASPECT,
        add_special_tokens=True,
        truncation=False,
    )

    return len(
        encoded["input_ids"]
    )


def make_target_centered_text(text):
    """
    If the sentence-aspect pair exceeds MAX_LENGTH, retain the
    largest balanced word window around an ADHD occurrence that
    fits the model.

    If multiple ADHD mentions occur, the mention closest to the
    middle of the sentence is selected.

    Returns:
        model_text
        was_truncated
        original_pair_token_count
        model_pair_token_count
    """

    text = str(text).strip()

    original_pair_token_count = (
        get_paired_token_length(text)
    )

    if original_pair_token_count <= MAX_LENGTH:
        return (
            text,
            False,
            original_pair_token_count,
            original_pair_token_count,
        )

    target_matches = list(
        re.finditer(
            r"\bADHD\b",
            text,
            flags=re.IGNORECASE
        )
    )

    if not target_matches:
        raise ValueError(
            "Target-centred truncation was required, "
            "but ADHD was not found in the sentence."
        )

    sentence_midpoint = len(text) / 2

    target_match = min(
        target_matches,
        key=lambda match: abs(
            (
                match.start()
                + match.end()
            ) / 2
            - sentence_midpoint
        )
    )

    word_matches = list(
        re.finditer(
            r"\S+",
            text
        )
    )

    target_word_index = next(
        index
        for index, word_match
        in enumerate(word_matches)
        if (
            word_match.start()
            <= target_match.start()
            < word_match.end()
        )
    )

    # Binary-search for the largest balanced word radius
    # that fits the model.
    lower_radius = 0

    upper_radius = max(
        target_word_index,
        len(word_matches)
        - target_word_index
        - 1
    )

    best_text = (
        word_matches[
            target_word_index
        ].group()
    )

    best_pair_token_count = (
        get_paired_token_length(
            best_text
        )
    )

    while lower_radius <= upper_radius:

        radius = (
            lower_radius
            + upper_radius
        ) // 2

        start_index = max(
            0,
            target_word_index - radius
        )

        end_index = min(
            len(word_matches),
            target_word_index + radius + 1
        )

        candidate_text = " ".join(
            word_match.group()
            for word_match
            in word_matches[
                start_index:end_index
            ]
        )

        candidate_pair_token_count = (
            get_paired_token_length(
                candidate_text
            )
        )

        if (
            candidate_pair_token_count
            <= MAX_LENGTH
        ):
            best_text = candidate_text

            best_pair_token_count = (
                candidate_pair_token_count
            )

            lower_radius = radius + 1

        else:
            upper_radius = radius - 1

    if not re.search(
        r"\bADHD\b",
        best_text,
        flags=re.IGNORECASE
    ):
        raise ValueError(
            "Target-centred truncation unexpectedly "
            "removed the ADHD target."
        )

    if best_pair_token_count > MAX_LENGTH:
        raise ValueError(
            "Target-centred sentence still exceeds "
            f"{MAX_LENGTH} tokens."
        )

    return (
        best_text,
        True,
        original_pair_token_count,
        best_pair_token_count,
    )


print(
    "\nPreparing target-centred model inputs..."
)

prepared_inputs = [
    make_target_centered_text(
        sentence
    )
    for sentence in tqdm(
        analysis_df["sentence"],
        desc="Target-centred preprocessing"
    )
]

prepared_df = pd.DataFrame(
    prepared_inputs,
    columns=[
        "model_sentence",
        "was_target_centered_truncated",
        "original_pair_token_count",
        "model_pair_token_count",
    ],
    index=analysis_df.index
)

analysis_df = pd.concat(
    [
        analysis_df,
        prepared_df
    ],
    axis=1
)


# =========================================================
# Truncation checks
# =========================================================

n_truncated = int(
    analysis_df[
        "was_target_centered_truncated"
    ].sum()
)

truncation_proportion = (
    analysis_df[
        "was_target_centered_truncated"
    ].mean()
)

print(
    "\nTarget-centred truncated sentences:",
    n_truncated
)

print(
    "Target-centred truncation proportion:",
    f"{truncation_proportion:.4%}"
)

print(
    "Maximum original paired-token length:",
    int(
        analysis_df[
            "original_pair_token_count"
        ].max()
    )
)

print(
    "Maximum model paired-token length:",
    int(
        analysis_df[
            "model_pair_token_count"
        ].max()
    )
)

if (
    analysis_df[
        "model_pair_token_count"
    ].max()
    > MAX_LENGTH
):
    raise ValueError(
        "At least one prepared input still exceeds "
        f"{MAX_LENGTH} tokens."
    )

if not (
    analysis_df[
        "model_sentence"
    ].str.contains(
        r"\bADHD\b",
        case=False,
        regex=True,
        na=False
    )
).all():
    raise ValueError(
        "At least one prepared model sentence "
        "does not contain ADHD."
    )

if n_truncated > 0:
    truncated_example = analysis_df.loc[
        analysis_df[
            "was_target_centered_truncated"
        ],
        [
            "original_pair_token_count",
            "model_pair_token_count",
            "model_sentence",
        ]
    ].iloc[0]

    print(
        "\nExample target-centred truncated input:"
    )

    print(
        "Original paired-token count:",
        truncated_example[
            "original_pair_token_count"
        ]
    )

    print(
        "Model paired-token count:",
        truncated_example[
            "model_pair_token_count"
        ]
    )

    print(
        truncated_example[
            "model_sentence"
        ]
    )


# =========================================================
# Load model locally
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:", device)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )
)

model.to(device)
model.eval()


# =========================================================
# Determine model label order
# =========================================================

id2label = {
    int(index): str(label).lower()
    for index, label
    in model.config.id2label.items()
}

print(
    "Model label mapping:",
    id2label
)


def find_label_index(label_name):
    matches = [
        index
        for index, label
        in id2label.items()
        if label_name.lower()
        in label.lower()
    ]

    if len(matches) == 1:
        return matches[0]

    return None


negative_index = find_label_index(
    "negative"
)

neutral_index = find_label_index(
    "neutral"
)

positive_index = find_label_index(
    "positive"
)

if None in (
    negative_index,
    neutral_index,
    positive_index,
):
    if model.config.num_labels != 3:
        raise ValueError(
            "Could not identify the ABSA label order."
        )

    print(
        "Named labels were not found in the "
        "configuration. Using the expected order "
        "[negative, neutral, positive]."
    )

    negative_index = 0
    neutral_index = 1
    positive_index = 2

print(
    "Using label indices:",
    {
        "negative": negative_index,
        "neutral": neutral_index,
        "positive": positive_index,
    }
)


# =========================================================
# Run sentence-level ABSA
# =========================================================

all_scores = []
all_negative_probabilities = []
all_neutral_probabilities = []
all_positive_probabilities = []


for start in tqdm(
    range(
        0,
        len(analysis_df),
        BATCH_SIZE
    ),
    desc="Calculating ABSA sentiment"
):
    batch_df = analysis_df.iloc[
        start:start + BATCH_SIZE
    ]

    batch_sentences = (
        batch_df[
            "model_sentence"
        ]
        .tolist()
    )

    batch_aspects = [
        ASPECT
    ] * len(batch_sentences)

    # Inputs have already been restricted to <=512 tokens.
    # No automatic end-truncation is applied here.
    inputs = tokenizer(
        batch_sentences,
        batch_aspects,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    if (
        inputs["input_ids"].shape[1]
        > MAX_LENGTH
    ):
        raise ValueError(
            "A batch exceeded the model's "
            f"{MAX_LENGTH}-token limit."
        )

    with torch.inference_mode():
        outputs = model(**inputs)

        probabilities = F.softmax(
            outputs.logits,
            dim=-1
        )

    negative_probability = (
        probabilities[
            :,
            negative_index
        ]
        .detach()
        .cpu()
        .numpy()
    )

    neutral_probability = (
        probabilities[
            :,
            neutral_index
        ]
        .detach()
        .cpu()
        .numpy()
    )

    positive_probability = (
        probabilities[
            :,
            positive_index
        ]
        .detach()
        .cpu()
        .numpy()
    )

    # Continuous sentiment index:
    # negative = 0
    # neutral  = 0.5
    # positive = 1
    batch_scores = (
        0.0 * negative_probability
        + 0.5 * neutral_probability
        + 1.0 * positive_probability
    )

    all_scores.extend(
        batch_scores.tolist()
    )

    all_negative_probabilities.extend(
        negative_probability.tolist()
    )

    all_neutral_probabilities.extend(
        neutral_probability.tolist()
    )

    all_positive_probabilities.extend(
        positive_probability.tolist()
    )


# =========================================================
# Attach sentence-level predictions
# =========================================================

analysis_df[
    "absa_sentiment_score"
] = all_scores

analysis_df[
    "negative_probability"
] = all_negative_probabilities

analysis_df[
    "neutral_probability"
] = all_neutral_probabilities

analysis_df[
    "positive_probability"
] = all_positive_probabilities

analysis_df[
    "probability_sum"
] = (
    analysis_df[
        "negative_probability"
    ]
    + analysis_df[
        "neutral_probability"
    ]
    + analysis_df[
        "positive_probability"
    ]
)


# =========================================================
# Sanity checks
# =========================================================

print(
    "\nProbability-sum range:",
    analysis_df[
        "probability_sum"
    ].min(),
    "to",
    analysis_df[
        "probability_sum"
    ].max()
)

print(
    "Sentence-level ABSA score range:",
    analysis_df[
        "absa_sentiment_score"
    ].min(),
    "to",
    analysis_df[
        "absa_sentiment_score"
    ].max()
)

if not analysis_df[
    "probability_sum"
].between(
    0.999,
    1.001
).all():
    raise ValueError(
        "Some prediction probabilities do not "
        "sum to approximately 1."
    )

if not analysis_df[
    "absa_sentiment_score"
].between(
    0,
    1
).all():
    raise ValueError(
        "Some ABSA scores fall outside 0-1."
    )


# =========================================================
# Calculate annual ABSA indices
# =========================================================

annual_scores = (
    analysis_df
    .groupby("year")
    .agg(
        n_sentences=(
            "absa_sentiment_score",
            "size"
        ),
        avg_absa_sentiment_index=(
            "absa_sentiment_score",
            "mean"
        ),
        sd_absa_sentiment_index=(
            "absa_sentiment_score",
            "std"
        ),
        n_target_centered_truncated=(
            "was_target_centered_truncated",
            "sum"
        ),
    )
    .reset_index()
    .sort_values("year")
)

annual_scores[
    "se_absa_sentiment_index"
] = (
    annual_scores[
        "sd_absa_sentiment_index"
    ]
    / np.sqrt(
        annual_scores["n_sentences"]
    )
)

annual_scores[
    "target_centered_truncation_proportion"
] = (
    annual_scores[
        "n_target_centered_truncated"
    ]
    / annual_scores["n_sentences"]
)

annual_scores.insert(
    0,
    "target",
    ASPECT
)

annual_scores = annual_scores[
    [
        "target",
        "year",
        "n_sentences",
        "avg_absa_sentiment_index",
        "se_absa_sentiment_index",
        "sd_absa_sentiment_index",
        "n_target_centered_truncated",
        "target_centered_truncation_proportion",
    ]
]

# =========================================================
# Check annual output
# =========================================================

print(
    "\nAnnual output shape:",
    annual_scores.shape
)

print("\nAnnual ABSA results:")
print(annual_scores)

if TEST_N is None:
    expected_years = set(
        range(
            MIN_YEAR,
            MAX_YEAR + 1
        )
    )

    observed_years = set(
        annual_scores["year"]
    )

    missing_years = sorted(
        expected_years
        - observed_years
    )

    print(
        "\nMissing study years:",
        missing_years
    )


# =========================================================
# Save annual output
# =========================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

annual_scores.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nSaved annual ABSA results to: "
    f"{OUTPUT_FILE}"
)