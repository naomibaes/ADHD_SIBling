#!/usr/bin/env python
"""
Optimized sentence extraction + length stats script with hybrid full‐doc / windowed approach

Processes full-text TSV files:
 - Extracts target-matching sentences (with context) using SpaCy’s sentencizer
 - For very large articles, falls back to a character-window around each hit
 - Computes token and word counts
 - Drops fragments, URLs, and one-word “sentences”
 - Outputs:
     - {file}_sentences.csv
     - {file}_length_stats.csv
"""

import os
import time
import re
import pandas as pd
import spacy
from tqdm import tqdm

# === Config ===
INPUT_FILE   = "/home/ec2-user/SageMaker/ADHD_conceptual_change/input/ADHD_US_newsstream_normalized-ADHD.tsv"
OUTPUT_DIR   = "/home/ec2-user/SageMaker/ADHD_conceptual_change/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME   = "/home/ec2-user/SageMaker/models/en_core_web_sm-3.7.1"
MAX_FULL_DOC = 50_000       # chars: threshold for full-doc parsing
WINDOW       = 500          # chars before/after each term match in fallback
BATCH_SIZE   = 64
N_PROCESS    = max(os.cpu_count() - 1, 1)

# === Load SpaCy model ===
print("⏳ Loading SpaCy model …")
nlp = spacy.load(MODEL_NAME, disable=["ner","lemmatizer","textcat"])
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")
print("✔ Model loaded.")

# === Helper to extract term from filename ===
def extract_target_term(path: str) -> str:
    base = os.path.basename(path)
    return base.split("_")[0].lower()   # e.g. "schizophrenia_US_News.tsv" → "schizophrenia"

# === Main extraction function ===
def extract_sentences_with_target(file_path: str):
    target = extract_target_term(file_path)
    print(f"\n🎯 Target term = '{target}'")
    start_all = time.time()

    # regex to find exact word matches
    target_re = re.compile(rf"\b{re.escape(target)}\b", flags=re.IGNORECASE)

    # 1) Load corpus
    print("📖 Reading corpus …")
    df = pd.read_csv(file_path, sep="\t")
    df.fillna("", inplace=True)

    # 2) Prepare metadata + texts
    meta_cols = ["id","title","publisher","source_type","pub_province","year"]
    records  = df[meta_cols + ["text"]].to_dict(orient="records")

    extracted = []
    total_docs = len(records)
    print(f"🚀 Processing {total_docs} documents …")

    # 3) Loop through records
    for rec in tqdm(records, total=total_docs):
        raw = rec["text"].strip()
        if not raw:
            continue

        # decide full-doc vs windowed
        if len(raw) <= MAX_FULL_DOC:
            doc = nlp(raw)
            sents = list(doc.sents)
        else:
            # find all target hits in the big raw text
            sents = []
            for m in target_re.finditer(raw):
                start = max(m.start() - WINDOW, 0)
                end   = min(m.end()   + WINDOW, len(raw))
                snippet = raw[start:end]
                subdoc = nlp(snippet)
                sents.extend(subdoc.sents)

        # filter & record
        for sent in sents:
            s = sent.text.strip()
            if not target_re.search(s):
                continue
            words = s.split()
            # drop fragments, URLs, one-word
            if len(words) <= 1 or s.lower().startswith(("http","www")):
                continue

            extracted.append({
                **{k: rec[k] for k in meta_cols},
                "raw_sent": s,
                "full_article_text": raw,
                "sentence_len": len(sent),
                "sentence_word_len": len(words),
                "article_len": len(doc) if 'doc' in locals() else len(subdoc),
                "article_word_len": len(raw.split())
            })

    # 4) Finalize & timing
    df_out = pd.DataFrame(extracted)
    elapsed = time.time() - start_all
    print(f"⏱ Extraction complete: {len(extracted)} sentences from {total_docs} docs in {elapsed:.2f}s")

    # 5) Save sentences CSV
    sent_file = os.path.join(
        OUTPUT_DIR,
        os.path.basename(file_path).replace(".tsv", "_sentences.csv")
    )
    df_out.to_csv(sent_file, index=False)
    print(f"✔ Saved extracted sentences → {sent_file}")

    # 6) Compute summary stats
    stats = {
        "sentence_len_min":    df_out["sentence_len"].min(),
        "sentence_len_med":    df_out["sentence_len"].median(),
        "sentence_len_max":    df_out["sentence_len"].max(),
        "sentence_len_mean":   df_out["sentence_len"].mean(),
        "sentence_len_sd":     df_out["sentence_len"].std(),
        "sentence_word_min":   df_out["sentence_word_len"].min(),
        "sentence_word_med":   df_out["sentence_word_len"].median(),
        "sentence_word_max":   df_out["sentence_word_len"].max(),
        "sentence_word_mean":  df_out["sentence_word_len"].mean(),
        "article_len_min":     df_out["article_len"].min(),
        "article_len_med":     df_out["article_len"].median(),
        "article_len_max":     df_out["article_len"].max(),
        "article_len_mean":    df_out["article_len"].mean(),
        "article_word_len_min":df_out["article_word_len"].min(),
        "article_word_len_med":df_out["article_word_len"].median(),
        "article_word_len_max":df_out["article_word_len"].max(),
        "article_word_len_mean":df_out["article_word_len"].mean(),
        "n_sentences":         len(df_out),
        "n_articles":          df_out["id"].nunique()
    }
    stats_df = pd.DataFrame([stats])

    stats_file = os.path.join(
        OUTPUT_DIR,
        os.path.basename(file_path).replace(".tsv", "_length_stats.csv")
    )
    stats_df.to_csv(stats_file, index=False)
    print(f"📊 Saved length stats → {stats_file}")

    # 7) Total runtime
    total_time = time.time() - start_all
    print(f"\n🏁 Total script runtime: {total_time:.2f} seconds")


# === Run ===
if __name__ == "__main__":
    extract_sentences_with_target(INPUT_FILE)
