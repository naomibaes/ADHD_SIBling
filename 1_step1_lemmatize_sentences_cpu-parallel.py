#!/usr/bin/env python

"""
Batch-lemmatize the 'raw_sent' column in a large CSV using spaCy + en_core_web_sm,
saving *_lemmatized.csv in OUTPUT_DIR. Streams in chunks, runs
multi-process nlp.pipe for maximum CPU utilisation.
"""

import os, time, pandas as pd, spacy
from tqdm.auto import tqdm

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
INPUT_FILE   = "output/ADHD_US_newsstream_normalized-ADHD_sentences_cleaned.csv"
OUTPUT_DIR   = "output"
TARGET_WORD  = "schizophrenia"           # Not used here, but kept for consistency
MODEL_NAME   = "/home/ec2-user/SageMaker/models/en_core_web_sm-3.7.1"          # Small model – quick load, CPU friendly
CHUNK_ROWS   = 5000                      # Rows per pandas chunk
BATCH_SIZE   = 500                       # Per-process spaCy batch size
N_PROCS      = max(1, os.cpu_count() - 1)  # Use all but 1 core

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD MODEL (disable heavy components)
# --------------------------------------------------
disable = ["ner", "parser", "textcat"]
print(f"[Info] Loading spaCy model: {MODEL_NAME} (n_process={N_PROCS}) …")
nlp = spacy.load(MODEL_NAME, disable=disable)
print(f"[Info] Components active: {nlp.pipe_names}")

# --------------------------------------------------
# LEMMATIZATION HELPER
# --------------------------------------------------
def lemmatize_doc(doc):
    """Return whitespace-joined lemma string, filtered for stop/punct/space."""
    return " ".join(
        t.lemma_.lower() if not t.is_stop and not t.is_punct and not t.is_space
        else t.text.lower()
        for t in doc
    )

# --------------------------------------------------
# STREAM READ → PROCESS → STREAM WRITE
# --------------------------------------------------
start = time.time()
out_path = os.path.join(OUTPUT_DIR, os.path.basename(INPUT_FILE).replace(".csv", "_lemmatized.csv"))
csv_kwargs = dict(sep=",", quoting=0)

with pd.read_csv(INPUT_FILE, chunksize=CHUNK_ROWS, **csv_kwargs) as reader, \
     open(out_path, "w", encoding="utf-8") as fout:

    header_written = False
    total_rows = 0
    chunk_idx = 0

    for chunk in tqdm(reader, desc="Chunks", unit="chunk"):
        chunk_idx += 1
        texts = chunk["raw_sent"].astype(str).tolist()

        # === Process with spaCy pipe in parallel ===
        lemmas = []
        for doc in nlp.pipe(texts, batch_size=BATCH_SIZE, n_process=N_PROCS, disable=[]):
            lemmas.append(lemmatize_doc(doc))
        chunk["lemmatized_sent"] = lemmas

        # === Write to file incrementally ===
        chunk.to_csv(fout, header=not header_written, index=False, **csv_kwargs)
        header_written = True
        total_rows += len(chunk)

        print(f"[✓] Chunk {chunk_idx} saved → {len(chunk)} rows | Total: {total_rows}")

elapsed = time.time() - start
print(f"\n[✔] Finished lemmatizing {total_rows} sentences in {elapsed/60:.1f} min → {out_path}")
