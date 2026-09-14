# Authors: Naomi Baes and ChatGPT

import os
import glob
import pandas as pd
from collections import Counter
import re

# --- Extracts cooccurrences around the target word ---
def extract_contexts(sentence, target, window_size=5):
    # ↓— normalize for case-insensitive matching
    sentence = sentence.lower()
    target   = target.lower()

    words = sentence.split()
    target_indices = [i for i, w in enumerate(words) if w == target]
    contexts = []
    for idx in target_indices:
        start = max(idx - window_size, 0)
        end   = min(idx + window_size + 1, len(words))
        context_words = [words[i] for i in range(start, end) if i != idx]
        contexts.extend(context_words)
    return contexts

# --- Filter: exclude numbers, symbols, and any tokens containing '-' ---
def is_valid_word(token):
    # only letters, no digits or hyphens
    return bool(re.fullmatch(r"[A-Za-z]+", token))

# --- Process input files to extract cooccurrences by year ---
def process_files(input_dir, output_base_dir):
    os.makedirs(output_base_dir, exist_ok=True)
    log_file_path = os.path.join(output_base_dir, 'warnings.log')

    for file_path in glob.glob(f"{input_dir}/*_lemmatized.csv"):
        file_base = os.path.basename(file_path).replace('.csv','')
        target    = file_base.split('_')[0]  # e.g., 'schizophrenia' from filename

        try:
            df = pd.read_csv(file_path, delimiter=',')
            if 'lemmatized_sent' not in df.columns or 'year' not in df.columns:
                raise ValueError("Expected columns 'lemmatized_sent' and 'year' not found")

            for year, group in df.groupby('year'):
                context_counter = Counter()
                for _, row in group.iterrows():
                    sentence = str(row["lemmatized_sent"])
                    contexts = extract_contexts(sentence, target)
                    # filter out invalid tokens (no hyphens allowed)
                    filtered = [w.lower() for w in contexts if is_valid_word(w)]
                    context_counter.update(filtered)

                output_file = os.path.join(
                    output_base_dir,
                    f"{file_base}_{year}_cooccurrences.csv"
                )
                with open(output_file, "w", encoding='utf-8') as f:
                    f.write("cooccurrence,count\n")
                    for word, count in context_counter.items():
                        f.write(f"{word},{count}\n")
                print(f"✔ Written: {output_file}")

        except Exception as e:
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"▲ Error processing {file_path}: {e}\n")

# === Run Script ===
input_dir  = "output"             # ← LEAMMATIZED FILES LIVE HERE
output_dir = os.path.join(input_dir, "cooccurrences_by_year")
process_files(input_dir, output_dir)
