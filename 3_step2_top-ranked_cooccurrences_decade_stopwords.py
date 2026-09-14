import os
import pandas as pd
from collections import Counter, defaultdict
import re

# --- Config ---
input_dir      = "output/cooccurrences_by_year"
output_file    = "output/ADHD_top_cooccurrences_by_decade_nostopwords.csv"
top_n          = 1000
stopword_file  = "../resources/english_nltk_stopwords.txt"  # one stopword per line

# --- Load stopwords ---
with open(stopword_file, 'r', encoding='utf-8') as f:
    stop_words = {line.strip().lower() for line in f if line.strip()}

# --- Helpers ---
def extract_year_from_filename(fname):
    m = re.search(r"(\d{4})_cooccurrences\.csv$", fname)
    return int(m.group(1)) if m else None

def get_decade(year):
    return (year // 10) * 10 #original grouping by decade
    #return (year // 5) * 5 #grouping by 5-year period

# --- Aggregate counts by decade ---
decade_cooccurrences = defaultdict(Counter)

for fname in os.listdir(input_dir):
    if not fname.endswith("_cooccurrences.csv"):
        continue

    year = extract_year_from_filename(fname)
    if year is None:
        continue

    decade = get_decade(year)
    path = os.path.join(input_dir, fname)

    with open(path, 'r', encoding='utf-8') as file:
        next(file)  # skip header
        for i, line in enumerate(file, start=2):
            parts = line.strip().split(',')
            if len(parts) != 2:
                print(f"▲ Skipped bad line {i} in {fname}")
                continue
            cooccurrence = parts[0].lower()
            count     = int(parts[1])
            decade_cooccurrences[decade][cooccurrence] += count

# --- Build final top‐N per decade, excluding stopwords & dashes ---
rows = []
for decade in sorted(decade_cooccurrences):
    total = sum(decade_cooccurrences[decade].values())

    filtered = {
        coll: cnt
        for coll, cnt in decade_cooccurrences[decade].items()
        if coll not in stop_words
           and '-' not in coll
    }

    top = Counter(filtered).most_common(top_n)
    for coll, cnt in top:
        rows.append({
            'decade': decade,
            'cooccurrence': coll,
            'count': cnt,
            'total_cooccurrencees_decade': total,
            'normalized_count': round(cnt/total if total else 0, 6)
        })

df_out = pd.DataFrame(rows)
df_out.to_csv(output_file, index=False)
print(f"✔ Saved: {output_file}")
