import os
import pandas as pd
from collections import Counter, defaultdict
import re

# --- Config ---
input_dir = "output/cooccurrences_by_year"
output_file = "output/ADHD_top_cooccurrences_by_year.csv"
top_n = 10  # How many top cooccurrences per year to keep

# --- Helper to extract year ---
def extract_year_from_filename(fname):
    match = re.search(r"(\d{4})_cooccurrences\.csv$", fname)
    return int(match.group(1)) if match else None

# --- Read and process each file ---
yearly_cooccurrences = defaultdict(Counter)

for fname in os.listdir(input_dir):
    if not fname.endswith("_cooccurrences.csv"):
        continue
    year = extract_year_from_filename(fname)
    if year is None:
        continue

    fpath = os.path.join(input_dir, fname)
    try:
        df = pd.read_csv(fpath, on_bad_lines='skip') # if panfas >=1.3
        if 'cooccurrence' in df.columns and 'count' in df.columns:
            for _, row in df.iterrows():
                cooccurrence = str(row['cooccurrence']).strip().lower()
                count = int(row['count'])
                yearly_cooccurrences[year][cooccurrence] += count
    except Exception as e:
        print(f"⚠️ Failed to process {fname}: {e}")

# --- Compile results ---
rows = []
for year in sorted(yearly_cooccurrences):
    total = sum(yearly_cooccurrences[year].values())
    top = yearly_cooccurrences[year].most_common(top_n)
    for rank, (word, count) in enumerate(top, 1):
        rel_count = (count / total) * 100 if total > 0 else 0
        rows.append({
            "year": year,
            "rank": rank,
            "word": word,
            "count": count,
            "total_cooccurrences_year": total,
            "rel_count": round(rel_count, 3)
        })

# --- Save output ---
result_df = pd.DataFrame(rows)
result_df.to_csv(output_file, index=False)
print(f"✅ Saved: {output_file}")
