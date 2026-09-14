import os
import pandas as pd
from collections import Counter, defaultdict
import re

# --- Config ---
input_dir = "output/cooccurrences_by_year"
output_file = "output/ADHD_top_cooccurrences_by_decade.csv"
top_n = 10  # Change if needed

# --- Helper to extract year ---
def extract_year_from_filename(fname):
    match = re.search(r"(\d{4})_cooccurrences\.csv$", fname)
    return int(match.group(1)) if match else None

# --- Helper to get decade ---
def get_decade(year):
    return (year // 10) * 10

# --- Process files ---
decade_cooccurrences = defaultdict(Counter)

for fname in os.listdir(input_dir):
    if not fname.endswith("_cooccurrences.csv"):
        continue

    year = extract_year_from_filename(fname)
    if year is None:
        continue

    decade = get_decade(year)
    fpath = os.path.join(input_dir, fname)

    try:
        with open(fpath, 'r', encoding='utf-8') as file:
            header = next(file)  # skip header
            for i, line in enumerate(file, start=2):  # line numbers start from 2
                try:
                    parts = line.strip().split(',')
                    if len(parts) != 2:
                        raise ValueError("Expected 2 fields")
                    cooccurrence = parts[0].strip().lower()
                    count = int(parts[1])
                    decade_cooccurrences[decade][cooccurrence] += count
                except Exception as line_error:
                    print(f"⚠️ Skipped bad line {i} in {fname}: {line.strip()} ({line_error})")

    except Exception as e:
        print(f"❌ Failed to process {fname}: {e}")

# --- Write normalized results ---
rows = []
for decade in sorted(decade_cooccurrences):
    total = sum(decade_cooccurrences[decade].values())
    top = decade_cooccurrences[decade].most_common(top_n)
    for cooccurrence, count in top:
        norm_count = count / total if total > 0 else 0
        rows.append({
            'decade': decade,
            'cooccurrence': cooccurrence,
            'count': count,
            'total_cooccurrences_decade': total,
            'normalized_count': round(norm_count, 6)
        })

df_out = pd.DataFrame(rows)
df_out.to_csv(output_file, index=False)
print(f"✅ Saved: {output_file}")

