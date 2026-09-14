import re
import pandas as pd

# --- 1) Paths to your files ---
input_tsv  = '/home/ec2-user/SageMaker/ADHD_conceptual_change/input/ADHD_US_newsstream.tsv'
output_tsv = '/home/ec2-user/SageMaker/ADHD_conceptual_change/input/ADHD_US_newsstream_normalized-ADHD.tsv'

# --- 2) Define regex patterns for ADHD variants ---
# Long-form variants (allow spaces, slashes, hyphens)
LONG = r'Attention\s*[-/ ]?\s*Deficit\s*[-/ ]?\s*Hyperactivity\s*[-/ ]?\s*Disorder'
# Acronym variants (ADHD, A.D.H.D., A D H D)
ACRO = r'A\.?\s*D\.?\s*H\.?\s*D\.?'

# Combined pattern to catch:
#   (a) LONG alone
#   (b) LONG (ADHD)
#   (c) ACRO alone
#   (d) ADHD (LONG)
pattern = re.compile(
    rf"""
    \b(?:{LONG})\b(?:\s*\(\s*(?:{ACRO})\s*\))?   # long form ± (ADHD)
    |                                           # OR
    \b(?:{ACRO})\b(?:\s*\(\s*(?:{LONG})\s*\))?   # ADHD ± (long form)
    """,
    flags=re.IGNORECASE | re.VERBOSE
)

# For reporting: parentheticals of either form
paren_pattern = re.compile(rf'\(\s*(?:{ACRO}|{LONG})\s*\)', flags=re.IGNORECASE)

# --- 3) Load data ---
df = pd.read_csv(input_tsv, sep='\t', dtype=str)
df['text'] = df['text'].fillna('')

# --- 4) Summary counts BEFORE replacement ---
total_matches = df['text'].str.count(pattern).sum()
total_parentheticals = df['text'].str.count(paren_pattern).sum()

# --- 5) Keep a copy of the original column ---
original_texts = df['text'].copy()

# --- 6) Apply normalization ---
df['text'] = df['text'].str.replace(pattern, 'ADHD', regex=True)

# --- 7) Validation checks ---
changes = original_texts != df['text']
changed_rows = df.loc[changes]

if not changed_rows.empty:
    print("\n✅ Found changes. Showing up to 5 examples:\n")
    for i, idx in enumerate(changed_rows.index[:5]):
        print(f"Example {i+1}:")
        print(" Before:", original_texts.loc[idx])
        print(" After :", df['text'].loc[idx])
        print("-" * 60)
else:
    print("\n⚠️ No changes detected — check your regex patterns.")

# Count how many parentheticals still remain after cleanup
remaining_parentheticals = df['text'].str.count(paren_pattern).sum()

# --- 8) Save cleaned file ---
df.to_csv(output_tsv, sep='\t', index=False)

# --- 9) Print summary ---
print(f"\nSummary:")
print(f"✓ Loaded {len(df):,} rows from {input_tsv}")
print(f"✓ Total ADHD-variant occurrences replaced: {total_matches}")
print(f"✓ Total parentheticals removed: {total_parentheticals}")
print(f"✓ Remaining parentheticals after cleanup: {remaining_parentheticals}")
print(f"✓ Cleaned text written to {output_tsv}")
