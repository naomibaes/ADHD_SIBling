import pandas as pd
import os

# === File paths ===
target_file = '0_corpus_processing/output/descriptives/text_counts_byyear.csv' # number of articles with target term by year
total_file = '/home/ec2-user/SageMaker/data/usnewsstream_article_counts_byyear.csv' # number of total articles by year
output_file = 'output/rel_freq_text_counts_byyear_ADHD.csv'

# === Load files ===
df_target = pd.read_csv(target_file)
df_total = pd.read_csv(total_file, thousands=',')

# === Merge by 'year' ===
df_merged = pd.merge(df_total, df_target, on='year', how='left', suffixes=('_total', '_target'))

# === Fill missing target counts with 0 ===
df_merged['count_target'] = df_merged['count_target'].fillna(0)

# === Calculate proportion ===
df_merged['proportion'] = df_merged['count_target'] / df_merged['count_total']

# === Save to output ===
df_merged.to_csv(output_file, index=False)
print(f"Saved proportion file to: {output_file}")
