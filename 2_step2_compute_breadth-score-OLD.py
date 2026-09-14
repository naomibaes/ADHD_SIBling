#!/usr/bin/env python3
"""
Summarise XL-LEXEME sentence-vector files
─────────────────────────────────────────
• Looks for files matching  *_XLL.txt   in OUTPUT_DIR
  (each file = one row-per-sentence, 1024 floats per row)
• Extracts   <term>   and   <year>   from the name
• Calculates cosine-dissimilarity between all pairs of rows
• Writes one CSV row per file:
      year, cosine_dissim_mean, cosine_dissim_sd,
      cosine_dissim_se, term
"""

from __future__ import annotations
import os, re, math, argparse, itertools, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────
# Configuration – edit as needed
# ──────────────────────────────────────────────────────────
OUTPUT_DIR   = Path("output/embeddings/sentence_samples")   # folder with *_XLL.txt files
CSV_OUT      = Path("output/breadth_scores.csv")
# regex:  ..._<YEAR>_XLL.txt   (+ capture arbitrary leading term string)
FILE_RE      = re.compile(r"(?P<term>[A-Za-z_]+?)_.*?(?P<year>\d{4})_XLL\.txt$", re.I)

# ──────────────────────────────────────────────────────────
def triangular_cosine_dissim(mat: np.ndarray) -> np.ndarray:
    """Upper-triangular (1-cosine similarity)/2) (vectorised)."""
    sims = cosine_similarity(mat) # +1 (cosine similarity)
    dis  = 1.0 - sims             # 0 -> 2 (cosine distance)
    dis  = (1.0 - sims) / 2.0     # 0 -> 1 (cosine distance, rescaled to 0-1)
    # Pick the upper-triangular part (k=1 ⇒ exclude diagonal)
    iu   = np.triu_indices_from(dis, k=1)
    return dis[iu]

# ──────────────────────────────────────────────────────────
def summarise_file(path: Path) -> dict[str, float] | None:
    """Return summary statistics for one vector file, or None on failure."""
    match = FILE_RE.search(path.name)
    if not match:
        print(f"⚠️  Skipping (name pattern mismatch): {path.name}")
        return None

    year = match.group("year")
    term = match.group("term").lower()

    try:
        vecs = np.loadtxt(path, dtype=np.float32)
        if vecs.ndim == 1:      # single sentence → no pair to compare
            return None
        dis  = triangular_cosine_dissim(vecs)
        mean = float(np.mean(dis))
        sd   = float(np.std(dis, ddof=0))
        se   = sd / math.sqrt(dis.size) if dis.size else float("nan")

        return dict(
            year                 = year,
            cosine_dissim_mean   = mean,
            cosine_dissim_sd     = sd,
            cosine_dissim_se     = se,
            term                 = term
        )
    except Exception as exc:      # any IO / parse / numeric error
        print(f"❗ Error reading {path.name}: {exc}")
        return None

# ──────────────────────────────────────────────────────────
def main(out_dir: Path, csv_out: Path) -> None:
    rows: list[dict] = []

    for file in sorted(out_dir.glob("*_XLL.txt")):
        summary = summarise_file(file)
        if summary:
            rows.append(summary)

    if not rows:
        print("No summaries produced – nothing to write.")
        return

    df = pd.DataFrame(rows)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False, float_format="%.6f")
    print(f"✅  Wrote {len(df)} rows → {csv_out}")

# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarise XL-LEXEME sentence-vector files")
    parser.add_argument("--dir",  default=OUTPUT_DIR, type=Path,
                        help=f"Directory to scan (default: {OUTPUT_DIR})")
    parser.add_argument("--csv",  default=CSV_OUT,    type=Path,
                        help=f"Output CSV path (default: {CSV_OUT})")
    args = parser.parse_args(namespace=argparse.Namespace(dir=OUTPUT_DIR, csv=CSV_OUT))
    main(args.dir, args.csv)
