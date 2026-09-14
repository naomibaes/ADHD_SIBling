#!/usr/bin/env python
"""
Encode every *.txt file in INPUT_DIR with XL-LEXEME.

• One sentence per line in each file.
• Keep ≤512 tokens centred on the target word (target is inferred from filename).
• Build InputExample(texts=[sentence], positions=[start, end]) for each line.
• Encode with batch_size=BATCH_SIZE, write <file>_XLL.txt (float32, one row per sentence).
• Runs safely on a single Tesla T4 (small batch, torch.cuda.empty_cache()).

--------------------------------------------------------------
"""

import os, re, time, datetime, traceback, numpy as np, torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ─────────────────────────── CONFIG ──────────────────────────
INPUT_DIR  = "output/embeddings/sentence_samples"           # folder of .txt files
MODEL_PATH = "/home/ec2-user/SageMaker/models/xl-lexeme"
N_WORKERS  = 1                               # keep 1 on a single GPU
BATCH_SIZE = 100                               # safe for T4
MAX_TOKENS = 512
LOG_PATH   = os.path.join(INPUT_DIR, "skipped_files.log")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ──────────────────────────────────────────────────────────────


def encode_file(fname: str):
    """
    Worker: encode one .txt file → <file>_XLL.txt
    """
    try:
        if not fname.endswith(".txt"):
            return False, "not_txt"

        target  = fname.split("_")[0].lower()          # "schizophrenia_…" → "schizophrenia"
        in_path = os.path.join(INPUT_DIR, fname)

        with open(in_path, encoding="utf-8") as f:
            sents = [ln.rstrip() for ln in f if ln.strip()]

        # Local import (one per worker ➜ avoids double-loading in parent)
        from WordTransformer import WordTransformer, InputExample
        pat = re.compile(rf"\b({re.escape(target)})\b", flags=re.I)

        # ─ helper: truncate to a 512-token window around target ─
        def trunc(tokens, tgt_idx, max_len=MAX_TOKENS):
            half = max_len // 2
            start = max(0, tgt_idx - half)
            end   = min(len(tokens), start + max_len)
            start = max(0, end - max_len)
            return tokens[start:end], tgt_idx - start

        examples = []
        for sent in sents:
            m = pat.search(sent)
            if not m:                               # skip lines without the target
                continue
            tgt_idx = len(sent[:m.start()].split()) # approx token index
            tokens  = sent.split()
            tokens_trunc, new_idx = trunc(tokens, tgt_idx)
            examples.append(
                InputExample(
                    texts=" ".join(tokens_trunc),
                    positions=[new_idx, new_idx + 1]   # ← span fixes IndexError
                )
            )

        if not examples:
            return False, "no_target"

        # safety check
        if not all(hasattr(ex, "texts") for ex in examples):
            return False, f"bad_example_type ({set(type(e) for e in examples)})"

        # ─ encode ─
        t0     = time.perf_counter()
        model  = WordTransformer(MODEL_PATH)
        embs   = model.encode(
            examples,
            batch_size=BATCH_SIZE,
            show_progress_bar=False
        )
        t1     = time.perf_counter()

        out_path = os.path.join(INPUT_DIR, fname.replace(".txt", "_XLL.txt"))
        np.savetxt(out_path, np.asarray(embs, dtype=np.float32), fmt="%.6f")
        torch.cuda.empty_cache()                    # keep GPU memory low
        return True, f"{out_path}\t{t1-t0:1.1f}s"

    except Exception as e:                          # catch & stringify traceback
        return False, f"error: {e}\n{traceback.format_exc(limit=1)}"


# ─────────────────────────── MAIN ────────────────────────────
def main():
    files    = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".txt"))
    skipped  = []

    ctx_name = "spawn" if __name__ == "__main__" else "fork"
    ctx      = mp.get_context(ctx_name)
    print(f"[Info] ctx={ctx_name} │ GPU workers={N_WORKERS}")

    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as pool:
        future_to_file = {pool.submit(encode_file, f): f for f in files}

        for fut in tqdm(as_completed(future_to_file),
                        total=len(future_to_file),
                        ncols=80,
                        desc="Encoding"):
            fname = future_to_file[fut]
            ok, info = fut.result()
            if ok:
                out, sec = info.split("\t")
                print(f"[✓] {fname:40s} → {out}  ({sec})")
            else:
                print(f"[✗] {fname} skipped: {info}")
                skipped.append((fname, info))

    # log skipped
    if skipped:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(LOG_PATH, "a", encoding="utf-8") as lg:
            lg.write(f"\n=== {ts} │ {len(skipped)} files skipped ===\n")
            for f, reason in skipped:
                lg.write(f"{f}\t{reason}\n")
        print(f"[Log] {len(skipped)} skipped → {LOG_PATH}")
    else:
        print("[Log] No files were skipped ✓")


if __name__ == "__main__":
    main()
