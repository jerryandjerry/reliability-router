"""Phase 4 prep: split judge_input.jsonl into ready-to-judge batch files.

The pipeline does all data handling here so a judge agent never slices, filters or paginates
anything — it opens one small file, judges the rows in it, and writes one file.
"""
import json
import os
import shutil

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(DIR, "judge_input.jsonl")
BATCH_DIR = os.path.join(DIR, "judge_batches")
OUT_DIR = os.path.join(DIR, "judge_out")
BATCH = 50

rows = [json.loads(line) for line in open(SRC, encoding="utf-8")]

shutil.rmtree(BATCH_DIR, ignore_errors=True)
os.makedirs(BATCH_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

n_batches = (len(rows) + BATCH - 1) // BATCH
for b in range(n_batches):
    chunk = rows[b * BATCH:(b + 1) * BATCH]
    # drop the global index; the agent only needs what it judges
    payload = [{k: v for k, v in r.items() if k != "idx"} for r in chunk]
    with open(os.path.join(BATCH_DIR, f"batch_{b:03d}.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)

sizes = [os.path.getsize(os.path.join(BATCH_DIR, f"batch_{b:03d}.json")) for b in range(n_batches)]
print(f"rows {len(rows)} -> {n_batches} batch files of {BATCH} in {BATCH_DIR}")
print(f"batch file size: min {min(sizes)/1024:.0f} KB  max {max(sizes)/1024:.0f} KB  "
      f"mean {sum(sizes)/len(sizes)/1024:.0f} KB")
print(f"~{sum(sizes)/len(sizes)/4:.0f} tokens per batch")
