"""Copy the per-image GT records of the Test and diagnostic images into a small stand-alone set.

  python3 tools/eval/slim_records.py OUT_DIR   # -> OUT_DIR/data/anchor_gt/test_records/{train,val}/{details.jsonl,index.json}

The analysis (score.py, analyze.py, consistency.py) reads per-image records through render.record(). The full
generate.py build is 370 MB over every COCO image; this subset (2,596 images, ~28 MB) is enough to score the
released runs. render._gt_dir() falls back to it when data/anchor_gt/{split}/index.json is absent.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GT = os.path.join(ROOT, "data", "anchor_gt")


def main(out):
    keys = {}
    for f in ("test_draft.jsonl", "test_diag.jsonl"):
        for line in open(os.path.join(GT, f)):
            it = json.loads(line)
            keys.setdefault(it["split"], set()).add(it["key"])
    for split, ks in sorted(keys.items()):
        rows = {r["key"]: r for r in json.load(open(os.path.join(GT, split, "index.json")))}
        od = os.path.join(out, "data", "anchor_gt", "test_records", split)
        os.makedirs(od, exist_ok=True)
        idx, off = [], 0
        with open(os.path.join(GT, split, "details.jsonl"), "rb") as src, open(os.path.join(od, "details.jsonl"), "wb") as dst:
            for k in sorted(ks):
                r = rows[k]
                src.seek(r["off"])
                b = src.read(r["len"])
                dst.write(b + b"\n")
                idx.append(dict(r, off=off, len=r["len"]))
                off += r["len"] + 1
        json.dump(idx, open(os.path.join(od, "index.json"), "w"))
        print(f"{split}: {len(idx)} images -> {od}")


if __name__ == "__main__":
    main(sys.argv[1])
