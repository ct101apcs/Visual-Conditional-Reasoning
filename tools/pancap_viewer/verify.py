"""Batch-verify every sample of a split and write data/index/report_<split>.json (used by the viewer filters).

  python tools/pancap_viewer/verify.py --split val
  python tools/pancap_viewer/verify.py --split train --workers 16
"""
import argparse
import collections
import json
import os
from multiprocessing import Pool

from tqdm import tqdm

import pancap

_split = None


def _init(name):
    global _split
    _split = pancap.Split(name)


def _check(key):
    r = pancap.verify(_split, key)
    return key, dict(status=r["status"], codes=sorted({i["code"] for i in r["issues"]}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "o365val"])
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    args = ap.parse_args()

    keys = pancap.Split(args.split).keys
    report = {}
    with Pool(args.workers, initializer=_init, initargs=(args.split,)) as pool:
        for key, r in tqdm(pool.imap_unordered(_check, keys, chunksize=64), total=len(keys)):
            report[key] = r

    out = os.path.join(pancap.DATA_ROOT, "index", f"report_{args.split}.json")
    json.dump(report, open(out, "w"))

    status = collections.Counter(r["status"] for r in report.values())
    codes = collections.Counter(c for r in report.values() for c in r["codes"])
    print(f"\n{args.split}: {len(report)} samples  " + "  ".join(f"{k}={v}" for k, v in status.most_common()))
    for code, n in codes.most_common():
        print(f"  {code:24s} {n:7d}  ({100 * n / len(report):.1f}%)")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
