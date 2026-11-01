"""Prototype: how many (image, anchor, rule) instances with crisp GT can PanCap val yield?

Candidates are thing segments (stuff is out of scope). Things below CAND_MIN are not
marked in P1, so they are not part of the question; pass --small-as-u to instead count
them as U (ignore) candidates. --min-cands N keeps only anchors with >= N marked candidates.
"""
import argparse
import collections
import json
import os
import statistics as st
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools/pancap_viewer"))
from pancap import parse_caption, rgb2id  # noqa: E402

meta = json.load(open(f"{ROOT}/data/index/val.json"))
S = meta["samples"]
capdir = f"{ROOT}/data/{meta['captions']}"
maskdir = f"{ROOT}/data/{meta['mask_dir']}"

SUBJ_MIN, SUBJ_MAX, CAND_MIN = 0.02, 0.6, 0.002
MAX_MARKS = 20
SIZE_MARGIN = 1.5      # larger: ratio > 1.5 pos, < 1/1.5 neg, else U
TOUCH_PX = 1.5         # touch: Euclidean gap <= 1.5 px pos, > FAR_FRAC*diag neg, else U
FAR_FRAC = 0.02
POS_MARGIN = 0.02      # left/above: centroid vs anchor's left/top edge, +-2% of width/height is U
U_DROP = 0.3           # drop an instance if > 30% of its candidates are U

ap = argparse.ArgumentParser()
ap.add_argument("--small-as-u", action="store_true")
ap.add_argument("--min-cands", type=int, default=1)
args = ap.parse_args()


def AND(a, b):
    if a is False or b is False:
        return False
    return None if a is None or b is None else True


def NOT(a):
    return None if a is None else (not a)


def EXCEPT(a, b):
    return AND(a, NOT(b))


def band(x, lo, hi):
    """True above hi, False below lo, None in between."""
    return True if x > hi else (False if x < lo else None)


stats = collections.Counter()
rule_pairs = collections.defaultdict(collections.Counter)
rule_inst = collections.defaultdict(lambda: dict(kept=0, dropped=0, empty=0, sizes=[]))
n_cands, n_subj_per_img = [], []

for k in sorted(S):
    s = S[k]
    W, H = s["width"], s["height"]
    A = W * H
    capf = os.path.join(capdir, k + ".txt")
    if not os.path.exists(capf):
        continue
    tags = [t for t in parse_caption(open(capf).read()) if t["type"] == "tag" and t["text"].strip()]
    phrase_ids = collections.defaultdict(set)
    for t in tags:
        phrase_ids[t["text"].strip().lower()].update(t["ids"])
    # an anchor needs >= 1 single-id tag whose phrase refers to no other segment
    phrase_of = {}
    for t in tags:
        p = t["text"].strip().lower()
        if len(t["ids"]) == 1 and phrase_ids[p] == set(t["ids"]):
            phrase_of.setdefault(t["ids"][0], p)
    m = rgb2id(np.array(Image.open(os.path.join(maskdir, k + ".png"))))
    subs = [g for g in s["segments"] if g["isthing"] and not g["iscrowd"]
            and SUBJ_MIN <= g["area"] / A <= SUBJ_MAX and g["id"] in phrase_of]
    n_subj_per_img.append(len(subs))
    if not subs:
        continue
    stats["imgs_with_anchor"] += 1
    diag = (W * W + H * H) ** 0.5
    geo = {}
    for g in s["segments"]:
        ys, xs = np.nonzero(m == g["id"])
        if len(xs):
            geo[g["id"]] = dict(cx=xs.mean(), cy=ys.mean(), x0=xs.min(), y0=ys.min(), area=len(xs))
    for sub in subs:
        sid = sub["id"]
        if sid not in geo:
            continue
        dist = ndimage.distance_transform_edt(m != sid)
        cands = [g for g in s["segments"] if g["isthing"] and not g["iscrowd"]
                 and g["id"] != sid and g["id"] in geo
                 and (args.small_as_u or geo[g["id"]]["area"] / A >= CAND_MIN)]
        n_cands.append(len(cands))
        stats["anchors"] += 1
        if len(cands) > MAX_MARKS:
            stats["anchors_over_mark_cap"] += 1
            continue
        if len(cands) < args.min_cands:
            stats["anchors_below_min_cands"] += 1
            continue
        sg = geo[sid]
        labels = collections.defaultdict(list)
        for g in cands:
            o = geo[g["id"]]
            if o["area"] / A < CAND_MIN:
                lab = dict.fromkeys(["same_cat", "larger", "touch", "left", "above"], None)
            else:
                gap = dist[m == g["id"]].min()
                lab = dict(
                    same_cat=g["category_id"] == sub["category_id"],
                    larger=band(o["area"] / sg["area"], 1 / SIZE_MARGIN, SIZE_MARGIN),
                    touch=True if gap <= TOUCH_PX else (False if gap > FAR_FRAC * diag else None),
                    left=band(sg["x0"] - o["cx"], -POS_MARGIN * W, POS_MARGIN * W),
                    above=band(sg["y0"] - o["cy"], -POS_MARGIN * H, POS_MARGIN * H),
                )
            lab["left&~same_cat"] = AND(lab["left"], NOT(lab["same_cat"]))
            lab["touch&~same_cat"] = AND(lab["touch"], NOT(lab["same_cat"]))
            lab["larger&same_cat"] = AND(lab["larger"], lab["same_cat"])
            lab["touch EXCEPT larger"] = EXCEPT(lab["touch"], lab["larger"])
            for r, v in lab.items():
                rule_pairs[r]["pos" if v is True else "neg" if v is False else "ign"] += 1
                labels[r].append(v)
        for r, vs in labels.items():
            ri = rule_inst[r]
            if sum(v is None for v in vs) / len(vs) > U_DROP:
                ri["dropped"] += 1
                continue
            npos = sum(v is True for v in vs)
            ri["kept"] += 1
            ri["empty"] += npos == 0
            ri["sizes"].append(npos)

print("images", len(S), dict(stats))
print("anchors/img: mean %.2f" % st.mean(n_subj_per_img),
      "hist", sorted(collections.Counter(min(x, 6) for x in n_subj_per_img).items()))
print("thing candidates per anchor (all sizes): mean %.1f median %s p90 %s" % (
    st.mean(n_cands), st.median(n_cands), sorted(n_cands)[int(.9 * len(n_cands))]))
print("\npair-level labels" + (" (sub-floor candidates counted as U)" if args.small_as_u else " (marked candidates only)") + ":")
for r, c in rule_pairs.items():
    tot = sum(c.values())
    print(f"  {r:22s} pos {c['pos']/tot:6.1%}  neg {c['neg']/tot:6.1%}  ignore {c['ign']/tot:6.1%}")
print(f"\ninstance-level (drop if >{U_DROP:.0%} of candidates are U):")
for r, ri in rule_inst.items():
    n = ri["kept"] + ri["dropped"]
    sz = ri["sizes"]
    print(f"  {r:22s} kept {ri['kept']:5d}/{n} ({ri['kept']/n:5.1%})  empty {ri['empty']/max(ri['kept'],1):6.1%}"
          f"  |S*| mean {st.mean(sz):.2f}  |S*|>=2 {sum(x >= 2 for x in sz)/max(len(sz),1):6.1%}")
