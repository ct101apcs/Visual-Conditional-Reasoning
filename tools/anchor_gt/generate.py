"""Synthesise ANCHOR ground truth from COCONut-PanCap: eligible subjects, marked candidates,
predicate values and the GT set of every rule template for every subject.

  python tools/anchor_gt/generate.py --split val
  python tools/anchor_gt/generate.py --split train --workers 14
  python tools/anchor_gt/generate.py --sample          # draft Test sample + summary.json

Outputs go to data/anchor_gt/:
  {split}/details.jsonl   one record per image (segments with their funnel stage, anchors, GT)
  {split}/index.json      one compact row per image (+ byte offset into details.jsonl)
  {split}/stats.json      funnel and rule statistics for the split
  test_draft.jsonl        stratified draft sample (val + held-out train slice)
  test_diag.jsonl         caption-contact diagnostic slice (C01 + its T01 pair)
  summary.json            everything the viewer needs that is not per image
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools/pancap_viewer"))
sys.path.insert(0, HERE)
from pancap import parse_caption, rgb2id  # noqa: E402
import lexicon  # noqa: E402
import rules as R  # noqa: E402

DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "anchor_gt")
CATS = {c["id"]: c for c in json.load(open(os.path.join(DATA, "index/categories.json")))}

# Anchor funnel stages, in order. A segment's "stage" is the last one it passed.
STAGES = [
    ("thing", "thing segment, not iscrowd, non-empty mask"),
    ("size", f"covers {R.SUBJ_MIN:.0%}–{R.SUBJ_MAX:.0%} of the image"),
    ("phrase_unique", "cited by a single-id caption tag whose phrase refers to no other segment"),
    ("not_group", "phrase does not describe a group/plural or a reflection"),
    ("not_mismatch", "phrase does not name only a different COCO category (mis-tag)"),
    ("clutter", f"{R.MIN_CANDS}–{R.MAX_CANDS} marked candidates (things ≥{R.CAND_MIN:.1%} of the image)"),
    ("rule_kept", f"≥1 rule template keeps ≤{R.U_DROP:.0%} of candidates in the ignore band"),
]
STAGE_IDX = {s: i for i, (s, _) in enumerate(STAGES)}

SUPER_GROUP = {"person": "human", "animal": "animal", "vehicle": "vehicle", "food": "food"}


def subject_type_of(cat_id):
    return SUPER_GROUP.get(CATS[cat_id]["supercategory"], "object")


def in_test_slice(key):
    """Held-out image slice of train used for Test items: 10% by hash."""
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 10 == 0


def geometry(m, seg_ids, W, H, diag):
    """Per-segment pixel geometry, plus a cropped distance map for touch gaps."""
    objs = ndimage.find_objects(m)
    R_far = int(np.ceil(R.FAR_FRAC * diag)) + 2
    geo = {}
    for sid in seg_ids:
        if sid >= len(objs) + 1 or objs[sid - 1] is None:
            continue
        ys_sl, xs_sl = objs[sid - 1]
        y0, y1, x0, x1 = ys_sl.start, ys_sl.stop, xs_sl.start, xs_sl.stop
        crop = m[y0:y1, x0:x1] == sid
        ys, xs = np.nonzero(crop)
        area = len(xs)
        if not area:
            continue
        # interior point for the number mark: maximum of the inner distance transform
        inner = ndimage.distance_transform_edt(np.pad(crop, 1))[1:-1, 1:-1]
        iy, ix = np.unravel_index(np.argmax(inner), inner.shape)
        # outer distance map within the bbox grown by the "far" radius
        Y0, Y1, X0, X1 = max(0, y0 - R_far), min(H, y1 + R_far), max(0, x0 - R_far), min(W, x1 + R_far)
        geo[sid] = dict(area=area, cx=xs.mean() + x0, cy=ys.mean() + y0, x0=x0, y0=y0, x1=x1 - 1, y1=y1 - 1,
                        mark=[int(ix + x0), int(iy + y0)], edge=int(min(x0, y0, W - x1, H - y1)),
                        win=(Y0, Y1, X0, X1),
                        dmap=ndimage.distance_transform_edt(m[Y0:Y1, X0:X1] != sid))
    return geo


def gap(geo, m, a, b):
    """Smallest Euclidean distance from b's pixels to a's pixels (inf beyond a's window)."""
    Y0, Y1, X0, X1 = geo[a]["win"]
    sel = m[Y0:Y1, X0:X1] == b
    return float(geo[a]["dmap"][sel].min()) if sel.any() else float("inf")


def process(args):
    split, key, meta, cap_path, mask_path, vstatus = args
    W, H = meta["width"], meta["height"]
    A, diag = W * H, (W * W + H * H) ** 0.5
    rec = dict(key=key, split=split, W=W, H=H, verify=vstatus)
    caption = open(cap_path, encoding="utf-8", errors="replace").read() if os.path.exists(cap_path) else None
    if caption is None or not os.path.exists(mask_path):
        rec.update(img_stage="missing", segs=[], anchors=[])
        return rec

    toks = parse_caption(caption)
    tags = [t for t in toks if t["type"] == "tag" and t["text"].strip()]
    rel = lexicon.contact_pairs(toks)
    phrase_ids = collections.defaultdict(set)
    for t in tags:
        phrase_ids[t["text"].strip().lower()].update(t["ids"])
    unique_phr = collections.defaultdict(list)       # seg id -> its unique single-id phrases
    for t in tags:
        p = t["text"].strip()
        if len(t["ids"]) == 1 and phrase_ids[p.lower()] == set(t["ids"]) and p not in unique_phr[t["ids"][0]]:
            unique_phr[t["ids"][0]].append(p)

    m = rgb2id(np.array(Image.open(mask_path).convert("RGB")))
    segs = meta["segments"]
    things = [g for g in segs if g["isthing"]]
    geo = geometry(m, [g["id"] for g in things], W, H, diag)
    stuff = {CATS[g["category_id"]]["name"] for g in segs if not g["isthing"] and g["category_id"] in CATS}
    rec["scene"] = lexicon.scene_tags(caption)
    rec["scene"]["io_stuff"] = lexicon.scene_from_stuff(stuff)
    rec["people_mask"] = any(g["category_id"] == 1 for g in things)

    marked = [g for g in things if not g["iscrowd"] and g["id"] in geo and geo[g["id"]]["area"] / A >= R.CAND_MIN]
    seg_rows, anchors = [], []
    for g in things:
        sid, cat = g["id"], CATS[g["category_id"]]["name"]
        row = dict(id=sid, cat=cat, crowd=bool(g["iscrowd"]), area=geo[sid]["area"] if sid in geo else 0)
        seg_rows.append(row)
        stage, why = None, None
        # walk the funnel; `stage` is the last stage passed
        if g["iscrowd"] or sid not in geo:
            why = "crowd" if g["iscrowd"] else "empty mask"
        else:
            stage = "thing"
            frac = geo[sid]["area"] / A
            phr = unique_phr.get(sid, [])
            statuses = {p: lexicon.phrase_category_status(p, cat) for p in phr}
            phr.sort(key=lambda p: statuses[p] != "match")       # prefer a phrase naming the category
            flags = {p: sorted(lexicon.phrase_flags(p, cat)) for p in phr}
            good = [p for p in phr if not {"group", "reflection"} & set(flags[p])]
            if not R.SUBJ_MIN <= frac <= R.SUBJ_MAX:
                why = f"area {frac:.1%}"
            elif not phr:
                stage, why = "size", ("not cited by the caption" if not any(sid in t["ids"] for t in tags)
                                      else "only multi-id tags or a phrase shared with other segments")
            elif not good:
                stage, why = "phrase_unique", "group/reflection phrase: " + phr[0]
            elif all(statuses[p] == "mismatch" for p in good):
                stage, why = "not_group", f"mis-tag: '{good[0]}' on {cat}"
            else:
                stage = "not_mismatch"
                phrase = next(p for p in good if statuses[p] != "mismatch")
                cands = [c for c in marked if c["id"] != sid]
                if not R.MIN_CANDS <= len(cands) <= R.MAX_CANDS:
                    why = f"{len(cands)} candidates"
                else:
                    stage = "clutter"
                    anchors.append(build_anchor(g, cat, phrase, statuses[phrase], flags[phrase], frac,
                                                cands, geo, m, W, H, diag, rel))
                    if any(r["keep"] for r in anchors[-1]["rules"].values()):
                        stage = "rule_kept"
                    else:
                        why = "all rules exceed the ignore-band limit"
        row.update(stage=stage, why=why)
    rec["segs"] = seg_rows
    rec["anchors"] = anchors
    rec["marks"] = {str(g["id"]): geo[g["id"]]["mark"] for g in marked}
    best = max([STAGE_IDX[r["stage"]] for r in seg_rows if r["stage"]] or [-1])
    rec["img_stage"] = STAGES[best][0] if best >= 0 else "no_thing"
    return rec


def build_anchor(g, cat, phrase, status, flags, frac, cands, geo, m, W, H, diag, rel):
    sid = g["id"]
    s = dict(geo[sid], cat=cat)
    preds, ch = {}, {}
    sphys = lexicon.PHYS_SIZE.get(cat)
    for c in cands:
        cid, ccat = c["id"], CATS[c["category_id"]]["name"]
        o = dict(geo[cid], cat=ccat, gap=gap(geo, m, sid, cid))
        q = R.predicates(o, s, W, H, diag)
        q["contact"] = R.contact_value(o["gap"], frozenset((sid, cid)) in rel, diag)
        preds[cid] = q
        # counter-heuristic: the measured answer disagrees with an easy shortcut
        f = []
        ophys = lexicon.PHYS_SIZE.get(ccat)
        if sphys and ophys and ((q["larger"] is True and ophys < sphys) or (q["smaller"] is True and ophys > sphys)):
            f.append("larger")      # pixel size vs typical physical size
        bbox_overlap = o["x0"] <= s["x1"] and s["x0"] <= o["x1"] and o["y0"] <= s["y1"] and s["y0"] <= o["y1"]
        if q["touch"] is False and bbox_overlap:
            f.append("touch")       # boxes overlap, outlines do not meet
        if q["left"] is False and o["x0"] < s["x0"] - R.POS_MARGIN * W:
            f.append("left")        # box starts further left, centre does not
        if q["above"] is False and o["y0"] < s["y0"] - R.POS_MARGIN * H:
            f.append("above")
        if f:
            ch[cid] = f
    ids = [c["id"] for c in cands]
    cc = {}
    for i, a in enumerate(ids):            # candidate-candidate touch, for CLOSURE
        for b in ids[i + 1:]:
            v = R.touch_value(gap(geo, m, a, b), diag)
            if v is not False:
                cc[(a, b)] = v
    # the shortcut answer for each counter-heuristic candidate: its flagged predicates flipped
    flip = {"larger": ["larger", "smaller"], "touch": ["touch"], "left": ["left"], "above": ["above"]}
    shortcut = {c: dict(preds[c], **{p: R.NOT(preds[c][p]) for f in fs for p in flip[f]}) for c, fs in ch.items()}
    rule_out = {}
    for t in R.TEMPLATES:
        S, U, keep, vals = R.evaluate(t["id"], preds, ids, cc)
        # counter-heuristic: answering this candidate by the shortcut would change its GT label
        chi = sorted(c for c in shortcut if c not in U and
                     (c in S) != (t["fn"]({**preds, c: shortcut[c]}, ids, cc)[c] is True))
        if t.get("closure"):
            # cell = (same kind, added through touching a same-kind object); fires when the closure adds one
            cells = [R.cell(t, preds[c]) + ("U" if vals[c] is None else "T" if vals[c] and not preds[c]["same_cat"] else "F")
                     for c in ids]
            fire = sum(cl.endswith("T") for cl in cells)
        else:
            cells = [R.cell(t, preds[c]) for c in ids]
            fire = sum(bool(t.get("fire") and t["fire"](preds[c])) for c in ids)
        rule_out[t["id"]] = dict(S=S, U=U, keep=keep, ch=chi, cells=" ".join(cells), fire=fire)
    diag_out = {}
    for t in R.DIAG_TEMPLATES:
        S, U, keep, _ = R.evaluate(t["id"], preds, ids, cc)
        diag_out[t["id"]] = dict(S=S, U=U, keep=keep, ch=[], cells=" ".join(R.cell(t, preds[c]) for c in ids), fire=0)
    return dict(
        id=sid, cat=cat, phrase=phrase, phrase_status=status, flags=flags,
        text_ok=status == "match" and "uncertain" not in flags,
        type_cat=subject_type_of(g["category_id"]), type_phrase=lexicon.phrase_subject_type(phrase),
        frac=round(frac, 4), cands=ids,
        cand_cats={str(c["id"]): CATS[c["category_id"]]["name"] for c in cands},
        preds={str(k): v for k, v in preds.items()},
        cc=[[a, b, v] for (a, b), v in cc.items()],
        ch={str(k): v for k, v in ch.items()},
        rules=rule_out,
        diag=diag_out,
        contact_stated={str(c): rel[frozenset((sid, c))] for c in ids if frozenset((sid, c)) in rel},
    )


def run_split(split, workers):
    R.check_template_set()
    meta = json.load(open(os.path.join(DATA, "index", split + ".json")))
    rp = os.path.join(DATA, "index", f"report_{split}.json")
    report = json.load(open(rp)) if os.path.exists(rp) else {}
    keys = sorted(meta["samples"])
    jobs = [(split, k, meta["samples"][k], os.path.join(DATA, meta["captions"], k + ".txt"),
             os.path.join(DATA, meta["mask_dir"], k + ".png"), report.get(k, {}).get("status", "ok")) for k in keys]
    od = os.path.join(OUT, split)
    os.makedirs(od, exist_ok=True)
    index, t0 = [], time.time()
    funnel_img = collections.Counter()
    funnel_anchor = collections.Counter()
    why = collections.Counter()
    with open(os.path.join(od, "details.jsonl"), "wb") as f, Pool(workers) as pool:
        for n, rec in enumerate(pool.imap(process, jobs, chunksize=16)):
            ok = rec["verify"] != "error" and rec["img_stage"] != "missing"
            if not ok:
                rec["img_stage"] = "verify_error" if rec["verify"] == "error" else rec["img_stage"]
            line = (json.dumps(rec, separators=(",", ":")) + "\n").encode()
            off = f.tell()
            f.write(line)
            funnel_img["all"] += 1
            if ok:
                funnel_img["verify_ok"] += 1
                for s, _ in STAGES:
                    if rec["img_stage"] in STAGE_IDX and STAGE_IDX[rec["img_stage"]] >= STAGE_IDX[s]:
                        funnel_img[s] += 1
                for sg in rec["segs"]:
                    if sg["stage"]:
                        for s, _ in STAGES:
                            if STAGE_IDX[sg["stage"]] >= STAGE_IDX[s]:
                                funnel_anchor[s] += 1
                    if sg["why"]:
                        nxt = STAGES[STAGE_IDX[sg["stage"]] + 1][0] if sg["stage"] else "thing"
                        why[nxt] += 1
            an = [a for a in rec["anchors"] if any(r["keep"] for r in a["rules"].values())]
            index.append(dict(
                key=rec["key"], stage=rec["img_stage"], off=off, len=len(line),
                n_anchors=len(an), n_text=sum(a["text_ok"] for a in an),
                io=rec.get("scene", {}).get("io"), io_stuff=rec.get("scene", {}).get("io_stuff"),
                settings=rec.get("scene", {}).get("settings", []), food=rec.get("scene", {}).get("food"),
                people=rec.get("people_mask"),
                types=sorted({a["type_cat"] for a in an}),
                cats=sorted({a["cat"] for a in an}),
                rules=sorted({t for a in an for t, r in a["rules"].items() if r["keep"]}),
                test_slice=split == "val" or in_test_slice(rec["key"]),
            ))
            if n % 2000 == 0:
                print(f"  {split}: {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    json.dump(index, open(os.path.join(od, "index.json"), "w"), separators=(",", ":"))
    stats = dict(images=funnel_img, anchors=funnel_anchor, rejected_at=why, seconds=round(time.time() - t0))
    json.dump(stats, open(os.path.join(od, "stats.json"), "w"), indent=1)
    print(split, json.dumps(stats))


# ---------------------------------------------------------------- sampling + summary
SIZE_BINS = ["0", "1", "2", "3", "4", "5+"]
BIN_SHARE = {"0": 0.15, "1": 0.17, "2": 0.17, "3": 0.17, "4": 0.17, "5+": 0.17}
TARGET = {"L1": 300, "L2": 500, "L3": 500}       # direct items per template (§3.2): 5×300 + 9×500 + 2×500 = 7,000
MAX_PER_ANCHOR = 3                               # target + twin and/or L1 companions (§3.2 nested sampling)
MIN_OPP = 0.70                                   # share of NOT/EXCEPT/CLOSURE items where that clause decides ≥1 candidate
L1_OF = {"touch": "T01", "larger": "T02", "smaller": "T02", "left": "T03", "above": "T04", "same_cat": "T05"}


def size_bin(n):
    return "5+" if n >= 5 else str(n)


def iter_details(split):
    p = os.path.join(OUT, split, "details.jsonl")
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                yield json.loads(line)


def has_opp_clause(t):
    return bool(t["neg"] or t["exc"] or t.get("closure"))


def make_item(a, t, r, role, group, rng):
    return dict(item_id=f"{t['id']}-{a['split']}-{a['key']}-{a['id']}", group=group, role=role,
                split=a["split"], key=a["key"], anchor=a["id"], tid=t["id"], level=t["level"],
                n=len(r["S"]), S=r["S"], U=r["U"], fire=r["fire"], ch=r["ch"], clutter=len(a["cands"]),
                para=rng.randrange(3), modalities=["text", "contour"])


DIAG_TARGET = 300        # C01 items in the caption-contact slice
DIAG_EMPTY = 0.15        # share of them with S* = {}; the rest have >= 1 caption-stated contact


def sample_diag(anchors, items, rng):
    """Caption-contact slice: each anchor gets C01 and its pair T01 (same candidates, two wordings).

    Anchors whose T01 is already in the main draft come first (no extra direct call), then other
    main-draft anchors (no extra probe), then fresh ones. A T01 item is added only when the main
    draft lacks it, so T01 items are never duplicated.
    """
    group_of = lambda a: f"{a['split']}-{a['key']}-{a['id']}"
    main_groups = {it["group"] for it in items}
    main_items = {(it["group"], it["tid"]) for it in items}
    out = []
    for t in R.DIAG_TEMPLATES:
        tid, pair = t["id"], t["pair"]
        ok = [a for a in anchors if a["diag"][tid]["keep"] and a["rules"][pair]["keep"]]
        rng.shuffle(ok)
        ok.sort(key=lambda a: ((group_of(a), pair) not in main_items, group_of(a) not in main_groups))
        n_empty = round(DIAG_TARGET * DIAG_EMPTY)
        got = collections.Counter()
        for a in ok:
            empty = not a["diag"][tid]["S"]
            if got["empty" if empty else "pos"] >= (n_empty if empty else DIAG_TARGET - n_empty):
                continue
            got["empty" if empty else "pos"] += 1
            g = group_of(a)
            out.append(make_item(a, t, a["diag"][tid], "diag", g, rng))
            if (g, pair) not in main_items:
                out.append(make_item(a, R.TEMPLATE_BY_ID[pair], a["rules"][pair], "diag_pair", g, rng))
    return out


def sample_nested(anchors, rng):
    """Subject-first draft Test sample (§3.2).

    Each anchor gets one L2/L3 *target* chosen by the largest remaining quota deficit, then up to two
    more templates: its twin, then the L1 rules for its primitives (*companions*). L2/L3 quotas are
    stratified by |S*|; L1 quotas are counts only, so companions keep their natural |S*|.
    Pass 1 takes one anchor per image, pass 2 any anchor, pass 3 relaxes the non-empty |S*| bins.
    """
    quota = {t["id"]: {b: round(TARGET[t["level"]] * BIN_SHARE[b]) for b in SIZE_BINS}
             for t in R.TEMPLATES if t["level"] != "L1"}
    total = {t["id"]: TARGET[t["level"]] for t in R.TEMPLATES}
    got = collections.defaultdict(collections.Counter)      # tid -> bin counts + "all" + "no_opp"
    items, used_anchor, used_img = [], set(), set()

    def fits(t, r, relax):
        tid = t["id"]
        if not r["keep"] or got[tid]["all"] >= total[tid]:
            return False
        if has_opp_clause(t) and not r["fire"] and got[tid]["no_opp"] >= round((1 - MIN_OPP) * total[tid]):
            return False
        if t["level"] == "L1":
            return True
        b = size_bin(len(r["S"]))
        if relax:
            return b != "0" or got[tid]["0"] < quota[tid]["0"]
        return got[tid][b] < quota[tid][b]

    def add(a, t, role, group):
        r = a["rules"][t["id"]]
        got[t["id"]]["all"] += 1
        got[t["id"]][size_bin(len(r["S"]))] += 1
        got[t["id"]]["no_opp"] += has_opp_clause(t) and not r["fire"]
        items.append(make_item(a, t, r, role, group, rng))

    for pass_ in (1, 2, 3):
        order = anchors[:]
        rng.shuffle(order)
        for a in order:
            ak = (a["split"], a["key"], a["id"])
            if ak in used_anchor or (pass_ == 1 and (a["split"], a["key"]) in used_img):
                continue
            # most under-filled L2/L3 template that this anchor can serve
            opts = [t for t in R.TEMPLATES if t["level"] != "L1" and fits(t, a["rules"][t["id"]], pass_ == 3)]
            if not opts:
                continue
            t = max(opts, key=lambda t: (1 - got[t["id"]]["all"] / total[t["id"]], rng.random()))
            group = f"{a['split']}-{a['key']}-{a['id']}"
            add(a, t, "target", group)
            taken = [t["id"]]
            extra = ([R.TEMPLATE_BY_ID[t["twin"]]] if t.get("twin") else []) + \
                [R.TEMPLATE_BY_ID[L1_OF[p]] for p in t["prims"] if p in L1_OF]
            for e in extra:
                if len(taken) >= MAX_PER_ANCHOR or e["id"] in taken:
                    continue
                if fits(e, a["rules"][e["id"]], pass_ == 3):
                    add(a, e, "twin" if e["id"] == t.get("twin") else "companion", group)
                    taken.append(e["id"])
            # spare slots: more L2/L3 targets, preferring ones that share a primitive with the first
            # (fewer anchors -> fewer probe calls, and more within-anchor operator contrasts)
            while len(taken) < MAX_PER_ANCHOR:
                more = [u for u in R.TEMPLATES if u["level"] != "L1" and u["id"] not in taken
                        and fits(u, a["rules"][u["id"]], pass_ == 3)]
                if not more:
                    break
                u = max(more, key=lambda u: (bool(set(u["prims"]) & set(t["prims"])),
                                             1 - got[u["id"]]["all"] / total[u["id"]], rng.random()))
                add(a, u, "target", group)
                taken.append(u["id"])
            used_anchor.add(ak)
            used_img.add((a["split"], a["key"]))
    # L1 quotas still open: standalone L1 items on unused anchors
    order = [a for a in anchors if (a["split"], a["key"], a["id"]) not in used_anchor]
    rng.shuffle(order)
    for a in order:
        for t in R.TEMPLATES:
            if t["level"] == "L1" and fits(t, a["rules"][t["id"]], True):
                add(a, t, "target", f"{a['split']}-{a['key']}-{a['id']}")
                break
    return items


def summarise_and_sample(seed=0):
    rng = random.Random(seed)
    checks = R.check_template_set()
    splits = [s for s in ("val", "train") if os.path.exists(os.path.join(OUT, s, "index.json"))]
    anchors = []                                       # Test pool: val + held-out train slice, text_ok subjects
    rule_stats = {s: {t["id"]: collections.Counter() for t in R.TEMPLATES} for s in splits}
    sizes = {s: {t["id"]: collections.Counter() for t in R.TEMPLATES} for s in splits}
    scene = {s: collections.defaultdict(collections.Counter) for s in splits}
    twin_mismatch = collections.Counter()
    diag_stats = {s: {t["id"]: collections.Counter() for t in R.DIAG_TEMPLATES} for s in splits}
    diag_sizes = {s: {t["id"]: collections.Counter() for t in R.DIAG_TEMPLATES} for s in splits}
    contact_audit = {s: collections.Counter() for s in splits}   # caption-stated pairs vs. mask geometry
    for split in splits:
        for rec in iter_details(split):
            if rec["img_stage"] != "rule_kept" or rec["verify"] == "error":
                continue
            sc = rec["scene"]
            scene[split]["io"][sc["io"]] += 1
            scene[split]["io_stuff"][sc["io_stuff"]] += 1
            scene[split]["io_agree"]["agree" if sc["io"] == sc["io_stuff"] else
                                     ("undecided" if "unknown" in (sc["io"], sc["io_stuff"]) else "disagree")] += 1
            scene[split]["food"][str(sc["food"])] += 1
            scene[split]["people_in_image"][str(rec["people_mask"])] += 1
            for st in sc["settings"] or ["(none)"]:
                scene[split]["settings"][st] += 1
            for a in rec["anchors"]:
                if not any(r["keep"] for r in a["rules"].values()):
                    continue
                scene[split]["subject_type"][a["type_cat"]] += 1
                scene[split]["subject_type_phrase_agree"]["agree" if a["type_cat"] == a["type_phrase"] else
                                                          ("phrase unknown" if a["type_phrase"] == "unknown" else "disagree")] += 1
                scene[split]["subject_category"][a["cat"]] += 1
                scene[split]["text_ok"][str(a["text_ok"])] += 1
                scene[split]["clutter"][len(a["cands"])] += 1
                for x, y in R.TWINS:
                    twin_mismatch[f"{x}/{y}"] += a["rules"][x]["S"] != a["rules"][y]["S"]
                for tid, r in a["rules"].items():
                    rs = rule_stats[split][tid]
                    rs["instances"] += 1
                    rs["kept"] += r["keep"]
                    rs["pairs"] += len(a["cands"])
                    rs["pairs_U"] += len(r["U"])
                    rs["pairs_pos"] += len(r["S"])
                    if not r["keep"]:
                        continue
                    n = len(r["S"])
                    sizes[split][tid][size_bin(n)] += 1
                    rs["empty"] += n == 0
                    rs["sum_S"] += n
                    rs["ch"] += bool(r["ch"])
                    rs["opp"] += r["fire"] > 0
                for tid, r in a["diag"].items():
                    ds = diag_stats[split][tid]
                    ds["instances"] += 1
                    ds["kept"] += r["keep"]
                    ds["pairs"] += len(a["cands"])
                    ds["pairs_U"] += len(r["U"])
                    if r["keep"]:
                        diag_sizes[split][tid][size_bin(len(r["S"]))] += 1
                        ds["empty"] += not r["S"]
                        ds["sum_S"] += len(r["S"])
                ca = contact_audit[split]
                for c, q in a["preds"].items():
                    stated = a["contact_stated"].get(c)
                    ca["pairs_outlines_meet"] += q["touch"] is True
                    ca["pairs_outlines_meet_stated"] += q["touch"] is True and stated is not None
                    if stated:
                        ca["stated_" + stated] += 1
                        ca["stated_" + ("meet" if q["touch"] is True else "near" if q["touch"] is None else "far")] += 1
                if a["text_ok"] and (split == "val" or in_test_slice(rec["key"])):
                    anchors.append(dict(split=split, key=rec["key"], id=a["id"], cands=a["cands"],
                                        rules={t: {k: r[k] for k in ("S", "U", "keep", "fire", "ch")}
                                               for t, r in a["rules"].items()},
                                        diag={t: {k: r[k] for k in ("S", "U", "keep", "fire", "ch")}
                                              for t, r in a["diag"].items()}))
    items = sample_nested(anchors, rng)
    diag_items = sample_diag(anchors, items, rng)
    with open(os.path.join(OUT, "test_diag.jsonl"), "w") as f:
        for it in diag_items:
            f.write(json.dumps(it) + "\n")
    with open(os.path.join(OUT, "test_draft.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    draft = collections.defaultdict(collections.Counter)
    for it in items:
        d = draft[it["tid"]]
        d[size_bin(it["n"])] += 1
        d["all"] += 1
        d["role_" + it["role"]] += 1
        d["src_" + it["split"]] += 1
        d["ch"] += bool(it["ch"])
        d["opp"] += it["fire"] > 0
    groups = collections.Counter(it["group"] for it in items)
    in_group = collections.defaultdict(set)
    for it in items:
        in_group[it["group"]].add(it["tid"])
    for it in items:                               # how many L2/L3 items have their L1 companions on the same anchor
        t = R.TEMPLATE_BY_ID[it["tid"]]
        if t["level"] == "L1":
            continue
        comp = {L1_OF[p] for p in t["prims"] if p in L1_OF}
        have = comp & in_group[it["group"]]
        draft[it["tid"]]["paired_all"] += have == comp
        draft[it["tid"]]["paired_any"] += bool(have)
    closure_groups = {it["group"] for it in items if R.TEMPLATE_BY_ID[it["tid"]].get("closure")}
    stats = {s: json.load(open(os.path.join(OUT, s, "stats.json"))) for s in splits}
    summary = dict(
        generated=time.strftime("%Y-%m-%d %H:%M"),
        thresholds={k: getattr(R, k) for k in ["SUBJ_MIN", "SUBJ_MAX", "CAND_MIN", "MIN_CANDS", "MAX_CANDS",
                                               "SIZE_MARGIN", "TOUCH_PX", "FAR_FRAC", "POS_MARGIN", "BORDER_PX",
                                               "BORDER_FRAC", "U_DROP"]},
        stages=[dict(id=s, desc=d) for s, d in STAGES],
        funnel={s: stats[s] for s in splits},
        predicates=R.PREDICATES,
        modalities=R.MODALITIES,
        templates=[{k: v for k, v in t.items() if k not in ("fn", "fire")} | {"has_opp": has_opp_clause(t)}
                   for t in R.TEMPLATES],
        extensional_twins=R.TWINS, template_checks=checks, twin_mismatch=twin_mismatch,
        rule_stats=rule_stats, size_hist=sizes, scene=scene,
        draft=dict(n=len(items), per_template=draft, targets=TARGET, bin_share=BIN_SHARE, min_opp=MIN_OPP,
                   max_per_anchor=MAX_PER_ANCHOR, anchors=len(groups),
                   images=len({(it["split"], it["key"]) for it in items}),
                   templates_per_anchor=collections.Counter(groups.values()),
                   probes_with_touches=len(closure_groups),
                   calls_per_model=dict(direct=2 * len(items), probe=2 * len(groups))),
        diagnostic=dict(
            templates=[{k: v for k, v in t.items() if k != "fn"} for t in R.DIAG_TEMPLATES],
            rule_stats=diag_stats, size_hist=diag_sizes, contact_audit=contact_audit,
            sample=dict(n=len(diag_items), target=DIAG_TARGET, empty_share=DIAG_EMPTY,
                        per_role=collections.Counter(it["role"] for it in diag_items),
                        per_size=collections.Counter(size_bin(it["n"]) for it in diag_items if it["role"] == "diag"),
                        anchors=len({it["group"] for it in diag_items}),
                        anchors_in_main=len({it["group"] for it in diag_items} & set(groups)),
                        calls_per_model=dict(direct=2 * len(diag_items)))),   # direct only: no probe needed
    )
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=str)
    print("draft items", len(items), "anchors", len(groups), "images", summary["draft"]["images"],
          "per-anchor", dict(summary["draft"]["templates_per_anchor"]))
    print("diagnostic items", len(diag_items), dict(summary["diagnostic"]["sample"]["per_role"]),
          "sizes", dict(summary["diagnostic"]["sample"]["per_size"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "train"])
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N images")
    ap.add_argument("--sample", action="store_true")
    a = ap.parse_args()
    if a.split:
        if a.limit:
            meta = json.load(open(os.path.join(DATA, "index", a.split + ".json")))
            k = sorted(meta["samples"])[: a.limit]
            for key in k:
                r = process((a.split, key, meta["samples"][key], os.path.join(DATA, meta["captions"], key + ".txt"),
                             os.path.join(DATA, meta["mask_dir"], key + ".png"), "ok"))
                print(key, r["img_stage"], [(s["id"], s["cat"], s["stage"], s["why"]) for s in r["segs"]][:6])
        else:
            run_split(a.split, a.workers)
    if a.sample:
        summarise_and_sample()
