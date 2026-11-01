"""Build Lite and the model-call jobs for result rows 1-3, and render every model input.

  python tools/eval/build_jobs.py            # writes data/anchor_gt/lite.json, data/eval/jobs.jsonl, data/eval/renders/

Rows (paper_scaffold.md, "Supplementary tables"):
  1  Table 2/3, S4-S6: Test direct (M-contour), shared probe per Test anchor, pairwise replica on Lite
  2  S7: probe re-run (two sampled draws) per Lite anchor, rule-specific probe on 500 Lite items
  3  S3: negation minimal pairs on 500 Lite items, caption-contact slice in M-text and M-contour
  4  S2: the whole Test set again in M-text (direct + shared probe); margin and subset variants are offline
  5  S9: Lite in M-mark and in the alternative marker style (render.py style="alt"), direct + shared probe
Jobs are model-independent; run.py executes them for one model.
"""
import collections
import json
import os
import random
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools", "anchor_gt"))
sys.path.insert(0, HERE)
import rules as R  # noqa: E402
import render  # noqa: E402

GT = os.path.join(ROOT, "data", "anchor_gt")
EV = os.path.join(ROOT, "data", "eval")

LITE_N = 1500
NEG_N = 500                 # negation minimal pairs (design §5.1f)
RULE_PROBE_N = 500          # probe-equivalence check (design §4.3)
REPLICA_N = 1500            # single-pair yes/no questions, balanced per primitive and answer
REPLICA_PRIMS = ["same_cat", "larger", "touch", "left", "above", "border"]   # smaller is larger's complement
RETEST_SEEDS = [1, 2]
SAMPLED = dict(temperature=0.7, top_p=0.8, top_k=20)  # Qwen3-VL's own generation_config, for the re-run only
MAX_TOK = dict(direct=1024, probe=4096, replica=16)


def read_jsonl(p):
    return [json.loads(l) for l in open(p)]


def sample_lite(items, rng):
    """Whole anchors from Test until each template has its Test share of LITE_N (fewer probe calls)."""
    total = collections.Counter(it["tid"] for it in items)
    quota = {t: round(LITE_N * n / len(items)) for t, n in total.items()}
    groups = collections.defaultdict(list)
    for it in items:
        groups[it["group"]].append(it)
    order = sorted(groups)
    rng.shuffle(order)
    got, lite = collections.Counter(), []
    for g in order:
        take = [it for it in groups[g] if got[it["tid"]] < quota[it["tid"]]]
        for it in take:
            got[it["tid"]] += 1
            lite.append(it)
    return lite


def stratified(items, n, key, rng):
    by = collections.defaultdict(list)
    for it in items:
        by[key(it)].append(it)
    for v in by.values():
        rng.shuffle(v)
    out, i = [], 0
    while len(out) < n and any(i < len(v) for v in by.values()):
        out += [v[i] for v in by.values() if i < len(v)][: n - len(out)]
        i += 1
    return out


def replica_pairs(lite_groups, rng):
    pool = collections.defaultdict(list)            # (prim, answer) -> [(group, cand)]
    for g, (split, key, aid) in lite_groups.items():
        a = render.anchor(split, key, aid)
        for c in a["cands"]:
            q = a["preds"][str(c)]
            for p in REPLICA_PRIMS:
                if q[p] is not None:
                    pool[(p, bool(q[p]))].append((g, c))
    per = REPLICA_N // (2 * len(REPLICA_PRIMS))
    out = []
    for (p, ans), v in sorted(pool.items()):
        for g, c in rng.sample(v, min(per, len(v))):
            out.append(dict(group=g, cand=c, prim=p, answer=ans))
    return out


def render_dir(mod, style):
    return mod + ("-alt" if style == "alt" else "")


def _render(args):
    split, key, aid, mod, style = args
    p = os.path.join(EV, "renders", render_dir(mod, style), f"{split}-{key}-{aid}.jpg")
    if not os.path.exists(p):
        render.render(split, key, aid, mod, p, style)
    return p


def main():
    rng = random.Random(0)
    test = read_jsonl(os.path.join(GT, "test_draft.jsonl"))
    diag = read_jsonl(os.path.join(GT, "test_diag.jsonl"))
    lite = sample_lite(test, rng)
    neg = stratified([it for it in lite if it["tid"] in R.POSITIVE_TEXT], NEG_N, lambda it: it["tid"], rng)
    rule_probe = stratified(lite, RULE_PROBE_N, lambda it: it["tid"], rng)
    lite_groups = {it["group"]: (it["split"], it["key"], it["anchor"]) for it in lite}
    replica = replica_pairs(lite_groups, rng)
    json.dump(dict(items=[it["item_id"] for it in lite], neg=[it["item_id"] for it in neg],
                   rule_probe=[it["item_id"] for it in rule_probe], replica=replica,
                   anchors=len(lite_groups), per_template=collections.Counter(it["tid"] for it in lite)),
              open(os.path.join(GT, "lite.json"), "w"), indent=1)

    closure_groups = {it["group"] for it in test if R.TEMPLATE_BY_ID[it["tid"]].get("closure")}
    test_groups = {it["group"]: (it["split"], it["key"], it["anchor"]) for it in test}
    jobs = []

    def phrase(split, key, aid):
        return render.anchor(split, key, aid)["phrase"]

    def job(row, kind, mod, split, key, aid, prompt, mtok, ref, style="outline", **kw):
        sp = dict(temperature=0.0, max_tokens=mtok)
        if "seed" in kw:
            sp.update(SAMPLED, seed=kw["seed"])
        jid = f"{kind}|{mod}|{ref}" + (f"|s{kw['seed']}" if "seed" in kw else "")
        jobs.append(dict(job_id=jid, row=row, kind=kind, modality=mod, style=style, split=split, key=key, anchor=aid,
                         image=f"data/eval/renders/{render_dir(mod, style)}/{split}-{key}-{aid}.jpg", prompt=prompt,
                         sampling=sp, **{k: v for k, v in kw.items() if k != "seed"}))

    def direct(row, kind, it, mod, text=None, para=None, style="outline"):
        para = it["para"] if para is None else para
        ph = phrase(it["split"], it["key"], it["anchor"]) if mod == "text" else None
        marks = render.anchor(it["split"], it["key"], it["anchor"])["cands"]
        p = R.render_direct(it["tid"], para, mod, ph, it["anchor"], text=text, marks=marks)
        job(row, kind, mod, it["split"], it["key"], it["anchor"], p, MAX_TOK["direct"], it["item_id"] + f"|p{para}",
            style=style, item_id=it["item_id"], tid=it["tid"], para=para)

    def probe(row, kind, g, mod, seed=None, tid=None, item_id=None, style="outline"):
        split, key, aid = test_groups[g]
        ph = phrase(split, key, aid) if mod == "text" else None
        marks = render.anchor(split, key, aid)["cands"]
        if tid:
            p = R.render_probe(tid, mod, ph, aid, marks=marks)
        else:
            p = R.render_probe_all(mod, ph, aid, touches=g in closure_groups, marks=marks)
        kw = dict(group=g, touches=g in closure_groups and not tid)
        if seed is not None:
            kw["seed"] = seed
        if tid:
            kw.update(tid=tid, item_id=item_id)
        job(row, kind, mod, split, key, aid, p, MAX_TOK["probe"], item_id or g, style=style, **kw)

    # row 1
    for it in test:
        direct(1, "direct", it, "contour")
    for g in sorted(test_groups):
        probe(1, "probe", g, "contour")
    for i, r in enumerate(replica):
        split, key, aid = lite_groups[r["group"]]
        marks = render.anchor(split, key, aid)["cands"]
        job(1, "replica", "contour", split, key, aid, R.render_replica(r["prim"], r["cand"], "contour", sid=aid, marks=marks),
            MAX_TOK["replica"], f"{r['group']}|{r['cand']}|{r['prim']}", group=r["group"], cand=r["cand"],
            prim=r["prim"], answer=r["answer"])
    # row 2
    for g in sorted(lite_groups):
        for s in RETEST_SEEDS:
            probe(2, "probe_retest", g, "contour", seed=s)
    for it in rule_probe:
        probe(2, "probe_rule", it["group"], "contour", tid=it["tid"], item_id=it["item_id"])
    # row 3
    for it in neg:
        direct(3, "direct_pos", it, "contour", text=R.POSITIVE_TEXT[it["tid"]], para=0)
        if it["para"] != 0:                      # the Test run used another paraphrase: add the negated para 0
            direct(3, "direct_neg0", it, "contour", para=0)
    main_t01 = {it["group"]: it for it in test if it["tid"] == "T01"}
    for it in diag:
        for mod in ("text", "contour"):
            direct(3, "direct_diag", it, mod)
        if it["tid"] == "C01" and it["group"] in main_t01:   # paired T01 is in Test, which runs M-contour only
            direct(3, "direct_diag", main_t01[it["group"]], "text")

    # row 4
    for it in test:
        direct(4, "direct_text", it, "text")
    for g in sorted(test_groups):
        probe(4, "probe_text", g, "text")
    # row 5
    lite_items = [it for it in test if it["group"] in lite_groups and it["item_id"] in {x["item_id"] for x in lite}]
    for it in lite_items:
        direct(5, "direct_mark", it, "mark")
        direct(5, "direct_alt", it, "contour", style="alt")
    for g in sorted(lite_groups):
        probe(5, "probe_mark", g, "mark")
        probe(5, "probe_alt", g, "contour", style="alt")

    assert len({j["job_id"] for j in jobs}) == len(jobs), "duplicate job ids"
    need = sorted({(j["split"], j["key"], j["anchor"], j["modality"], j["style"]) for j in jobs})
    with Pool(14) as pool:
        for i, _ in enumerate(pool.imap_unordered(_render, need, chunksize=8)):
            if i % 1000 == 0:
                print(f"render {i}/{len(need)}", flush=True)
    os.makedirs(EV, exist_ok=True)
    with open(os.path.join(EV, "jobs.jsonl"), "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    c = collections.Counter((j["row"], j["kind"]) for j in jobs)
    print("lite items", len(lite), "anchors", len(lite_groups), "| replica", len(replica), "| neg", len(neg))
    for k in sorted(c):
        print(" row", k[0], k[1], c[k])
    print("total jobs", len(jobs), "renders", len(need))


if __name__ == "__main__":
    main()
