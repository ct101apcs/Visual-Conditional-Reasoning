"""Compute the paper's result tables (Table 2, Table 3, S1-S9) from finished runs.

  python3 tools/eval/analyze.py qwen3-vl-4b-instruct qwen3-vl-8b-instruct internvl3-8b-hf
  -> data/eval/results/results.md and results.json

95% CIs come from an image-clustered bootstrap (B=1000): images are resampled, and every item and
candidate of a drawn image comes with it. Candidates in an ignore band (U) are never scored.
"""
import collections
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools", "anchor_gt"))
sys.path.insert(0, HERE)
import rules as R  # noqa: E402
import render  # noqa: E402
import score  # noqa: E402

GT = os.path.join(ROOT, "data", "anchor_gt")
OUT = os.path.join(ROOT, "data", "eval", "results")
B = 1000
NAMES = {"qwen3-vl-4b-instruct": "Qwen3-VL-4B", "qwen3-vl-8b-instruct": "Qwen3-VL-8B", "internvl3-8b-hf": "InternVL3-8B",
         "qwen3-vl-8b-thinking": "Qwen3-VL-8B-Thinking", "gemma-3-4b-it": "Gemma-3-4B"}
L1_OF = {"touch": "T01", "larger": "T02", "smaller": "T02", "left": "T03", "above": "T04", "same_cat": "T05"}
REPLICA_PRIMS = ["same_cat", "larger", "touch", "left", "above", "border"]
OPS = [("AND", ["T06", "T07"], "keep"), ("AND-NOT", ["T08", "T09", "T10", "T11"], "fire"),
       ("EXCEPT", ["T12", "T13", "T14"], "fire"), ("EXCEPT, base clause", ["T12", "T13", "T14"], "keep"),
       ("L3 nested", ["T15"], "fire"), ("CLOSURE", ["T16"], "added")]


# ---------------------------------------------------------------- bootstrap
class Boot:
    """Image-clustered bootstrap. stat(w) receives one weight per image (multinomial counts)."""

    def __init__(self, clusters, seed=0):
        self.ids = sorted(set(clusters))
        self.ix = {c: i for i, c in enumerate(self.ids)}
        rng = np.random.default_rng(seed)
        n = len(self.ids)
        self.W = rng.multinomial(n, np.full(n, 1 / n), size=B).astype(float)

    def idx(self, clusters):
        return np.array([self.ix[c] for c in clusters])

    def ci(self, stat):
        est = stat(np.ones(len(self.ids)))
        draws = np.array([stat(w) for w in self.W])
        draws = draws[np.isfinite(draws)]
        lo, hi = np.percentile(draws, [2.5, 97.5]) if len(draws) else (math.nan, math.nan)
        return est, lo, hi


def wmean(x, ci_idx):
    """stat factory: weighted mean of x whose rows belong to images ci_idx."""
    x = np.asarray(x, float)
    return lambda w: (w[ci_idx] * x).sum() / w[ci_idx].sum() if w[ci_idx].sum() else math.nan


# ---------------------------------------------------------------- data
def jl(p):
    return [json.loads(l) for l in open(p)]


def img(it):
    return f"{it['split']}-{it['key']}"


class Run:
    def __init__(self, name):
        self.name = name
        thinking = "thinking" in name
        jobs, resp = score.load(os.path.join(ROOT, "data", "eval", "runs", name))
        self.direct, self.probe, self.replica = {}, {}, {}
        self.parsed = collections.Counter()
        self.n = collections.Counter()
        self.trunc = collections.Counter()
        for jid, r in resp.items():
            j = jobs[jid]
            a = render.anchor(j["split"], j["key"], j["anchor"])
            k = j["kind"]
            self.trunc[k] += r.get("finish_reason") == "length"
            if k.startswith("direct"):
                d = score.parse_direct(score.answer_text(r["output"], thinking), a["cands"], a["id"])
                self.n[k] += 1
                self.parsed[k] += d["ok"]
                self.direct[(k, j["modality"], j["item_id"])] = set(d["ids"]) if d["ok"] else set()
            elif k.startswith("probe"):
                prims = R.TEMPLATE_BY_ID[j["tid"]]["probe"] if j.get("tid") else R.PROBE_PRIMS
                d = score.parse_probe(score.answer_text(r["output"], thinking), a["cands"], prims, j.get("touches"))
                self.n[k] += 1
                self.parsed[k] += d["ok"]
                ref = j.get("item_id") or j["group"]
                seed = jid.rsplit("|", 1)[1] if k == "probe_retest" else None
                self.probe[(k, j["modality"], ref, seed)] = d["ans"] if d["ok"] else {}
            else:
                d = score.parse_replica(score.answer_text(r["output"], thinking))
                self.n[k] += 1
                self.parsed[k] += d["ok"]
                self.replica[jid] = (j, d.get("answer"))

    def d(self, kind, mod, it):
        return self.direct.get((kind, mod, it["item_id"]))

    def p(self, kind, mod, group, seed=None):
        return self.probe.get((kind, mod, group, seed))


def anchor_of(it):
    return render.anchor(it["split"], it["key"], it["anchor"])


def scored(it):
    return [c for c in anchor_of(it)["cands"] if c not in set(it["U"])]


def exec_ids(it, ans):
    return set(score.execute(it["tid"], ans or {}, anchor_of(it)["cands"]))


def f1(pred, it):
    return score.set_f1(pred, it["S"], it["U"])


def fmt(est, lo=None, hi=None, d=3, pct=False):
    if est is None or (isinstance(est, float) and math.isnan(est)):
        return "—"
    m = 100 if pct else 1
    s = f"{est * m:.{d}f}"
    if lo is not None and not math.isnan(lo):
        s += f" ({lo * m:.{d}f}, {hi * m:.{d}f})"
    return s


def ci_mean(boot, rows):
    """rows: [(image, value)] -> (est, lo, hi)"""
    if not rows:
        return (math.nan, math.nan, math.nan)
    ix = boot.idx([r[0] for r in rows])
    return boot.ci(wmean([r[1] for r in rows], ix))


def ci_diff(boot, a, b):
    """mean(a) - mean(b), each [(image, value)], resampled jointly."""
    if not a or not b:
        return (math.nan, math.nan, math.nan)
    fa, fb = wmean([r[1] for r in a], boot.idx([r[0] for r in a])), wmean([r[1] for r in b], boot.idx([r[0] for r in b]))
    return boot.ci(lambda w: fa(w) - fb(w))


def kendall(x, y):
    n, s = len(x), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
    return s / (n * (n - 1) / 2)


# ---------------------------------------------------------------- GT under other margins (Table S2)
def regt(it, margins):
    """S, U of an item recomputed from stored raw measures under other thresholds (CLOSURE keeps stored cc)."""
    a, rec = anchor_of(it), render.record(it["split"], it["key"])
    W, H = rec["W"], rec["H"]
    diag = (W * W + H * H) ** 0.5
    sm, tp, pm = margins
    p = {}
    for c in a["cands"]:
        q = a["preds"][str(c)]
        gap = math.inf if q["gap"] is None else q["gap"]
        p[c] = dict(q, larger=R.band(q["ratio"], 1 / sm, sm), smaller=R.band(1 / q["ratio"], 1 / sm, sm),
                    touch=True if gap <= tp else (False if gap > R.FAR_FRAC * diag else None),
                    left=R.band(q["dx"], -pm * W, pm * W), above=R.band(q["dy"], -pm * H, pm * H))
    cc = {(min(x, y), max(x, y)): v for x, y, v in a["cc"]}
    vals = R.TEMPLATE_BY_ID[it["tid"]]["fn"](p, a["cands"], cc)
    S = [c for c in a["cands"] if vals[c] is True]
    U = [c for c in a["cands"] if vals[c] is None]
    return dict(it, S=S, U=U), len(U) <= R.U_DROP * len(a["cands"])


# ---------------------------------------------------------------- tables
def main(names):
    test = jl(os.path.join(GT, "test_draft.jsonl"))
    diag = jl(os.path.join(GT, "test_diag.jsonl"))
    lite = json.load(open(os.path.join(GT, "lite.json")))
    lite_ids, neg_ids = set(lite["items"]), set(lite["neg"])
    by_group = collections.defaultdict(dict)
    for it in test:
        by_group[it["group"]][it["tid"]] = it
    boot = Boot([img(it) for it in test + diag])
    runs = {n: Run(n) for n in names}
    res, md = {}, []

    def row_md(cells):
        return "| " + " | ".join(cells) + " |"

    # ---- per-item direct / executed F1 and gap, M-contour and M-text
    def item_scores(run, mod):
        dk, pk = ("direct", "probe") if mod == "contour" else ("direct_text", "probe_text")
        out = []
        for it in test:
            d = run.d(dk, mod, it)
            e = exec_ids(it, run.p(pk, mod, it["group"]))
            fd, fe = f1(d or set(), it), f1(e, it)
            out.append(dict(it=it, img=img(it), fd=fd, fe=fe, g=fe - fd))
        return out

    scores = {n: {m: item_scores(r, m) for m in ("contour", "text")} for n, r in runs.items()}

    def dG(rows):
        l1 = [(r["img"], r["g"]) for r in rows if r["it"]["level"] == "L1"]
        l23 = [(r["img"], r["g"]) for r in rows if r["it"]["level"] != "L1"]
        return ci_diff(boot, l23, l1)

    # ---- Table 2
    md.append("## Table 2. Pairwise accuracy versus set selection (Test, M-contour)\n")
    md.append("| Model | Replica acc. | Probe bal. acc. | Direct F1 L1 | L2 | L3 | Executed F1 L2/L3 | ΔG = G(L2/L3) − G(L1) | Rank pairwise → set |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    lv = lambda rows, L, key: ci_mean(boot, [(r["img"], r[key]) for r in rows if r["it"]["level"] in L])
    # trivial baselines
    md.append("| *Trivial baselines* | | | | | | | | |")
    empty = [(img(it), it["level"], f1(set(), it)) for it in test]
    md.append(row_md(["Answer `[]`", "—", "—"] + [fmt(*ci_mean(boot, [(i, v) for i, l, v in empty if l == L]), d=2)
                                                for L in ("L1", "L2", "L3")] + ["—", "—", "—"]))
    rng = np.random.default_rng(1)
    prior = collections.defaultdict(list)
    for it in test:
        sc = scored(it)
        if sc:
            prior[it["tid"]].append(len(set(it["S"]) & set(sc)) / len(sc))
    prior = {t: float(np.mean(v)) for t, v in prior.items()}
    rand = []
    for it in test:
        sc = scored(it)
        v = np.mean([f1({c for c in sc if rng.random() < prior[it["tid"]]}, it) for _ in range(20)])
        rand.append((img(it), it["level"], v))
    md.append(row_md(["Random at prior (per template)", "—", "—"] + [fmt(*ci_mean(boot, [(i, v) for i, l, v in rand if l == L]), d=2)
                                                                    for L in ("L1", "L2", "L3")] + ["—", "—", "—"]))
    md.append("| Class list only, no image | not run | | | | | | | |")
    md.append("| SAM 3 Agent | not run | | | | | | | |")
    md.append("| Off-the-shelf segmenter + rule program | not run | | | | | | | |")
    t2 = {}
    for n, r in runs.items():
        rows = scores[n]["contour"]
        rep = [(img(j), float(a == j["answer"])) for j, a in r.replica.values()]
        # probe balanced accuracy over the replica primitives, on decided candidates of Test anchors
        pr = []
        for g, its in by_group.items():
            it = next(iter(its.values()))
            a, ans = anchor_of(it), r.p("probe", "contour", g) or {}
            for c in a["cands"]:
                q = a["preds"][str(c)]
                for p in REPLICA_PRIMS:
                    if q[p] is not None:
                        m = (ans.get(c) or {}).get(p)
                        pr.append((img(it), p, bool(q[p]), m is not None and m == q[p]))
        pix = boot.idx([x[0] for x in pr])
        prim = np.array([x[1] for x in pr])
        pos = np.array([x[2] for x in pr])
        cor = np.array([x[3] for x in pr], float)

        def balacc(w):
            ww = w[pix]
            accs = []
            for p in REPLICA_PRIMS:
                for s in (True, False):
                    m = (prim == p) & (pos == s)
                    accs.append((ww[m] * cor[m]).sum() / ww[m].sum())
            return float(np.mean(accs))
        t2[n] = dict(rep=ci_mean(boot, rep), bal=boot.ci(balacc),
                     d1=lv(rows, ["L1"], "fd"), d2=lv(rows, ["L2"], "fd"), d3=lv(rows, ["L3"], "fd"),
                     d23=lv(rows, ["L2", "L3"], "fd"), e23=lv(rows, ["L2", "L3"], "fe"), dG=dG(rows))
    order_pair = sorted(runs, key=lambda n: -t2[n]["rep"][0])
    order_set = sorted(runs, key=lambda n: -t2[n]["d23"][0])
    md.append("| *Open VLMs* | | | | | | | | |")
    for n in sorted(runs, key=lambda n: -t2[n]["d23"][0]):
        t = t2[n]
        md.append(row_md([NAMES[n], fmt(*t["rep"], d=2), fmt(*t["bal"], d=2), fmt(t["d1"][0], d=2), fmt(t["d2"][0], d=2),
                          fmt(t["d3"][0], d=2), fmt(t["e23"][0], d=2), "**" + fmt(*t["dG"]) + "**",
                          f"{order_pair.index(n) + 1} → {order_set.index(n) + 1}"]))
    tau = kendall([t2[n]["rep"][0] for n in runs], [t2[n]["d23"][0] for n in runs])
    md.append(f"| **Kendall τ, replica vs. direct set (n = {len(runs)} models)** | | | | | | | τ = {tau:.2f} (no CI with {len(runs)} models) | |")
    md.append("\nDirect F1 cells: L1/L2/L3 point estimates; CIs in results.json. Executed F1 runs the rule on the model's own shared-probe answers.\n")
    res["table2"] = {n: {k: list(v) for k, v in t.items()} for n, t in t2.items()} | {"tau": tau}

    # ---- Table 3 (per model)
    def cell_rows(run, it, kind):
        """(candidate, in_cell, must_select) for the deciding cell of `kind`."""
        a, t = anchor_of(it), R.TEMPLATE_BY_ID[it["tid"]]
        out = []
        for c in scored(it):
            q = a["preds"][str(c)]
            if kind == "keep":
                ok = c in it["S"]
            elif kind == "fire":
                ok = bool(t["fire"](q))
            else:  # CLOSURE: added through a same-kind object
                ok = c in it["S"] and not q["same_cat"]
            if ok:
                out.append((c, c in it["S"]))
        return out

    t3 = {}
    for n, r in runs.items():
        t3[n] = {}
        for label, tids, kind in OPS:
            err, cost, comp, perc, over = [], [], [], [], []
            for it in test:
                if it["tid"] not in tids:
                    continue
                d = r.d("direct", "contour", it) or set()
                e = exec_ids(it, r.p("probe", "contour", it["group"]))
                for c, must in cell_rows(r, it, kind):
                    wrong = (c in d) != must
                    err.append((img(it), float(wrong)))
                    if wrong:
                        ok_e = (c in e) == must
                        comp.append((img(it), float(ok_e)))
                    # paired L1 errors on the same anchor
                    l1e = []
                    for p in R.TEMPLATE_BY_ID[it["tid"]]["prims"]:
                        u = by_group[it["group"]].get(L1_OF.get(p, ""))
                        if u and c not in u["U"]:
                            du = r.d("direct", "contour", u) or set()
                            l1e.append(float((c in du) != (c in u["S"])))
                    if l1e:
                        cost.append((img(it), float(wrong) - float(np.mean(l1e))))
                if kind == "fire":   # over-applied: wrongly dropping a candidate the clause does not exclude
                    for c in it["S"]:
                        over.append((img(it), float(c not in d)))
            t3[n][label] = dict(err=ci_mean(boot, err), cost=ci_mean(boot, cost), comp=ci_mean(boot, comp),
                                over=ci_mean(boot, over) if over else (math.nan,) * 3, n=len(err))
    md.append("## Table 3. Where selections go wrong, by operator (Test, M-contour)\n")
    md.append("Each cell: median across models [min – max]. Per-model values with CIs: Table 3a below.\n")
    md.append("| Operator | Deciding cell | Direct error in cell | Cost vs. matching L1 (pp) | Composition / perception share | Under-applied | Over-applied |")
    md.append("|---|---|---|---|---|---|---|")
    cellname = {"AND": "(T, T): must keep", "AND-NOT": "base T, negated clause T: must drop",
                "EXCEPT": "base T, exception T: must drop", "EXCEPT, base clause": "base T, exception F: must keep",
                "L3 nested": "(T, F, T): must drop", "CLOSURE": "added via a same-kind object: must keep"}
    for label, tids, kind in OPS:
        def agg(key, pct=True, d=0):
            v = [t3[n][label][key][0] for n in runs if not math.isnan(t3[n][label][key][0])]
            if not v:
                return "—"
            m = 100 if pct else 1
            return f"{np.median(v) * m:.{d}f} [{min(v) * m:.{d}f} – {max(v) * m:.{d}f}]"
        cs =[t3[n][label]["comp"][0] for n in runs if not math.isnan(t3[n][label]["comp"][0])]
        cp = f"{np.median(cs) * 100:.0f}% / {100 - np.median(cs) * 100:.0f}%" if cs else "—"
        under = agg("err") if kind == "fire" else "—"
        over = agg("over") if kind == "fire" else (agg("err") if label == "EXCEPT, base clause" else "—")
        md.append(row_md([f"{label} ({', '.join(tids)})", cellname[label], agg("err") + "%", agg("cost", d=1),
                          cp, under + ("%" if under != "—" else ""), over + ("%" if over != "—" else "")]))
    md.append("\n### Table 3a. Per model: direct error in the deciding cell, % (CI) · paired cost vs. L1, pp (CI) · composition share, %\n")
    md.append("| Operator | " + " | ".join(NAMES[n] for n in runs) + " |")
    md.append("|---|" + "---|" * len(runs))
    for label, _, _ in OPS:
        cells = []
        for n in runs:
            t = t3[n][label]
            cells.append(f"{fmt(*t['err'], d=1, pct=True)} · {fmt(*t['cost'], d=1, pct=True)} · {fmt(t['comp'][0], d=0, pct=True)} (n={t['n']})")
        md.append(row_md([label] + cells))
    res["table3"] = {n: {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()} for k, v in t.items()}
                     for n, t in t3.items()}

    # ---- S1 composition
    md.append("\n## Table S1. Testbed composition (Test draft)\n")
    md.append("| Template | Level | Wording | Items | \\|S\\*\\| = 0 / 1 / 2 / 3 / 4 / 5+ | Opportunity cell present | Has a counter-heuristic candidate | Mean candidates | Val / train slice |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    bins = ["0", "1", "2", "3", "4", "5+"]
    sb = lambda k: "5+" if k >= 5 else str(k)
    for t in R.TEMPLATES + R.DIAG_TEMPLATES:
        its = [it for it in (test if t["id"] != "C01" else diag) if it["tid"] == t["id"] and (t["id"] != "C01" or it["role"] == "diag")]
        hist = collections.Counter(sb(it["n"]) for it in its)
        opp = f"{np.mean([it['fire'] > 0 for it in its]):.0%}" if (t["neg"] or t["exc"] or t.get("closure")) else "—"
        md.append(row_md([f"{t['id']} {t['name']}", t["level"], t["wording"], str(len(its)), " / ".join(str(hist[b]) for b in bins),
                          opp, f"{np.mean([bool(it['ch']) for it in its]):.0%}", f"{np.mean([it['clutter'] for it in its]):.1f}",
                          f"{sum(it['split'] == 'val' for it in its)} / {sum(it['split'] == 'train' for it in its)}"]))

    # ---- S2 robustness
    md.append("\n## Table S2. Robustness of the headline (ΔG with CI · clutter slope of the per-item gap per +4 candidates · top model)\n")
    md.append("| Variant | Items | " + " | ".join(NAMES[n] for n in runs) + " | τ (replica, set) | Top model |")
    md.append("|---|---|" + "---|" * len(runs) + "---|---|")

    def slope(rows):
        rr = [r for r in rows if r["it"]["level"] != "L1"]
        if not rr:
            return (math.nan,) * 3
        x = np.array([r["it"]["clutter"] / 4 for r in rr])
        y = np.array([r["g"] for r in rr])
        ix = boot.idx([r["img"] for r in rr])

        def st(w):
            ww = w[ix]
            if ww.sum() == 0:
                return math.nan
            mx, my = (ww * x).sum() / ww.sum(), (ww * y).sum() / ww.sum()
            vx = (ww * (x - mx) ** 2).sum()
            return (ww * (x - mx) * (y - my)).sum() / vx if vx else math.nan
        return boot.ci(st)

    def variant(label, rows_by_model, n_items):
        cells, d23 = [], {}
        for n in runs:
            rows = rows_by_model[n]
            g = dG(rows)
            s = slope(rows)
            d23[n] = np.mean([r["fd"] for r in rows if r["it"]["level"] != "L1"])
            cells.append(f"{fmt(*g)} · {fmt(s[0])}")
        tau_v = kendall([t2[n]["rep"][0] for n in runs], [d23[n] for n in runs])
        top = max(runs, key=lambda n: d23[n])
        md.append(row_md([label, str(n_items)] + cells + [f"{tau_v:.2f}", NAMES[top]]))
        return {n: dG(rows_by_model[n]) for n in runs}

    res["S2"] = {}
    res["S2"]["main"] = variant("**Main** (Table 2)", {n: scores[n]["contour"] for n in runs}, len(test))
    for label, margins in (("Tight margins (size 1.3×, borders ≤ 1 px, position ±1%)", (1.3, 1.0, 0.01)),
                           ("Loose margins (size 2.0×, borders ≤ 3 px, position ±4%)", (2.0, 3.0, 0.04))):
        new = {}
        for it in test:
            nit, keep = regt(it, margins)
            if keep:
                new[it["item_id"]] = nit
        rows = {}
        for n in runs:
            rows[n] = []
            for r in scores[n]["contour"]:
                nit = new.get(r["it"]["item_id"])
                if nit:
                    d = runs[n].d("direct", "contour", r["it"]) or set()
                    e = exec_ids(nit, runs[n].p("probe", "contour", nit["group"]))
                    fd, fe = f1(d, nit), f1(e, nit)
                    rows[n].append(dict(it=nit, img=r["img"], fd=fd, fe=fe, g=fe - fd))
        res["S2"][label] = variant(label, rows, len(new))
    keep20 = {it["item_id"] for it in test if len(it["U"]) <= 0.2 * it["clutter"]}
    res["S2"]["U20"] = variant("Ignore-band limit 20% (main: 30%)", {n: [r for r in scores[n]["contour"] if r["it"]["item_id"] in keep20] for n in runs}, len(keep20))
    md.append(row_md(["Ignore-band limit 40%", str(len(test))] + ["= main (Test was sampled at ≤ 30%)"] * len(runs) + ["", ""]))
    md.append(row_md(["GT-audit-clean subset", "—"] + ["pending human audit"] * len(runs) + ["", ""]))
    md.append(row_md(["Without cropped or fragmented subjects", "—"] + ["pending audit flag"] * len(runs) + ["", ""]))
    res["S2"]["text"] = variant("M-text instead of M-contour", {n: scores[n]["text"] for n in runs}, len(test))
    cells = []
    res["S2"]["retest_floor"] = {}
    for n, r in runs.items():
        fl = []
        for it in test:
            if it["item_id"] in lite_ids:
                a1, a2 = r.p("probe_retest", "contour", it["group"], "s1"), r.p("probe_retest", "contour", it["group"], "s2")
                if a1 is not None and a2 is not None:
                    fl.append((img(it), abs(f1(exec_ids(it, a1), it) - f1(exec_ids(it, a2), it))))
        rows = scores[n]["contour"]
        l23 = [(x["img"], x["g"]) for x in rows if x["it"]["level"] != "L1"]
        v = ci_diff(boot, l23, fl)
        res["S2"]["retest_floor"][n] = v
        cells.append(f"{fmt(*v)} (floor {np.mean([x[1] for x in fl]):.3f})")
    md.append(row_md(["Floor from probe re-run (two sampled draws) instead of L1 rules", str(len([i for i in test if i['item_id'] in lite_ids]))] + cells + ["", ""]))

    # ---- S3 wording / definition
    md.append("\n## Table S3. Wording and definition ablations\n")
    md.append("| Block | Contrast (same candidates) | n | Metric | " + " | ".join(NAMES[n] for n in runs) + " |")
    md.append("|---|---|---|---|" + "---|" * len(runs))
    res["S3"] = {}

    def under_rate(it, pred):
        t, a = R.TEMPLATE_BY_ID[it["tid"]], anchor_of(it)
        cell = [c for c in scored(it) if t.get("fire") and t["fire"](a["preds"][str(c)])]
        return [float(c in pred) for c in cell]

    tw_pairs = [(g["T12"], g["T11"]) for g in by_group.values() if "T12" in g and "T11" in g] + \
               [(g["T13"], g["T08"]) for g in by_group.values() if "T13" in g and "T08" in g]
    cf, cu = [], []
    for n, r in runs.items():
        df_ = [(img(x), f1(r.d("direct", "contour", x) or set(), x) - f1(r.d("direct", "contour", y) or set(), y)) for x, y in tw_pairs]
        du = []
        for x, y in tw_pairs:
            ux, uy = under_rate(x, r.d("direct", "contour", x) or set()), under_rate(y, r.d("direct", "contour", y) or set())
            if ux:
                du.append((img(x), float(np.mean(ux)) - float(np.mean(uy))))
        res["S3"].setdefault("twins", {})[n] = dict(dF1=ci_mean(boot, df_), dUnder=ci_mean(boot, du))
        cf.append(fmt(*ci_mean(boot, df_)))
        cu.append(fmt(*ci_mean(boot, du), d=1, pct=True) + " pp")
    md.append(row_md(["(a) Operator wording", "EXCEPT − AND-NOT twins (T12 vs. T11, T13 vs. T08; same truth set, same anchor)", str(len(tw_pairs)), "ΔF1"] + cf))
    md.append(row_md(["", "", "", "Δ under-applied rate"] + cu))
    neg_items = [it for it in test if it["item_id"] in neg_ids]
    cf, cu = [], []
    for n, r in runs.items():
        df_, du = [], []
        for it in neg_items:
            pos = r.d("direct_pos", "contour", it)
            neg = r.d("direct", "contour", it) if it["para"] == 0 else r.d("direct_neg0", "contour", it)
            if pos is None or neg is None:
                continue
            df_.append((img(it), f1(pos, it) - f1(neg, it)))
            up, un = under_rate(it, pos), under_rate(it, neg)
            if up:
                du.append((img(it), float(np.mean(up)) - float(np.mean(un))))
        res["S3"].setdefault("negation", {})[n] = dict(dF1=ci_mean(boot, df_), dUnder=ci_mean(boot, du), n=len(df_))
        cf.append(fmt(*ci_mean(boot, df_)))
        cu.append(fmt(*ci_mean(boot, du), d=1, pct=True) + " pp")
    md.append(row_md(["(b) Negation phrasing", "positive (\"a different kind\") − negated (\"not the same kind\"), Lite, paraphrase 0", str(len(neg_items)), "ΔF1"] + cf))
    md.append(row_md(["", "", "", "Δ under-applied rate"] + cu))
    # (c) definition uptake
    c01 = [it for it in diag if it["tid"] == "C01"]
    t01_of = {}
    for it in diag:
        if it["tid"] == "T01":
            t01_of[it["group"]] = it
    for it in test:
        if it["tid"] == "T01" and it["group"] in {c["group"] for c in c01}:
            t01_of.setdefault(it["group"], it)
    for mod in ("contour", "text"):
        r1, r2 = [], []
        for n, r in runs.items():
            agree_t, agree_c, incl = [], [], []
            for c in c01:
                t = t01_of.get(c["group"])
                if not t:
                    continue
                dc = r.d("direct_diag", mod, c)
                dt = r.d("direct_diag", mod, t)
                if dt is None and mod == "contour":
                    dt = r.d("direct", "contour", t)
                if dc is None or dt is None:
                    continue
                dec = [x for x in anchor_of(c)["cands"] if x not in c["U"]]
                sub = dict(c, U=[x for x in anchor_of(c)["cands"] if x not in dec])
                agree_t.append((img(c), f1(dt, sub)))
                agree_c.append((img(c), f1(dc, sub)))
                for x in t["S"]:
                    if x in c["U"]:
                        incl.append((img(c), float(x in dt) - float(x in dc)))
            res["S3"].setdefault(f"uptake_{mod}", {})[n] = dict(f1_borders=ci_mean(boot, agree_t), f1_touching=ci_mean(boot, agree_c),
                                                                 uptake=ci_mean(boot, incl), n_cand=len(incl))
            r1.append(f"{fmt(ci_mean(boot, agree_t)[0], d=2)} / {fmt(ci_mean(boot, agree_c)[0], d=2)}")
            r2.append(fmt(*ci_mean(boot, incl), d=1, pct=True) + f" pp (n={len(incl)})")
        md.append(row_md([f"(c) Definition uptake, M-{mod}", "\"borders\" T01 vs. \"physically touching\" C01", str(len(c01)) + " anchors",
                          "F1 on candidates both definitions decide (T01 / C01)"] + r1))
        md.append(row_md(["", "", "", "Inclusion of outline-meeting, contact-not-stated candidates: T01 − C01"] + r2))

    # ---- S4 counter-heuristic
    md.append("\n## Table S4. Counter-heuristic candidates (Test, M-contour). Cells per model: probe acc. flagged / unflagged · direct acc. flagged / unflagged · Δ direct (pp, CI)\n")
    md.append("| Shortcut that fails | Primitive | " + " | ".join(NAMES[n] for n in runs) + " |")
    md.append("|---|---|" + "---|" * len(runs))
    res["S4"] = {}
    for label, p, tid in (("typical physical size", "larger", "T02"), ("bounding boxes overlap", "touch", "T01"),
                          ("box edge vs. centre", "left", "T03"), ("box edge vs. centre", "above", "T04")):
        cells = []
        for n, r in runs.items():
            pa, da = {True: [], False: []}, {True: [], False: []}
            for g, its in by_group.items():
                it0 = next(iter(its.values()))
                a, ans = anchor_of(it0), r.p("probe", "contour", g) or {}
                for c in a["cands"]:
                    q = a["preds"][str(c)]
                    if q[p] is None:
                        continue
                    fl = p in (a["ch"].get(str(c)) or [])
                    m = (ans.get(c) or {}).get(p)
                    pa[fl].append((img(it0), float(m is not None and m == q[p])))
                u = its.get(tid)
                if u:
                    du = r.d("direct", "contour", u) or set()
                    for c in scored(u):
                        fl = p in (a["ch"].get(str(c)) or [])
                        da[fl].append((img(u), float((c in du) == (c in u["S"]))))
            dd = ci_diff(boot, da[True], da[False])
            res["S4"].setdefault(p, {})[n] = dict(probe=(np.mean([x[1] for x in pa[True]]) if pa[True] else math.nan,
                                                         np.mean([x[1] for x in pa[False]])),
                                                  direct=(np.mean([x[1] for x in da[True]]) if da[True] else math.nan,
                                                          np.mean([x[1] for x in da[False]]) if da[False] else math.nan),
                                                  delta=dd, n_flag=len(da[True]))
            pf = res["S4"][p][n]["probe"]
            dfl = res["S4"][p][n]["direct"]
            cells.append(f"{fmt(pf[0], d=2)} / {fmt(pf[1], d=2)} · {fmt(dfl[0], d=2)} / {fmt(dfl[1], d=2)} · {fmt(*dd, d=1, pct=True)} (n={len(da[True])})")
        md.append(row_md([label, p] + cells))

    # ---- S5 logistic model, S6 component difficulties
    import statsmodels.api as sm
    md.append("\n## Table S5. What makes a candidate hard. Odds ratios (CI), logistic GLM with image-clustered SEs; per model: direct · executed · direct×X interaction\n")
    feats = ["clutter_per4", "L2", "L3", "S_size", "log_min_area", "counter_heur", "text"]
    flab = {"clutter_per4": "+4 marked candidates", "L2": "Rule level L2 vs. L1", "L3": "Rule level L3 vs. L1", "S_size": "|S*| (+1)",
            "log_min_area": "Smallest-object area (log)", "counter_heur": "Counter-heuristic flag", "text": "M-text vs. M-contour"}
    s5 = {}
    for n, r in runs.items():
        X, y, D, grp = [], [], [], []
        for mod in ("contour", "text"):
            dk, pk = ("direct", "probe") if mod == "contour" else ("direct_text", "probe_text")
            for it in test:
                a = anchor_of(it)
                d = r.d(dk, mod, it) or set()
                e = exec_ids(it, r.p(pk, mod, it["group"]))
                areas = [a["frac"] * a["preds"][str(c)]["ratio"] for c in a["cands"]]
                base = [len(a["cands"]) / 4, it["level"] == "L2", it["level"] == "L3", len(it["S"]), math.log(min(areas)), 0, mod == "text"]
                for c in scored(it):
                    f = list(base)
                    f[5] = c in it["ch"]
                    must = c in it["S"]
                    for isd, pred in ((1, d), (0, e)):
                        X.append(f)
                        D.append(isd)
                        y.append(float((c in pred) == must))
                        grp.append(img(it))
        X = np.array(X, float)
        D = np.array(D, float)[:, None]
        Xs = np.hstack([np.ones((len(X), 1)), X, D, X * D])
        gi = np.unique(grp, return_inverse=True)[1]
        fit = sm.GLM(np.array(y), Xs, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": gi})
        b, V = fit.params, fit.cov_params()
        k = len(feats)
        s5[n] = {}
        for i, f in enumerate(feats):
            ie, idx_ = 1 + i, 1 + k + 1 + i
            be, se_e = b[ie], math.sqrt(V[ie, ie])
            bd = b[ie] + b[idx_]
            se_d = math.sqrt(V[ie, ie] + V[idx_, idx_] + 2 * V[ie, idx_])
            bi, se_i = b[idx_], math.sqrt(V[idx_, idx_])
            o = lambda m, s: (math.exp(m), math.exp(m - 1.96 * s), math.exp(m + 1.96 * s))
            s5[n][f] = dict(direct=o(bd, se_d), executed=o(be, se_e), interaction=o(bi, se_i))
    md.append("| Fixed effect | " + " | ".join(NAMES[n] for n in runs) + " |")
    md.append("|---|" + "---|" * len(runs))
    for f in feats:
        md.append(row_md([flab[f]] + [" · ".join(fmt(*s5[n][f][k], d=2) for k in ("direct", "executed", "interaction")) for n in runs]))
    res["S5"] = s5

    md.append("\n## Table S6. Component difficulties (logit; higher = harder), direct answers, M-contour; same kind = 0\n")
    comps = [c for c in R.COMPONENTS if c != "same_cat"]
    s6 = {}
    for n, r in runs.items():
        X, y, grp = [], [], []
        for it in test:
            t = R.TEMPLATE_BY_ID[it["tid"]]
            d = r.d("direct", "contour", it) or set()
            row = [1.0] + [float(c in t["components"]) for c in comps]
            for c in scored(it):
                X.append(row)
                y.append(float((c in d) == (c in it["S"])))
                grp.append(img(it))
        gi = np.unique(grp, return_inverse=True)[1]
        fit = sm.GLM(np.array(y), np.array(X), family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": gi})
        s6[n] = {c: (-fit.params[1 + i], -fit.params[1 + i] - 1.96 * fit.bse[1 + i], -fit.params[1 + i] + 1.96 * fit.bse[1 + i])
                 for i, c in enumerate(comps)}
    md.append("| Component | " + " | ".join(NAMES[n] for n in runs) + " |")
    md.append("|---|" + "---|" * len(runs))
    lab = {"touch": "borders (touch)", "size": "size", "left": "left", "above": "above", "border": "image edge", "AND": "AND",
           "NOT": "NOT", "EXCEPT": "EXCEPT (beyond NOT)", "CLOSURE": "CLOSURE"}
    md.append(row_md(["same kind (reference)"] + ["0"] * len(runs)))
    for c in comps:
        md.append(row_md([lab[c]] + [fmt(*s6[n][c], d=2) for n in runs]))
    res["S6"] = s6

    # ---- S7 probe validity
    md.append("\n## Table S7. Is the gap a probe artefact?\n")
    md.append("| Model | Format compliance, direct / probe | Probe test–retest agreement | Shared − rule-specific probe (Δ acc.) | larger/smaller self-contradiction | L1 floor G(L1) | Empty-item accuracy | False-empty rate on \\|S\\*\\| ≥ 1 |")
    md.append("|---|---|---|---|---|---|---|---|")
    res["S7"] = {}
    rule_items = [it for it in test if it["item_id"] in set(lite["rule_probe"])]
    for n, r in runs.items():
        comp_d = sum(r.parsed[k] for k in r.n if k.startswith("direct")) / sum(r.n[k] for k in r.n if k.startswith("direct"))
        comp_p = sum(r.parsed[k] for k in r.n if k.startswith("probe")) / sum(r.n[k] for k in r.n if k.startswith("probe"))
        agree = []
        for g in {it["group"] for it in test if it["item_id"] in lite_ids}:
            a1, a2 = r.p("probe_retest", "contour", g, "s1"), r.p("probe_retest", "contour", g, "s2")
            if a1 is None or a2 is None:
                continue
            for c, m1 in a1.items():
                m2 = a2.get(c) or {}
                for p in R.PROBE_PRIMS:
                    if m1.get(p) is not None and m2.get(p) is not None:
                        agree.append((g.rsplit("-", 1)[0], float(m1[p] == m2[p])))
        eq = []
        for it in rule_items:
            a = anchor_of(it)
            rs, sh = r.p("probe_rule", "contour", it["item_id"]) or {}, r.p("probe", "contour", it["group"]) or {}
            for c in a["cands"]:
                q = a["preds"][str(c)]
                for p in R.TEMPLATE_BY_ID[it["tid"]]["probe"]:
                    if p == "touches" or q.get(p) is None:
                        continue
                    ms, mr = (sh.get(c) or {}).get(p), (rs.get(c) or {}).get(p)
                    eq.append((img(it), float(ms == q[p]) - float(mr == q[p])))
        contra = []
        for g, its in by_group.items():
            it0 = next(iter(its.values()))
            a, ans = anchor_of(it0), r.p("probe", "contour", g) or {}
            for c in a["cands"]:
                if a["preds"][str(c)]["larger"] is None:
                    continue
                m = ans.get(c) or {}
                if m.get("larger") is not None and m.get("smaller") is not None:
                    contra.append((img(it0), float(m["larger"] == m["smaller"])))
        rows = scores[n]["contour"]
        floor = ci_mean(boot, [(x["img"], x["g"]) for x in rows if x["it"]["level"] == "L1"])
        emp, fe = [], []
        for x in rows:
            it = x["it"]
            d = (r.d("direct", "contour", it) or set()) - set(it["U"])
            if not it["S"]:
                emp.append((x["img"], float(not d)))
            else:
                fe.append((x["img"], float(not d)))
        res["S7"][n] = dict(compliance=(comp_d, comp_p), retest=float(np.mean([v for _, v in agree])) if agree else math.nan,
                            equiv=ci_mean(boot, eq), contradiction=ci_mean(boot, contra), floor=floor,
                            empty=ci_mean(boot, emp), false_empty=ci_mean(boot, fe))
        s = res["S7"][n]
        md.append(row_md([NAMES[n], f"{comp_d:.1%} / {comp_p:.1%}", f"{s['retest']:.1%}", fmt(*s["equiv"], d=1, pct=True) + " pp",
                          fmt(*s["contradiction"], d=1, pct=True) + "%", fmt(*s["floor"]), fmt(*s["empty"], d=1, pct=True) + "%",
                          fmt(*s["false_empty"], d=1, pct=True) + "%"]))

    # ---- S8 scale
    md.append("\n## Table S8. Does thinking or scale close the gap? (paired, same Test L2/L3 items, M-contour)\n")
    md.append("| Contrast | ΔF1 direct | ΔF1 executed | ΔG (per item) |")
    md.append("|---|---|---|---|")
    res["S8"] = {}
    think = Run("qwen3-vl-8b-thinking") if os.path.exists(os.path.join(ROOT, "data", "eval", "runs", "qwen3-vl-8b-thinking", "responses.jsonl")) else None
    lite_items = [it for it in test if it["item_id"] in lite_ids]

    def lite_rows(r):
        """Per-item scores on Lite, M-contour; items without a direct response are left out, not scored as []."""
        out = []
        for it in lite_items:
            d = r.d("direct", "contour", it)
            pa = r.p("probe", "contour", it["group"])
            if d is None or pa is None:
                continue
            fd, fe = f1(d, it), f1(exec_ids(it, pa), it)
            out.append(dict(it=it, img=img(it), fd=fd, fe=fe, g=fe - fd))
        return out

    if think and "qwen3-vl-8b-instruct" in runs:
        a8 = {x["it"]["item_id"]: x for x in lite_rows(runs["qwen3-vl-8b-instruct"]) if x["it"]["level"] != "L1"}
        at = {x["it"]["item_id"]: x for x in lite_rows(think) if x["it"]["level"] != "L1"}
        ks = sorted(set(a8) & set(at))
        cells = [fmt(*ci_mean(boot, [(at[k]["img"], at[k][f] - a8[k][f]) for k in ks])) for f in ("fd", "fe", "g")]
        res["S8"]["thinking-instruct"] = cells
        md.append(row_md([f"Qwen3-VL-8B thinking − no thinking (Lite, {len(ks)} L2/L3 items)"] + cells))
    else:
        md.append("| Qwen3-VL-8B thinking − no thinking | held (not run) | | |")
    if "qwen3-vl-8b-instruct" in runs and "qwen3-vl-4b-instruct" in runs:
        a8 = {x["it"]["item_id"]: x for x in scores["qwen3-vl-8b-instruct"]["contour"] if x["it"]["level"] != "L1"}
        a4 = {x["it"]["item_id"]: x for x in scores["qwen3-vl-4b-instruct"]["contour"] if x["it"]["level"] != "L1"}
        ks = sorted(set(a8) & set(a4))
        cells = [fmt(*ci_mean(boot, [(a8[k]["img"], a8[k][f] - a4[k][f]) for k in ks])) for f in ("fd", "fe", "g")]
        res["S8"]["8B-4B"] = cells
        md.append(row_md(["Qwen3-VL 8B − 4B"] + cells))
    md.append("| Qwen3-VL 32B − 8B | not run (bf16 32B ran out of GPU memory while quantizing on load) | | |")

    # ---- S9 pointing
    md.append("\n## Table S9. How you point at the subject (Lite L2/L3 items, direct F1)\n")
    md.append("| Model | M-text | M-contour | M-mark | Alt. marker | Max − min | Within-family scale gap (8B − 4B, M-contour) | Subject found in M-text |")
    md.append("|---|---|---|---|---|---|---|---|")
    lite23 = [it for it in test if it["item_id"] in lite_ids and it["level"] != "L1"]
    res["S9"] = {}
    sg = None
    if "qwen3-vl-8b-instruct" in runs and "qwen3-vl-4b-instruct" in runs:
        r8, r4 = runs["qwen3-vl-8b-instruct"], runs["qwen3-vl-4b-instruct"]
        sg = np.mean([f1(r8.d("direct", "contour", it) or set(), it) - f1(r4.d("direct", "contour", it) or set(), it) for it in lite23])
    for n, r in runs.items():
        v = {}
        for lab_, kind, mod in (("M-text", "direct_text", "text"), ("M-contour", "direct", "contour"),
                                ("M-mark", "direct_mark", "mark"), ("Alt. marker", "direct_alt", "contour")):
            v[lab_] = ci_mean(boot, [(img(it), f1(r.d(kind, mod, it) or set(), it)) for it in lite23])
        ests = [x[0] for x in v.values()]
        res["S9"][n] = v
        md.append(row_md([NAMES[n]] + [fmt(*v[k], d=2) for k in v] + [f"{max(ests) - min(ests):.3f}",
                          f"{sg:.3f}" if sg is not None and n.startswith("qwen") else "—", "not run (no grounding probe)"]))

    # ---- Lite panel: every model on the same Lite items (the only Test subset the thinking model ran)
    if think:
        md.append("\n## Table 2-Lite. Pairwise accuracy versus set selection on Lite (1,497 Test items, M-contour), incl. the thinking model\n")
        md.append("| Model | Replica acc. | Direct F1 L1 | L2 | L3 | Executed F1 L2/L3 | ΔG = G(L2/L3) − G(L1) | Items scored | Truncated (direct / probe) |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        res["lite"] = {}
        allr = dict(runs, **{"qwen3-vl-8b-thinking": think})
        for n, r in allr.items():
            rows = lite_rows(r)
            rep = [(img(j), float(a == j["answer"])) for j, a in r.replica.values()]
            g = dG(rows)
            tr = (r.trunc["direct"] / max(r.n["direct"], 1), r.trunc["probe"] / max(r.n["probe"], 1))
            res["lite"][n] = dict(rep=ci_mean(boot, rep), dG=g, n=len(rows), trunc=tr)
            md.append(row_md([NAMES[n], fmt(*ci_mean(boot, rep), d=2)] +
                             [fmt(np.mean([x["fd"] for x in rows if x["it"]["level"] == L]), d=2) for L in ("L1", "L2", "L3")] +
                             [fmt(np.mean([x["fe"] for x in rows if x["it"]["level"] != "L1"]), d=2), "**" + fmt(*g) + "**",
                              str(len(rows)), f"{tr[0]:.1%} / {tr[1]:.1%}"]))
        md.append("\nThinking run: the model's own sampling (T 1.0, top-p 0.95, top-k 20, seed 0), 16k-token budget; an answer whose reasoning is cut off counts as `[]` / no probe answer.\n")

    os.makedirs(OUT, exist_ok=True)
    head = (f"# ANCHOR results ({', '.join(NAMES[n] for n in runs)})\n\nGenerated by tools/eval/analyze.py. "
            f"Test = {len(test)} items on {len(by_group)} anchors; 95% CIs from an image-clustered bootstrap, B = {B}, "
            f"over {len(boot.ids)} images.\n")
    open(os.path.join(OUT, "results.md"), "w").write(head + "\n" + "\n".join(md) + "\n")
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    print(head + "\n" + "\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1:] or list(NAMES))
