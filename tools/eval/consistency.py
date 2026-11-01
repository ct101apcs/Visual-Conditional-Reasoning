"""Relation–set consistency: the headline tables of paper_scaffold.md v1.0 (Tables 2, 3 and S-C1 to S-C3).

  python3 tools/eval/consistency.py qwen3-vl-4b-instruct qwen3-vl-8b-instruct internvl3-8b-hf gemma-3-4b-it qwen3-vl-8b-thinking
  -> data/eval/results/consistency.md and consistency.json

Reads finished runs only; analyze.py and results.md are untouched (see tools/eval/MIGRATION.md).

For every scored candidate of an item, two answers of the same model each give an in/out decision.
  agreement       share of candidates where the two decisions match
  independent     the agreement expected if the two answers erred independently, given their accuracies
  error coupling  phi between the two answers' correctness: 1 = the same errors, 0 = independent errors
Because the ground truth is fixed per candidate, two decisions agree exactly when both are right or both wrong,
so agreement and coupling come from one 2x2 table.

Three pairs, from most to least alike in how the question is asked:
  retest     the shared probe drawn twice (row 2, sampled), each executed under the item's rule
  reworded   direct answers to the EXCEPT / AND-NOT twins on one anchor (T12~T11, T13~T08; same truth set)
  relations  the direct answer vs. the item's rule executed on the model's own probe answers
Primary set: Lite (1,497 Test items), where every model, the thinking model included, has every pair.
CIs: the image-clustered bootstrap of analyze.py (same seed, B = 1000).
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze as A  # noqa: E402

R = A.R
OUT = A.OUT
TWINS = (("T12", "T11"), ("T13", "T08"))
# paired model contrasts (same Lite candidates, same bootstrap draws): (label, minuend, subtrahend)
CONTRASTS = (("thinking − no thinking (Qwen3-VL-8B)", "qwen3-vl-8b-thinking", "qwen3-vl-8b-instruct"),
             ("8B − 4B (Qwen3-VL, no thinking)", "qwen3-vl-8b-instruct", "qwen3-vl-4b-instruct"))


class Pair:
    """Per-candidate correctness of two answers, kept per image for the clustered bootstrap."""

    def __init__(self, boot):
        self.boot = boot
        self.C = np.zeros((len(boot.ids), 4))   # per image: both right, a only, b only, both wrong

    def add(self, image, a_ok, b_ok):
        self.C[self.boot.ix[image], (0 if a_ok else 2) + (0 if b_ok else 1)] += 1

    @property
    def n(self):
        return int(self.C.sum())

    def _stats(self, cnt):
        """cnt: (..., 4) -> dict of arrays."""
        a, b, c, d = (cnt[..., k] for k in range(4))
        N = a + b + c + d
        with np.errstate(invalid="ignore", divide="ignore"):
            pa, pb = (a + b) / N, (a + c) / N
            phi = (a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
            return dict(agree=(a + d) / N, indep=pa * pb + (1 - pa) * (1 - pb), phi=phi, acc_a=pa, acc_b=pb)

    def ci(self, other=None):
        """(est, lo, hi) per statistic; with `other`, also the paired difference self - other."""
        est = self._stats(self.C.sum(0))
        draws = self._stats(self.boot.W @ self.C)
        out = {}
        for k in est:
            dr = draws[k][np.isfinite(draws[k])]
            lo, hi = np.percentile(dr, [2.5, 97.5]) if len(dr) else (math.nan, math.nan)
            out[k] = (float(est[k]), float(lo), float(hi))
        if other is not None:
            e2, d2 = other._stats(other.C.sum(0)), other._stats(other.boot.W @ other.C)
            for k in ("agree", "phi"):
                dd = draws[k] - d2[k]
                dd = dd[np.isfinite(dd)]
                lo, hi = np.percentile(dd, [2.5, 97.5]) if len(dd) else (math.nan, math.nan)
                out["d_" + k] = (float(est[k] - e2[k]), float(lo), float(hi))
        return out


def rate(boot, rows):
    return A.ci_mean(boot, rows)


def main(names):
    test = A.jl(os.path.join(A.GT, "test_draft.jsonl"))
    diag = A.jl(os.path.join(A.GT, "test_diag.jsonl"))
    lite_ids = set(json.load(open(os.path.join(A.GT, "lite.json")))["items"])
    boot = A.Boot([A.img(it) for it in test + diag])
    by_group = {}
    for it in test:
        by_group.setdefault(it["group"], {})[it["tid"]] = it
    res, md = {}, []
    kept = {}   # model -> Lite Pair objects, for the paired contrasts

    for n in names:
        r = A.Run(n)
        sets = {"lite": lambda it: it["item_id"] in lite_ids}
        # the full-Test block only for runs that answered (nearly) every Test item, e.g. not Lite-only runs
        if sum(r.d("direct", "contour", it) is not None for it in test) >= 0.99 * len(test):
            sets["test"] = lambda it: True
        res[n] = {}
        for sname, keep in sets.items():
            P = {k: Pair(boot) for k in ("retest", "reworded", "relations", "relations_L1",
                                         "relations_in", "relations_out")}
            cells = {lab: Pair(boot) for lab, _, _ in A.OPS}
            exc = {"under": {}, "over": {}}
            for lab, _, _ in A.OPS:
                exc["under"][lab], exc["over"][lab] = [], []
            for it in test:
                if not keep(it):
                    continue
                im = A.img(it)
                d = r.d("direct", "contour", it)
                if d is None:
                    continue
                pr = r.p("probe", "contour", it["group"])
                e = A.exec_ids(it, pr) if pr is not None else None
                cands = A.scored(it)
                if e is not None:
                    for c in cands:
                        m = c in it["S"]
                        a_ok, b_ok = (c in d) == m, (c in e) == m
                        if it["level"] == "L1":
                            P["relations_L1"].add(im, a_ok, b_ok)
                        else:
                            P["relations"].add(im, a_ok, b_ok)
                            P["relations_in" if m else "relations_out"].add(im, a_ok, b_ok)
                # retest: the two sampled probe draws, each executed under this item's rule
                if it["level"] != "L1":
                    p1, p2 = (r.p("probe_retest", "contour", it["group"], s) for s in ("s1", "s2"))
                    if p1 is not None and p2 is not None:
                        e1, e2 = A.exec_ids(it, p1), A.exec_ids(it, p2)
                        for c in cands:
                            m = c in it["S"]
                            P["retest"].add(im, (c in e1) == m, (c in e2) == m)
                # reworded: EXCEPT twin vs. its AND-NOT twin on the same anchor
                for x, y in TWINS:
                    if it["tid"] == x and y in by_group[it["group"]]:
                        iy = by_group[it["group"]][y]
                        dy = r.d("direct", "contour", iy)
                        if dy is not None:
                            for c in cands:
                                if c not in iy["U"]:
                                    m = c in it["S"]
                                    P["reworded"].add(im, (c in d) == m, (c in dy) == m)
                # operator cells (Table 3 deciding cells) and exception direction
                t = R.TEMPLATE_BY_ID[it["tid"]]
                a = A.anchor_of(it)
                for lab, tids, kind in A.OPS:
                    if it["tid"] not in tids:
                        continue
                    for c in cands:
                        q = a["preds"][str(c)]
                        if kind == "keep":
                            inc = c in it["S"]
                        elif kind == "fire":
                            inc = bool(t["fire"](q))
                        else:
                            inc = c in it["S"] and not q["same_cat"]
                        if not inc:
                            continue
                        m = c in it["S"]
                        if e is not None:
                            cells[lab].add(im, (c in d) == m, (c in e) == m)
                        if kind == "fire":
                            exc["under"][lab].append((im, float(c in d)))
                    if kind == "fire":
                        for c in it["S"]:
                            exc["over"][lab].append((im, float(c not in d)))
            out = {k: dict(p.ci(), n=p.n) for k, p in P.items() if p.n}
            if P["retest"].n and P["relations"].n:
                out["retest_minus_relations"] = P["retest"].ci(P["relations"])
            if P["relations_L1"].n and P["relations"].n:
                out["L1_minus_relations"] = P["relations_L1"].ci(P["relations"])
            if P["reworded"].n and P["relations"].n:
                out["reworded_minus_relations"] = P["reworded"].ci(P["relations"])
            out["cells"] = {lab: dict(p.ci(), n=p.n) for lab, p in cells.items() if p.n}
            out["exceptions"] = {kk: {lab: rate(boot, rows) for lab, rows in v.items() if rows}
                                 for kk, v in exc.items()}
            res[n][sname] = out
            if sname == "lite":
                kept[n] = dict(P, **{"cell:" + k: v for k, v in cells.items()})
        print(f"done {n}", flush=True)

    res["contrasts"] = {}
    for lab, a, b in CONTRASTS:
        if a in kept and b in kept:
            res["contrasts"][lab] = {k: kept[a][k].ci(kept[b][k]) for k in kept[a]
                                     if kept[a][k].n and kept[b][k].n}

    # ---------------------------------------------------------------- markdown
    def f(t, d=2, pct=False):
        return A.fmt(*t, d=d, pct=pct)

    def ap(s):
        return f"{f(s['agree'])} · φ {f(s['phi'])}"

    md.append("# Relation–set consistency (tools/eval/consistency.py)\n")
    md.append("Each cell: agreement (CI) · error coupling φ (CI). Agreement = share of scored candidates on which the "
              "two answers make the same in/out decision. φ = correlation of the two answers' correctness "
              "(1 = same errors, 0 = independent errors). M-contour, L2/L3 items unless noted. "
              "95% CIs: image-clustered bootstrap, B = 1000.\n")
    md.append("## Table 2. Three ways to ask the same question (Lite, 1,497 Test items)\n")
    md.append("| Model | Same probe, two draws | Reworded rule (twins) | Set vs. own relations | … expected if independent | Set vs. own relations, L1 rules | Δφ retest − relations | Δφ reworded − relations | Δφ L1 − L2/L3 |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for n in names:
        s = res[n]["lite"]
        cell = lambda k: ap(s[k]) if k in s else "not run"  # noqa: E731
        ind = f(s["relations"]["indep"]) if "relations" in s else "—"
        dd = {k: f(s[k]["d_phi"]) if k in s else "—"
              for k in ("retest_minus_relations", "reworded_minus_relations", "L1_minus_relations")}
        md.append(f"| {A.NAMES.get(n, n)} | {cell('retest')} | {cell('reworded')} | {cell('relations')} | {ind} | "
                  f"{cell('relations_L1')} | {dd['retest_minus_relations']} | {dd['reworded_minus_relations']} | "
                  f"{dd['L1_minus_relations']} |")
    md.append("\nRetest draws are sampled (T 0.7) where direct and probe answers are greedy, so the retest column is a "
              "conservative ceiling. The thinking model has no retest row yet (row 2 not run).\n")

    md.append("## Table S-C1. The same pairs on the full Test set (7,000 items; instruct models)\n")
    md.append("| Model | n cand. | Reworded rule (twins) | Set vs. own relations | … expected if independent | Set vs. own relations, L1 | φ within S* / within non-S* | Accuracy direct / executed |")
    md.append("|---|---|---|---|---|---|---|---|")
    for n in names:
        if "test" not in res[n]:
            continue
        s = res[n]["test"]
        md.append(f"| {A.NAMES.get(n, n)} | {s['relations']['n']} | {ap(s['reworded'])} | {ap(s['relations'])} | "
                  f"{f(s['relations']['indep'])} | {ap(s['relations_L1'])} | "
                  f"{f(s['relations_in']['phi'])} / {f(s['relations_out']['phi'])} | "
                  f"{f(s['relations']['acc_a'])} / {f(s['relations']['acc_b'])} |")
    md.append("\nφ within S* / non-S*: the coupling computed separately on candidates that must be kept and on those "
              "that must be dropped, so a shared yes/no bias cannot create it.\n")

    md.append("## Table S-C2. Error coupling of set vs. own relations, by operator cell (φ, CI · n)\n")
    labs = [lab for lab, _, _ in A.OPS]
    for sname, title in (("test", "Test, instruct models"), ("lite", "Lite, all models")):
        md.append(f"\n**{title}**\n")
        md.append("| Operator cell | " + " | ".join(A.NAMES.get(n, n) for n in names if sname in res[n]) + " |")
        md.append("|---|" + "---|" * sum(sname in res[n] for n in names))
        for lab in labs:
            row = [lab]
            for n in names:
                if sname not in res[n]:
                    continue
                c = res[n][sname]["cells"].get(lab)
                row.append(f"{f(c['phi'])} · {c['n']}" if c else "—")
            md.append("| " + " | ".join(row) + " |")

    md.append("\n## Table S-C3. Which way exceptions fail (direct answers, M-contour, %)\n")
    md.append("Under-applied: P(kept | the NOT/EXCEPT clause excludes it). Over-applied: P(dropped | in S*), "
              "over the same templates (definitions as Table 3 of results.md).\n")
    for sname, title in (("lite", "Lite, all models"), ("test", "Test, instruct models")):
        md.append(f"\n**{title}**\n")
        md.append("| Operator | Rate | " + " | ".join(A.NAMES.get(n, n) for n in names if sname in res[n]) + " |")
        md.append("|---|---|" + "---|" * sum(sname in res[n] for n in names))
        for lab, _, kind in A.OPS:
            if kind != "fire":
                continue
            for kk, nm in (("under", "under-applied"), ("over", "over-applied")):
                row = [lab, nm]
                for n in names:
                    if sname not in res[n]:
                        continue
                    v = res[n][sname]["exceptions"][kk].get(lab)
                    row.append(f(v, d=0, pct=True) if v else "—")
                md.append("| " + " | ".join(row) + " |")

    md.append("\n## Table S-C4. Paired model contrasts on Lite (Δ agreement · Δφ, CI)\n")
    md.append("Same Lite candidates and the same bootstrap draws for both models.\n")
    rows = [("relations", "Set vs. own relations, L2/L3"), ("relations_L1", "Set vs. own relations, L1"),
            ("reworded", "Reworded rule (twins)")] + [("cell:" + lab, "cell: " + lab) for lab, _, _ in A.OPS]
    labs_c = list(res["contrasts"])
    if labs_c:
        md.append("| Pair | " + " | ".join(labs_c) + " |")
        md.append("|---|" + "---|" * len(labs_c))
        for k, nm in rows:
            cellv = []
            for lab in labs_c:
                v = res["contrasts"][lab].get(k)
                cellv.append(f"{f(v['d_agree'])} · {f(v['d_phi'])}" if v else "—")
            md.append(f"| {nm} | " + " | ".join(cellv) + " |")

    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "consistency.md"), "w").write("\n".join(md) + "\n")
    json.dump(res, open(os.path.join(OUT, "consistency.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "consistency.md"))


if __name__ == "__main__":
    main(sys.argv[1:] or ["qwen3-vl-4b-instruct", "qwen3-vl-8b-instruct", "internvl3-8b-hf", "gemma-3-4b-it",
                          "qwen3-vl-8b-thinking"])
