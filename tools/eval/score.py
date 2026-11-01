"""Parse model outputs and score them against the generator's GT.

  python tools/eval/score.py data/eval/runs/<model>           # per-kind parse health + F1 summary

Parsing is lenient on form and strict on content: a JSON list / object anywhere in the text is
accepted (a free-form draft before it is allowed, design §4.3), "#3" and "3" both name mark 3, and
anything that is not a marked candidate is dropped and counted.
"""
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools", "anchor_gt"))
sys.path.insert(0, HERE)
import rules as R  # noqa: E402
import render  # noqa: E402

KEY2PRIM = {v: k for k, v in R.PROBE_KEY.items()}
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
LIST_RE = re.compile(r"\[[^\[\]{}]*\]")
NUM_RE = re.compile(r"#?\s*(\d+)")


def answer_text(text, thinking=False):
    """The part of an output a parser may read. Thinking models write reasoning, then </think>, then the
    answer (their template already opened <think>); reasoning cut off before </think> has no answer."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return "" if thinking else text


def _ids(s):
    return [int(x) for x in NUM_RE.findall(s)]


def parse_direct(text, cands, subject):
    t = THINK_RE.sub("", text)
    lists = LIST_RE.findall(t)
    if not lists:
        return dict(ok=False, why="no list")
    raw = _ids(lists[-1])                       # the last list is the answer; earlier ones may be drafting
    cs = set(cands)
    return dict(ok=True, ids=sorted({i for i in raw if i in cs}), subject=subject in raw,
                invalid=sorted({i for i in raw if i not in cs and i != subject}), n_lists=len(lists))


def _bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "yes"):
        return True
    if isinstance(v, str) and v.strip().lower() in ("false", "no"):
        return False
    return None


def parse_probe(text, cands, prims, touches=False):
    """{cand: {prim: bool|None}} plus `touches` lists. prims are internal names (rules.PROBE_PRIMS)."""
    t = THINK_RE.sub("", text)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return dict(ok=False, why="no object")
    s = t[i:j + 1]
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        try:
            obj = json.loads(re.sub(r",\s*([}\]])", r"\1", s.replace("True", "true").replace("False", "false")))
        except json.JSONDecodeError:
            return dict(ok=False, why="bad json", truncated=not t.rstrip().endswith("}"))
    if not isinstance(obj, dict):
        return dict(ok=False, why="not an object")
    cs, ans, extra, bad_field = set(cands), {}, [], 0
    for k, v in obj.items():
        ids = _ids(str(k))
        if len(ids) != 1 or ids[0] not in cs or not isinstance(v, dict):
            extra.append(k)
            continue
        row = {}
        for fk, fv in v.items():
            p = KEY2PRIM.get(fk.strip().lower())
            if p is None:
                bad_field += 1
            elif p == "touches":
                row[p] = [x for x in (_ids(json.dumps(fv)) if fv is not None else []) if x in cs]
            else:
                row[p] = _bool(fv)
        ans[ids[0]] = row
    missing = [c for c in cands if c not in ans]
    miss_field = sum(1 for c in ans for p in prims if ans[c].get(p) is None)
    return dict(ok=True, ans=ans, missing=missing, extra=extra, bad_field=bad_field, miss_field=miss_field)


def parse_replica(text):
    t = THINK_RE.sub("", text).strip()
    m = re.search(r"\b([AB])\b", t) or re.search(r"\b(yes|no)\b", t, re.I)
    if not m:
        return dict(ok=False, why="no letter")
    return dict(ok=True, answer=m.group(1).upper() in ("A", "YES"))


def set_f1(pred, S, U):
    """Item F1 over scored candidates (U excluded); empty prediction for empty GT scores 1."""
    pred, S = set(pred) - set(U), set(S)
    if not pred and not S:
        return 1.0
    tp = len(pred & S)
    return 0.0 if tp == 0 else 2 * tp / (len(pred) + len(S))


def execute(tid, probe_ans, cands):
    """Run the rule program on a model's probe answers -> selected ids (None answers select nothing)."""
    p = {c: {k: probe_ans.get(c, {}).get(k) for k in R.PROBE_PRIMS} for c in cands}
    cc = {}
    for c in cands:
        for x in probe_ans.get(c, {}).get("touches") or []:
            if x != c:
                cc[(min(c, x), max(c, x))] = True
    vals = R.TEMPLATE_BY_ID[tid]["fn"](p, cands, cc)
    return sorted(c for c in cands if vals[c] is True)


def load(run_dir):
    jobs = {j["job_id"]: j for j in map(json.loads, open(os.path.join(ROOT, "data", "eval", "jobs.jsonl")))}
    resp = {}
    p = os.path.join(run_dir, "responses.jsonl")
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            resp[r["job_id"]] = r
    return jobs, resp


def items_by_id():
    out = {}
    for f in ("test_draft.jsonl", "test_diag.jsonl"):
        for l in open(os.path.join(ROOT, "data", "anchor_gt", f)):
            it = json.loads(l)
            out[it["item_id"]] = it
    return out


def report(run_dir):
    jobs, resp = load(run_dir)
    thinking = "thinking" in os.path.basename(run_dir.rstrip("/"))
    items = items_by_id()
    health = collections.defaultdict(collections.Counter)
    f1 = collections.defaultdict(list)
    probes = {}                                   # group -> parsed greedy probe (row 1)
    for jid, r in resp.items():
        j = jobs[jid]
        a = render.anchor(j["split"], j["key"], j["anchor"])
        h = health[j["kind"]]
        h["n"] += 1
        h["truncated"] += r.get("finish_reason") == "length"
        h["out_tok"] += r.get("n_out", 0)
        if j["kind"].startswith("direct"):
            d = parse_direct(answer_text(r["output"], thinking), a["cands"], a["id"])
            h["parsed"] += d["ok"]
            if d["ok"]:
                h["with_subject"] += d["subject"]
                h["with_invalid_id"] += bool(d["invalid"])
                h["multi_list"] += d["n_lists"] > 1
                it = items[j["item_id"]]
                f1[(j["kind"], j["modality"], it["level"])].append(set_f1(d["ids"], it["S"], it["U"]))
        elif j["kind"].startswith("probe"):
            prims = R.TEMPLATE_BY_ID[j["tid"]]["probe"] if j.get("tid") else R.PROBE_PRIMS
            d = parse_probe(answer_text(r["output"], thinking), a["cands"], prims, j.get("touches"))
            h["parsed"] += d["ok"]
            if d["ok"]:
                h["missing_cand"] += bool(d["missing"])
                h["missing_field"] += bool(d["miss_field"])
                h["unknown_field"] += bool(d["bad_field"])
                if j["kind"] == "probe":
                    probes[j["group"]] = d["ans"]
                    for c in a["cands"]:
                        q, m = a["preds"][str(c)], d["ans"].get(c, {})
                        for p in R.PROBE_PRIMS:
                            if q[p] is not None and m.get(p) is not None:
                                f1[("probe_acc", p)].append(float(q[p] == m[p]))
                        if q["larger"] is not None and m.get("larger") is not None and m.get("smaller") is not None:
                            f1[("probe_larger_smaller_contradiction",)].append(float(m["larger"] == m["smaller"]))
        elif j["kind"] == "replica":
            d = parse_replica(answer_text(r["output"], thinking))
            h["parsed"] += d["ok"]
            if d["ok"]:
                f1[("replica_acc", j["prim"])].append(float(d["answer"] == j["answer"]))
    # executed F1 for direct Test items whose anchor probe is in
    for jid, r in resp.items():
        j = jobs[jid]
        if j["kind"] == "direct" and j["item_id"] in items:
            it = items[j["item_id"]]
            if it["group"] in probes:
                a = render.anchor(j["split"], j["key"], j["anchor"])
                f1[("executed", "contour", it["level"])].append(
                    set_f1(execute(it["tid"], probes[it["group"]], a["cands"]), it["S"], it["U"]))
    return health, f1


def main(run_dir):
    health, f1 = report(run_dir)
    print("parse health (share of responses)")
    for k, h in sorted(health.items()):
        n = h["n"]
        cols = " ".join(f"{c}={h[c] / n:.1%}" for c in sorted(h) if c not in ("n", "out_tok"))
        print(f"  {k:14s} n={n:5d} mean_out_tok={h['out_tok'] / n:6.0f} {cols}")
    print("scores (mean)")
    for k in sorted(f1, key=str):
        v = f1[k]
        print(f"  {' / '.join(map(str, k)):32s} n={len(v):5d} {sum(v) / len(v):.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
