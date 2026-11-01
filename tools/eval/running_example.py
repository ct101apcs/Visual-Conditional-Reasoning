"""Measured outputs for the paper's running example (Fig. 1): image 000000534270, subject #15.

  python3 tools/eval/running_example.py qwen3-vl-4b-instruct qwen3-vl-8b-instruct internvl3-8b-hf gemma-3-4b-it

Prints, per model and M-contour: the set answer, the executed set (rule run on the model's own relation
answers) and their F1 for T12 "bordering, except larger", its AND-NOT twin T11, and T14; then the model's
`touch` / `larger` relation answers next to the ground truth. The example items are not in Lite, so Lite-only
runs (the thinking model) print None.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze as A  # noqa: E402

IDS = ["T12-val-000000534270-15", "T11-val-000000534270-15", "T14-val-000000534270-15"]


def main(names):
    test = {it["item_id"]: it for it in A.jl(os.path.join(A.GT, "test_draft.jsonl"))}
    a = A.anchor_of(test[IDS[0]])
    gt = {c: (a["preds"][str(c)]["touch"], a["preds"][str(c)]["larger"]) for c in a["cands"]}
    print("candidates", a["cands"])
    print("ground truth (borders, larger):", gt)
    for n in names:
        r = A.Run(n)
        p = r.p("probe", "contour", test[IDS[0]]["group"])
        print(f"\n{n}")
        for i in IDS:
            it = test[i]
            d = r.d("direct", "contour", it)
            e = A.exec_ids(it, p) if p is not None else None
            fd = f"{A.f1(d, it):.2f}" if d is not None else "—"
            fe = f"{A.f1(e, it):.2f}" if e is not None else "—"
            print(f"  {it['tid']} S*={sorted(it['S'])}  set answer={sorted(d) if d is not None else None} (F1 {fd})"
                  f"  executed={sorted(e) if e is not None else None} (F1 {fe})")
        if p:
            ans = {c: (p.get(c, {}).get("touch"), p.get(c, {}).get("larger")) for c in a["cands"]}
            wrong = [c for c in a["cands"] if ans[c] != gt[c]]
            print(f"  relation answers (borders, larger): {ans}  wrong on: {wrong}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["qwen3-vl-4b-instruct", "qwen3-vl-8b-instruct", "internvl3-8b-hf", "gemma-3-4b-it"])
