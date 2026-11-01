"""SAM 3 Agent baseline (Table 2 row): the official agent (facebookresearch/sam3, sam3/agent) on Test items.

  # 1. driver VLM, OpenAI-compatible (anchor-vllm venv)
  ~/.venvs/anchor-vllm/bin/vllm serve <Qwen3-VL-8B-Instruct dir> --served-model-name qwen3-vl-8b-instruct \
      --port 8001 --max-model-len 32768 --gpu-memory-utilization 0.7 --limit-mm-per-prompt '{"image": 3}' --seed 0
  # 2. agent (sam3 venv: ~/sam3_official installed with pip install -e)
  ~/.venvs/sam3/bin/python tools/eval/sam3_agent.py --smoke 20      # -> runs/sam3-agent-qwen3-vl-8b-smoke
  ~/.venvs/sam3/bin/python tools/eval/sam3_agent.py                 # all 7,000 Test items; resumable

Protocol (choices recorded in tools/eval/README.md):
- Input: the raw COCO image (no marks) and the item's M-text rule sentence, with "marked object(s)" -> "object(s)"
  and the operational note on "borders" kept. The subject is named by its caption phrase, as in M-text.
- The agent is used unchanged: its system prompts, tool loop and the server's default sampling.
- Output -> answer set: a candidate is selected when one of the agent's final masks has IoU >= 0.5 with the
  candidate's ground-truth mask. Masks on unmarked objects or the subject are ignored. Every final mask's
  best-IoU candidate is stored too, so other thresholds can be re-scored offline.
- The driver sometimes answers without a parseable <tool> call (mostly on round 1: it restates the system prompt's
  instructions), and the official loop then raises. Such an item is re-run from scratch, up to --attempts times
  (fresh sample; the agent itself is unchanged). An item still failing, or hitting max_generations, is scored as [].
- The official example drives the agent with Qwen3-VL-8B-Thinking; this row uses the Instruct model at its
  generation_config defaults, matching the instruct rows of Table 2.
"""
import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools", "pancap_viewer"))
from pancap import rgb2id  # noqa: E402
sys.path.insert(0, HERE)
import render  # noqa: E402

DATA = os.path.join(ROOT, "data")
IOU_MIN = 0.5


def rule_query(prompt):
    """The M-text direct prompt's rule sentence(s), rewritten for an image without marks."""
    m = re.search(r"Rule: (.*?)\nAnswer with", prompt, re.S)
    q = m.group(1).strip()
    q = re.sub(r"\bmarked objects\b", "objects", q)
    q = re.sub(r"\bmarked object\b", "object", q)
    q = re.sub(r"\bThe marked\b", "The", q)
    subj = re.search(r"The subject is (.*?)\. It is not marked\.", prompt).group(1)
    return q + f" Do not include {subj} itself."


def load_items(lite=False):
    items = {json.loads(l)["item_id"]: json.loads(l) for l in open(os.path.join(DATA, "anchor_gt", "test_draft.jsonl"))}
    if lite:
        ids = set(json.load(open(os.path.join(DATA, "anchor_gt", "lite.json")))["items"])
        items = {k: v for k, v in items.items() if k in ids}
    q = {}
    for l in open(os.path.join(DATA, "eval", "jobs.jsonl")):
        j = json.loads(l)
        if j["kind"] == "direct_text" and j["item_id"] in items:
            q[j["item_id"]] = rule_query(j["prompt"])
    return [dict(items[k], query=q[k]) for k in sorted(items) if k in q]


def decode(counts, h, w):
    from pycocotools import mask as mu
    return mu.decode({"size": [h, w], "counts": counts.encode() if isinstance(counts, str) else counts}).astype(bool)


def match(final, it):
    """Best-IoU candidate for every final mask; candidates with IoU >= IOU_MIN are selected."""
    seg = rgb2id(np.array(Image.open(os.path.join(DATA, "masks", "png", f"{it['split']}2017", it["key"] + ".png")).convert("RGB")))
    cands = render.anchor(it["split"], it["key"], it["anchor"])["cands"]
    gt = {c: seg == c for c in cands + [it["anchor"]]}
    h, w = seg.shape
    out = []
    for counts in final.get("pred_masks", []):
        m = decode(counts, final.get("orig_img_h", h), final.get("orig_img_w", w))
        if m.shape != (h, w):
            m = np.array(Image.fromarray(m).resize((w, h), Image.NEAREST))
        ious = {c: float((m & g).sum() / max((m | g).sum(), 1)) for c, g in gt.items()}
        best = max(ious, key=ious.get) if ious else None
        out.append(dict(best=best, iou=round(ious[best], 4) if best is not None else 0.0))
    sel = sorted({o["best"] for o in out if o["iou"] >= IOU_MIN and o["best"] != it["anchor"]})
    return sel, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default="qwen3-vl-8b-instruct", help="served model name")
    ap.add_argument("--name", default="sam3-agent-qwen3-vl-8b")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--smoke", type=int, default=0, help="first N items per template, into runs/<name>-smoke")
    ap.add_argument("--lite", action="store_true")
    ap.add_argument("--keep-files", action="store_true", help="keep per-item SAM 3 renders (default: smoke only)")
    ap.add_argument("--max-generations", type=int, default=100)
    ap.add_argument("--attempts", type=int, default=3, help="re-run an item whose agent run fails on a malformed tool call")
    a = ap.parse_args()

    import torch
    from sam3 import build_sam3_image_model
    from sam3.agent.agent_core import agent_inference
    from sam3.agent.client_llm import send_generate_request
    from sam3.agent.client_sam3 import call_sam_service
    from sam3.model.sam3_image_processor import Sam3Processor

    name = a.name + ("-smoke" if a.smoke else "")
    out = os.path.join(DATA, "eval", "runs", name)
    os.makedirs(out, exist_ok=True)
    items = load_items(a.lite)
    if a.smoke:
        per = {}
        items = [it for it in items if per.setdefault(it["tid"], []).append(it) or len(per[it["tid"]]) <= a.smoke]
    resp_path = os.path.join(out, "responses.jsonl")
    done = {json.loads(l)["item_id"] for l in open(resp_path)} if os.path.exists(resp_path) else set()
    todo = [it for it in items if it["item_id"] not in done]
    lf = open(os.path.join(out, "run.log"), "a")

    def log(msg):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
        print(line, file=sys.stderr, flush=True)
        lf.write(line + "\n")
        lf.flush()

    log(f"start server={a.server} model={a.model} items={len(items)} done={len(done)} todo={len(todo)} out={out}")
    if not todo:
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_sam3_image_model()
    proc = Sam3Processor(model, confidence_threshold=0.5)
    sam_lock = threading.Lock()
    log("SAM 3 loaded")

    def sam(**kw):
        with sam_lock, torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
            return call_sam_service(proc, **kw)

    llm = partial(send_generate_request, server_url=a.server, model=a.model, api_key="EMPTY")
    n_llm = threading.local()

    def counted_llm(messages):
        n_llm.k += 1
        return llm(messages)

    def one(it):
        idir = os.path.join(out, "items", it["item_id"])
        img = os.path.join(DATA, "images", f"{it['split']}2017", it["key"] + ".jpg")
        n_llm.k = 0
        t0 = time.time()
        rec = dict(item_id=it["item_id"], tid=it["tid"], query=it["query"])
        failed = []
        for attempt in range(1, a.attempts + 1):
            shutil.rmtree(idir, ignore_errors=True)
            try:
                hist, final, _ = agent_inference(img, it["query"], send_generate_request=counted_llm, call_sam_service=sam,
                                                 max_generations=a.max_generations, output_dir=idir)
                sel, masks = match(final, it)
                last = next((c["text"] for m in reversed(hist) if m["role"] == "assistant" for c in m["content"]
                             if isinstance(c, dict) and c.get("type") == "text"), "")
                rec.update(ok=True, answer=sel, masks=masks, last_tool=last[-400:])
                break
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:300]}"
                rec.update(ok=False, answer=[], masks=[], error=err, trace=traceback.format_exc()[-800:])
                failed.append(err[:120])
                if "Exceeded maximum number" in err:   # a 100-call loop is an answer-less run, not a format slip
                    break
        rec.update(attempts=attempt, failed_attempts=failed, n_llm=n_llm.k, sec=round(time.time() - t0, 1))
        if not (a.keep_files or a.smoke):
            shutil.rmtree(idir, ignore_errors=True)
        return rec

    sys.stdout = open(os.devnull, "w")   # the agent prints every round from every thread; the log goes to stderr
    t0, n, errs, lock = time.time(), 0, 0, threading.Lock()
    with open(resp_path, "a") as rf, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, it) for it in todo]
        for f in as_completed(futs):
            rec = f.result()
            with lock:
                rf.write(json.dumps(rec) + "\n")
                rf.flush()
                n += 1
                errs += not rec["ok"]
                if n % 50 == 0 or n == len(todo):
                    el = time.time() - t0
                    log(f"{len(done) + n}/{len(items)} | {n / el:.2f} items/s | ETA {(len(todo) - n) / (n / el) / 60:.0f} min "
                        f"| errors {errs} ({errs / n:.1%})")
    log(f"finished {n} items in {(time.time() - t0) / 60:.1f} min, errors {errs}")


if __name__ == "__main__":
    main()
