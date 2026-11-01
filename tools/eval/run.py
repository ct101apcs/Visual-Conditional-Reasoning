"""Run data/eval/jobs.jsonl through one model with vLLM. Resumable: finished job ids are skipped.

  ~/.venvs/anchor-vllm/bin/python tools/eval/run.py --model Qwen/Qwen3-VL-8B-Instruct --rows 1,2,3
  ~/.venvs/anchor-vllm/bin/python tools/eval/run.py --model Qwen/Qwen3-VL-8B-Instruct --smoke 24
  --model also takes a local directory; the run name is then its last path part, lower case.

Writes data/eval/runs/<name>/{responses.jsonl, run.log, progress.json} (--smoke: runs/<name>-smoke/).
Each log line: rows, done/total, calls/s, output tok/s, ETA, and the chunk's truncation and parse-failure counts.
"""
import argparse
import collections
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

CHUNK = 256
# --thinking: the model's own sampling (Qwen3-VL-Thinking generation_config; greedy decoding makes thinking
# models loop), a fixed seed per job, and token budgets that leave room for the reasoning.
THINK_SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20)
THINK_MAX_TOKENS = {"direct": 16384, "probe": 16384, "replica": 4096}   # smoke 2026-10-30: at 8k/12k, 12-62% truncated


def log(fh, msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def smoke_pick(jobs, n, seed=0):
    """n jobs per kind, spread over templates / primitives / modalities so rare formats show up."""
    rng = random.Random(seed)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for j in jobs:
        by[j["kind"]][(j.get("tid"), j.get("prim"), j["modality"], j.get("touches"))].append(j)
    out = []
    for kind, strata in by.items():
        keys = sorted(strata, key=str)
        for v in strata.values():
            rng.shuffle(v)
        i, got = 0, []
        while len(got) < n and any(i < len(strata[k]) for k in keys):
            got += [strata[k][i] for k in keys if i < len(strata[k])][: n - len(got)]
            i += 1
        out += got
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", help="run directory name (default: last part of --model, lower case)")
    ap.add_argument("--rows", default="1,2,3")
    ap.add_argument("--smoke", type=int, default=0, help="only N jobs per kind, into runs/<name>-smoke")
    ap.add_argument("--gpu-mem", type=float, default=0.75)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--quantization", default=None, help='e.g. "fp8": quantize a bf16 checkpoint while loading')
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--thinking", action="store_true", help="reasoning model: see THINK_SAMPLING / THINK_MAX_TOKENS")
    ap.add_argument("--kinds", help="comma-separated job kinds to keep, e.g. direct,probe,replica")
    ap.add_argument("--lite", action="store_true", help="keep only jobs on Lite items / Lite anchors")
    ap.add_argument("--chunk", type=int, default=CHUNK, help="jobs per generate call (larger keeps long outputs batched)")
    a = ap.parse_args()

    rows = {int(x) for x in a.rows.split(",")}
    name = (a.name or a.model.rstrip("/").split("/")[-1].lower()) + ("-smoke" if a.smoke else "")
    out = os.path.join(ROOT, "data", "eval", "runs", name)
    os.makedirs(out, exist_ok=True)
    jobs = [j for j in map(json.loads, open(os.path.join(ROOT, "data", "eval", "jobs.jsonl"))) if j["row"] in rows]
    if a.kinds:
        jobs = [j for j in jobs if j["kind"] in a.kinds.split(",")]
    if a.lite:
        lite = json.load(open(os.path.join(ROOT, "data", "anchor_gt", "lite.json")))
        ids = set(lite["items"])
        groups = {i.split("-", 1)[1] for i in ids}
        jobs = [j for j in jobs if j.get("item_id") in ids or j.get("group") in groups]
    if a.smoke:
        jobs = smoke_pick(jobs, a.smoke)
    resp_path = os.path.join(out, "responses.jsonl")
    done = set()
    if os.path.exists(resp_path):
        done = {json.loads(l)["job_id"] for l in open(resp_path)}
    todo = sorted((j for j in jobs if j["job_id"] not in done), key=lambda j: (j["row"], j["kind"], j["job_id"]))
    lf = open(os.path.join(out, "run.log"), "a")
    log(lf, f"start model={a.model} rows={sorted(rows)} jobs={len(jobs)} done={len(done)} todo={len(todo)} out={out}")
    if not todo:
        return

    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    import score

    proc = AutoProcessor.from_pretrained(a.model)
    llm = LLM(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
              limit_mm_per_prompt={"image": 1}, seed=0, quantization=a.quantization, kv_cache_dtype=a.kv_cache_dtype)
    log(lf, "model loaded")

    def sampling(j):
        if not a.thinking:
            return j["sampling"]
        base = next(k for k in THINK_MAX_TOKENS if j["kind"].startswith(k))
        return dict(j["sampling"], **THINK_SAMPLING, max_tokens=THINK_MAX_TOKENS[base], seed=j["sampling"].get("seed", 0))

    def prompt(text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    t0, n_done, tok_out = time.time(), 0, 0
    per_kind = collections.Counter(j["kind"] for j in todo)
    with open(resp_path, "a") as rf:
        for s in range(0, len(todo), a.chunk):
            chunk = todo[s:s + a.chunk]
            inputs = [dict(prompt=prompt(j["prompt"]),
                           multi_modal_data={"image": Image.open(os.path.join(ROOT, j["image"])).convert("RGB")})
                      for j in chunk]
            params = [SamplingParams(**sampling(j)) for j in chunk]
            tc = time.time()
            res = llm.generate(inputs, params, use_tqdm=False)
            trunc = fail = 0
            for j, r in zip(chunk, res):
                o = r.outputs[0]
                rec = dict(job_id=j["job_id"], output=o.text, n_in=len(r.prompt_token_ids), n_out=len(o.token_ids),
                           finish_reason=o.finish_reason)
                rf.write(json.dumps(rec) + "\n")
                tok_out += rec["n_out"]
                trunc += o.finish_reason == "length"
                cands = score.render.anchor(j["split"], j["key"], j["anchor"])["cands"]
                if j["kind"].startswith("direct"):
                    fail += not score.parse_direct(score.answer_text(o.text, a.thinking), cands, j["anchor"])["ok"]
                elif j["kind"].startswith("probe"):
                    fail += not score.parse_probe(score.answer_text(o.text, a.thinking), cands, [])["ok"]
                else:
                    fail += not score.parse_replica(score.answer_text(o.text, a.thinking))["ok"]
            rf.flush()
            n_done += len(chunk)
            el = time.time() - t0
            rate = n_done / el
            kinds = ",".join(f"{k}:{v}" for k, v in collections.Counter(j["kind"] for j in chunk).items())
            log(lf, f"row {chunk[-1]['row']} {len(done) + n_done}/{len(jobs)} [{kinds}] chunk {time.time() - tc:.0f}s "
                    f"| {rate:.2f} calls/s {tok_out / el:.0f} out-tok/s | ETA {(len(todo) - n_done) / rate / 60:.0f} min "
                    f"| truncated {trunc} parse-fail {fail}")
            json.dump(dict(model=a.model, done=len(done) + n_done, total=len(jobs), calls_per_s=round(rate, 3),
                           eta_min=round((len(todo) - n_done) / rate / 60, 1), todo_by_kind=per_kind,
                           updated=time.strftime("%Y-%m-%d %H:%M:%S")),
                      open(os.path.join(out, "progress.json"), "w"), indent=1)
    log(lf, f"finished {len(todo)} jobs in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
