# ANCHOR model runs

Executes the model calls behind the paper's result tables ([paper_scaffold.md](../../paper_scaffold.md), "Supplementary tables").

```bash
python3 tools/eval/build_jobs.py                       # Lite + jobs.jsonl + renders (model-independent, ~15 s)
VLLM=~/.venvs/anchor-vllm/bin/python                   # vLLM 0.11.0 lives in its own venv
$VLLM tools/eval/run.py --model Qwen/Qwen3-VL-8B-Instruct --smoke 24      # N jobs per kind -> runs/<name>-smoke
$VLLM tools/eval/run.py --model Qwen/Qwen3-VL-8B-Instruct --rows 1,2,3    # resumable
python3 tools/eval/score.py data/eval/runs/qwen3-vl-8b-instruct           # parse health + scores
tail -f data/eval/runs/qwen3-vl-8b-instruct/run.log                       # track
```

| file | role |
|---|---|
| `render.py` | model input image: candidates outlined yellow with a black number box at their mark; M-contour adds the subject's magenta outline (no number); M-text leaves the subject unmarked |
| `build_jobs.py` | samples **Lite** (1.5k Test items, whole anchors, each template at its Test share) into `data/anchor_gt/lite.json`, picks the negation, rule-probe and replica subsets, writes `data/eval/jobs.jsonl`, renders `data/eval/renders/` |
| `run.py` | vLLM batch runner; appends `responses.jsonl`, logs `run.log`, writes `progress.json` |
| `score.py` | lenient parsers (`parse_direct`, `parse_probe`, `parse_replica`), item set F1 with U excluded, `execute` (rule program on a model's probe answers) |
| `analyze.py`, `consistency.py`, `running_example.py` | result tables (`data/eval/results/results.md`, `consistency.md`) and the Fig. 1 numbers; see [MIGRATION.md](MIGRATION.md) |
| `pack_outputs.sh`, `slim_records.py` | release bundle: jobs, run outputs and the GT records of the Test images, enough to recompute every table without images or the full GT build |

Jobs by result row:

| Row | Tables | Kind | Jobs | Notes |
|---|---|---|---|---|
| 1 | 2, 3, S4–S6 | `direct` | 7,000 | Test, M-contour, the item's paraphrase |
| | | `probe` | 3,120 | shared all-primitive probe per Test anchor; `touches` on CLOSURE anchors |
| | | `replica` | 1,500 | Lite, one yes/no question per (candidate, primitive), 125 yes + 125 no per primitive |
| 2 | S7 | `probe_retest` | 1,398 | Lite anchors × 2 sampled draws (T 0.7, top-p 0.8, top-k 20, seeds 1–2) |
| | | `probe_rule` | 500 | rule-specific probe (`render_probe`) on Lite items, stratified by template |
| 3 | S3 | `direct_pos` | 500 | NOT templates restated positively (`rules.POSITIVE_TEXT`, paraphrase 0) |
| | | `direct_neg0` | 331 | the negated paraphrase 0 where the Test run used another paraphrase |
| | | `direct_diag` | 1,140 | caption-contact slice: C01 + paired T01 in M-text and M-contour, plus 60 Test T01 items in M-text |
| 4 | S2 | `direct_text`, `probe_text` | 7,000 + 3,120 | the whole Test set in M-text; margin, ignore-band and clean-subset variants are re-scored offline |
| 5 | S9 | `direct_mark`, `probe_mark` | 1,497 + 699 | Lite in M-mark (the subject is numbered like every other object) |
| | | `direct_alt`, `probe_alt` | 1,497 + 699 | Lite in M-contour with the alternative marker style (`render.py style="alt"`: Set-of-Mark-like colour fills and coloured number tags, subject outline unchanged) |

All jobs use greedy decoding unless noted. Max output tokens: direct 1,024, probe 4,096, replica 16.

## The queue (`queue.sh`)

Runs every non-thinking model one at a time, from checkpoints already on disk (nothing is downloaded):

| Order | Run name | Checkpoint | Notes |
|---|---|---|---|
| 1 | `qwen3-vl-8b-instruct` | HF cache | rows 1–3 (launched first), then rows 4–5 |
| 2 | `qwen3-vl-4b-instruct` | `~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-4B-Instruct` | rows 1–5 |
| 3 | `qwen3-vl-32b-instruct` | `~/SAIH-SADL/Qwen3-VL-32B-Instruct` (bf16, 63 GB) | quantized to FP8 while loading, FP8 KV cache. If loading runs out of GPU memory, it is skipped; the fallback is the official FP8 checkpoint (35.5 GB download, not automatic) |
| 4 | `internvl3-8b-hf` | `~/.cache/modelscope/hub/models/OpenGVLab/InternVL3-8B-hf` | rows 1–5 |

Each new model first runs a 24-jobs-per-kind smoke test over rows 1–5; `check_smoke.py` fails it on < 95% parsed, > 2% truncated or > 5% probes missing a candidate, and the queue then skips that model. Held for later: Qwen3-VL-8B-Thinking (`~/SAIH-SADL/Qwen3-VL-8B-Thinking`).

`data/eval/logs/queue.log` has one line per step (START / END with exit code / SMOKE result / SKIP).

## Tracking a run

- `data/eval/runs/<name>/run.log`: one line per 256-job chunk (row, done/total, calls/s, output tok/s, ETA, truncated and parse-fail counts in that chunk). `tail -f` it.
- `data/eval/runs/<name>/progress.json`: the same as machine-readable state.
- `data/eval/runs/<name>/stdout.log`: vLLM's own output; look here when the run stops.
- `python3 tools/eval/score.py data/eval/runs/<name>`: works mid-run on whatever has finished.
- Restarting the same command resumes; finished job ids are skipped.

## Smoke test, 2026-10-29 (Qwen3-VL-8B-Instruct, 24 jobs per kind)

Runs: `runs/qwen3-vl-8b-instruct-smoke-v1` (before fixes) and `runs/qwen3-vl-8b-instruct-smoke` (after).

| Finding | Effect | Decision |
|---|---|---|
| Probes skipped visible marks: ≥ 1 candidate missing in 21–25% of probes (e.g. a probe listing #6–#12 while #4 and #5 were clearly drawn) | A skipped candidate counts as "not selected" when the probe is executed, so executed F1 drops for a reason unrelated to composition, and the gap shrinks | Every prompt now lists the mark numbers (`rules.objects_line`): "The marked objects are #4, #5, …". Direct and probe get the same information. Missing candidates: 0% after the fix |
| M-text read "the a person in blue jeans": 473 of 3,252 subject phrases carry their own determiner (a 408, another 43, an 12, one 5, the 3, this 2) | Ungrammatical subject reference in M-text | `rules.subject_ref` drops a leading determiner before adding "the". "another woman wearing a maroon top" becomes "the woman wearing a maroon top", which is still unique, since uniqueness was checked on the full phrase |
| `larger`/`smaller` answered "both false" on 48% of decided candidates | Not a bug: the model under-calls `smaller` (accuracy 0.56 vs. 0.88 for `larger`) | Kept. This is the S7 self-contradiction diagnostic; `score.py` now reports it as a rate |
| Format | 192/192 parsed, 0 truncated, no subject id in any answer, 0–4% answers with an id that is not a mark | No change. Mean output is 6–17 tokens for direct answers and about 440 for the shared probe |

Open choices made without a smoke signal, recorded here for review:
- The primary runs use greedy decoding. The S7 re-run uses two *sampled* draws (Qwen's own generation config), because a greedy re-run would only repeat itself.
- Negation minimal pairs use paraphrase 0 on both sides (`direct_pos` vs. `direct_neg0`, plus the Test run where its paraphrase is already 0).
- Images go in at native COCO resolution (≈ 300 visual tokens); mark numbers are legible at that size in the smoke renders.
