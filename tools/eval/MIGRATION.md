# From the v0.2 analysis to the v1.0 paper (relation–set consistency)

Written 2026-11-01, while the SAM 3 Agent run (`runs/sam3-agent-qwen3-vl-8b`) was still in progress. Nothing that run uses was changed:
- `analyze.py`, `run.py`, `score.py`, `render.py`, `build_jobs.py`, `sam3_agent.py` and `queue.sh` are untouched;
- `results.md` / `results.json` are untouched.

The new analysis lives in two new scripts that only read the runs:

| Script | Output | Role |
|---|---|---|
| `consistency.py` | `data/eval/results/consistency.md`, `consistency.json` | Tables 2 and 3 (part), Fig. 2, and S2–S5 of [paper_scaffold.md](../../paper_scaffold.md) v1.0 |
| `running_example.py` | stdout | the measured outputs behind Fig. 1 |

The scaffold this maps onto is [paper_scaffold.md](../../paper_scaffold.md). The old one is [archive/paper_scaffold_v0.2_selection-gap.md](../../archive/paper_scaffold_v0.2_selection-gap.md).

## 1. What changed in the story, in one paragraph

v0.2 measured one number per item: executed-set F1 minus set-answer F1 (the "selection gap", ΔG). It expected that number to be positive (H1). It is negative for every instruct model. v1.0 compares the same two answers **per object**, not per item:
- how often they make the same in/out decision (**agreement**), against what independent errors would give (**independence level**);
- whether they make the same errors (**error coupling φ**).

It sets these against two reference pairs: the same relation call drawn twice (*retest*), and the same rule worded two ways (*reworded*, the EXCEPT/AND-NOT twins). ΔG survives as a secondary result (Table 4).

The old Table 3 column "composition / perception share" is **retired**. It is P(executed right | set answer wrong). That equals the relation call's accuracy whenever the two answers err independently, which is what they do. So the column did not show "the model knew the relation but did not apply it".

## 2. Table map, old → new

| v0.2 (analyze.py / results.md) | v1.0 scaffold | Source after migration | What changes |
|---|---|---|---|
| Table 2 "Pairwise accuracy vs. set selection" | **Table 4** "Set accuracy, and what decomposing the question buys" | analyze.py Table 2 + Table 2-Lite | The rank and τ columns go. An M-text set-F1 column is added for the SAM 3 Agent comparison (§3). The ΔG column stays and is labelled as the pre-registered H1 |
| Table 3 "Where selections go wrong, by operator" | **Table 3** (exception rows) + **S5** + **S3** | consistency.py S-C3 (under/over, Lite incl. thinking); S-C2 (φ per cell) | The composition/perception column is dropped (§1). "Cost vs. matching L1" moves to S12's caption or is dropped. Direct error in cell = the under-applied rate for "fire" cells, so it is not repeated |
| — | **Table 2** "Three ways to ask the same question" | consistency.py Table 2 | New |
| — | **Fig. 2** φ by operator cell | consistency.py S-C2 (Lite) + S-C4 | New; replaces old Fig. 3 (rank plot) and old Fig. 2 (clutter marginal effects) |
| S1 Testbed composition | S1 | analyze.py S1 | unchanged |
| S2 Robustness of the headline (ΔG under variants) | **S8** | analyze.py S2 | Add φ (set vs. own relations) under each variant (§4.2). ΔG stays |
| S3 Wording and definition ablations | **S6**; (a) and (c) rows also feed **Table 3** | analyze.py S3 | unchanged. The thinking model needs result row 3 |
| S4 Counter-heuristic candidates | **S9** | analyze.py S4 | unchanged |
| S5 What makes a candidate hard (GLM) | **S11** (demoted) | analyze.py S5 | unchanged |
| S6 Component difficulties (LLTM) | **S12** (demoted) | analyze.py S6 | unchanged |
| S7 Is the gap a probe artefact? | **S7** "Is the mismatch a relation-call artefact?" | analyze.py S7 | unchanged. It now supports C1/C2: shared vs. rule-specific call, format, retest |
| S8 Does thinking or scale close the gap? | **S4** | consistency.py S-C4 (Δagreement, Δφ) + analyze.py S8 (ΔF1, ΔG) | Merge the two into one table: one row per contrast, with columns ΔF1 direct · ΔF1 executed · Δφ overall · Δφ AND-NOT · Δφ EXCEPT |
| S9 How you point at the subject | **S10** | analyze.py S9 | unchanged; one sentence in §3.4 |
| — | **S2** Table 2 on the full Test set, plus φ within S\* / non-S\* | consistency.py S-C1 | New |
| — | **S3** φ by operator cell, Test and Lite | consistency.py S-C2 | New |
| — | **S5** exception direction, Lite and Test | consistency.py S-C3 | New (Test values equal analyze.py Table 3a's "direct error in cell" for the fire cells, and its over-applied column) |

Terminology in column headers and captions:

| v0.2 | v1.0 |
|---|---|
| object selection, selection ability | set answer, set F1 |
| probe answers, pairwise answers | relation answers (the shared relation call) |
| executed-probe set | executed set |
| selection gap G, ΔG | decomposition gain ΔG (secondary) |
| distractor | not used |

## 3. Adding the SAM 3 Agent row when the run finishes

The run writes `runs/sam3-agent-qwen3-vl-8b/responses.jsonl`, one line per Test item: `item_id`, `tid`, `query`, `ok`, `answer` (selected mark numbers, IoU ≥ 0.5), `masks` (`best`, `iou` per final mask) and `last_tool`. It answers from the **raw image plus the rule in words**, so it is an M-text condition.

Steps:
1. **Wait for the run to end** before reading the file. A line being written can be cut off, and `json.loads` then fails on it. If you must read early, skip a last line that does not parse.
2. **Set F1 per item:** `score.set_f1(set(r["answer"]), it["S"], it["U"])`. Items with `ok == false`, or with no line at all, score as `[]`, as the agent's docstring says. Report that rate: 8.7% on 2026-11-01.
3. **Aggregate per level** (L1 / L2 / L3) with the image-clustered CI, using `analyze.ci_mean(boot, [(img(it), f1), …])`.
4. **Put it in Table 4's SAM 3 row.** The comparison column is the VLMs' **M-text** set F1 on L2/L3. `analyze.item_scores(run, "text")` already computes it per item; it is printed only inside S2 today. Add a per-level M-text block to the Table 4 code (analyze.py, the `t2` block around line 240), and replace the placeholder row at line 244.
5. **IoU sensitivity (supplement footnote):** re-score at IoU 0.3 / 0.5 / 0.7 from `masks[*].best` and `masks[*].iou`, with no new calls.
6. **The §5.5 sentence:** report F1 on T05 (same kind, which SAM 3's concept search covers) against T12 (bordering, except larger).
7. **Consistency is not defined** for the agent, because it gives no relation answers. Show "—" in Table 2.

A minimal loader to add to `analyze.py` once the run has ended:

```python
def sam3_sets(path=os.path.join(ROOT, "data", "eval", "runs", "sam3-agent-qwen3-vl-8b", "responses.jsonl")):
    """item_id -> selected marks; failed items count as []."""
    out = {}
    for l in open(path):
        r = json.loads(l)
        out[r["item_id"]] = set(r["answer"]) if r.get("ok") else set()
    return out
```

## 4. Changes still to make in `analyze.py` (after the SAM 3 run, not before)

1. **Table 3:** delete the `comp` list (lines 311–322) and the "Composition / perception share" column (lines 339, 351, 364). If a reviewer asks about it, report P(executed right | set answer wrong) next to P(executed right): consistency.py's S-C2 gives the 2×2 behind both.
2. **S2 → S8:** in `variant()` (line 405), also compute φ (set vs. own relations) under each margin variant. `regt()` already recomputes S and U, so call the `Pair` logic of consistency.py on the re-labelled items. A sign or level change under tight or loose margins must be reported.
3. **Table 2 → Table 4:**
   - drop the rank column and the τ rows;
   - add the M-text per-level block (§3);
   - keep the "not run" placeholders for the class-list and segmenter rows.
4. **S8 → S4:** keep the ΔF1/ΔG contrasts, and add the CONTRASTS of consistency.py for every thinking-on/off pair (§5).
5. **Optional cleanup:** move `Pair` from consistency.py into analyze.py, so both write one results file. Do this only once the paper's tables are frozen.

## 5. Adding the proprietary models

The pipeline is keyed by `job_id`. A model is "added" when `runs/<name>/responses.jsonl` holds one line per job:

```json
{"job_id": "direct|contour|T14-train-000000475887-3|p0", "output": "[2, 5]", "finish_reason": "stop", "n_in": 512, "n_out": 6}
```

Jobs come from `data/eval/jobs.jsonl`. Each has its `prompt` and `image` (a render under `data/eval/renders/`), plus `kind`, `row`, `modality` and `item_id`/`group`. `run.py` only drives vLLM, so an API model needs a small runner that reads these jobs and writes the lines above.

Priority, with the claims each part feeds:

| Priority | Job kinds | Calls | Feeds |
|---|---|---|---|
| 1 | `direct`, `probe` (row 1, Test, M-contour) | 10,120 | Table 2's set-vs-relations columns (on Lite subset), Table 4, C1, C2, C3, C6 |
| 2 | `probe_retest` (row 2) | 1,398 | Table 2's retest column (C1's reference level) |
| 3 | `direct_diag`, `direct_pos`, `direct_neg0` (row 3) | 1,971 | Table 3 definition uptake (C4), S6 |
| 4 | `direct_text`, `probe_text` (row 4) | 10,120 | Table 4 M-text column, S8 |
| 5 | `direct_mark`, `probe_mark`, `direct_alt`, `probe_alt`, `replica` | ~5,900 | S10, Table 4's relation-accuracy column |

The twin ("reworded") column needs no extra jobs. Both twins are in row 1.

Watch out for these pitfalls:
- **Run names containing "thinking".** `score.answer_text(text, thinking=True)` returns `""` when the output has no `</think>`. For an API reasoning model that hides its reasoning, store the visible answer and either keep "thinking" out of the run name, or write `output` as `"</think>" + answer`. A run named `…-thinking` without the tag scores every answer as `[]`.
- **Decoding.** Use temperature 0 where the API allows it, to match the greedy instruct rows. Record the decoding in the run directory. Where an API only samples (most reasoning modes), say so in Table 4's caption, as for Qwen3-VL-8B-Thinking.
- **Thinking on/off pairs.** Run both modes for at least one proprietary model, under separate run names. Add the pair to `CONTRASTS` in consistency.py. C5 needs ≥ 2 pairs.
- **Smoke gate.** Run `check_smoke.py` on 24 jobs per kind first (< 95% parsed, > 2% truncated, or > 5% of relation calls missing a candidate fails it).
- **Names.** Add each run to `NAMES` in analyze.py, which consistency.py imports.
- **Coverage.** consistency.py builds its full-Test block only for runs that answered ≥ 99% of Test `direct` jobs; otherwise the run appears in the Lite tables only. To save cost, a proprietary model can be run on Lite only (`--lite` equivalent: Lite items and Lite anchors). Table 2 is on Lite anyway.

Then run:

```bash
python3 tools/eval/analyze.py     <all run names>
python3 tools/eval/consistency.py <all run names>
```

## 6. Checking the claims after the proprietary rows arrive

The claim register in [paper_scaffold.md](../../paper_scaffold.md) fixes each test before the new data. Apply it mechanically:

| Claim | Read from | Pass |
|---|---|---|
| C1 | consistency.md Table 2: "Δφ reworded − relations"; agreement vs. independence level | Δφ CI > 0 for every new model; agreement − independence ≤ 0.15 |
| C2 | consistency.md Table 2: "Δφ L1 − L2/L3" | CI includes 0 or lies below it. Today Gemma fails (0.11, CI 0.03 to 0.19) |
| C3 | consistency.md S-C3 | under-applied > over-applied, non-overlapping CIs, AND-NOT and EXCEPT, every non-thinking mode |
| C4 | results.md S3 block (c), M-contour | inclusion difference CI > 0 |
| C5 | consistency.md S-C4, cells AND-NOT and EXCEPT; S-C3 under-applied | Δφ CI > 0 in both cells and lower under-application, in ≥ 2 thinking pairs |
| C6, C7 | results.md Table 2 (→ Table 4) | as in the register |

Apply Holm over C1–C5 within each new model. Then update every `[validate]` range in the scaffold: the abstract, §1, §5 and §7.

## 7. Regenerating Fig. 1

```bash
python3 tools/eval/running_example.py qwen3-vl-4b-instruct qwen3-vl-8b-instruct
```

The example (`val-000000534270-15`; T12, T11, T14) is in Test but not in Lite. Lite-only runs therefore have no outputs for it. If a proprietary model is run on Lite only, keep Fig. 1 on Qwen3-VL-4B.
