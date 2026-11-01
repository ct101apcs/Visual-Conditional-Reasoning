# Data

Only the testbed's ground truth and the result tables are in git. Everything else (about 25 GB) is rebuilt from public sources or shipped as a release asset.

## In git

| Path | What | Size |
|---|---|---|
| `anchor_gt/test_draft.jsonl` | **The Test set**: 7,000 items (L1 1,500 · L2 4,500 · L3 1,000) on 3,120 subjects in 2,568 images. One item = image key, subject (`anchor`), template (`tid`), GT set `S`, ignore set `U`, the opportunity count `fire`, counter-heuristic candidates `ch`, the paraphrase and the modalities | 2.0 MB |
| `anchor_gt/test_diag.jsonl` | Caption-contact diagnostic: 300 `C01` ("physically touching") items + 240 paired `T01` ("borders") items | 0.15 MB |
| `anchor_gt/lite.json` | **Lite**: 1,497 Test item ids drawn as whole subjects, plus the negation, rule-probe and replica subsets | 0.2 MB |
| `anchor_gt/summary.json` | Thresholds, subject funnel, predicate rates, template checks, size quotas | 40 KB |
| `anchor_gt/regex_audit_haiku.json`, `visual_audit_sonnet.json` | LLM spot-checks of the regex caption labels and of the rendered items (not human audits) | 0.2 MB |
| `eval/results/` | `consistency.md/json` (Tables 2–3, S-C1…S-C4) and `results.md/json` (Table 4, S1–S12) | 0.2 MB |

## Not in git

| Path | What | How to get it |
|---|---|---|
| `coconut_pancap/` | COCONut-PanCap grounded captions (train 118k, val 5k) | Hugging Face release of COCONut-PanCap ([2502.02589](https://arxiv.org/abs/2502.02589)); unpack with `tools/pancap_viewer/prepare_data.py` |
| `masks/` | COCONut-S panoptic masks (train) and relabeled COCO val masks | COCONut releases on Hugging Face; `prepare_data.py` converts the parquet shards to `masks/png/` |
| `images/` | COCO 2017 images (19 GB) | val2017 from cocodataset.org; train via `tools/pancap_viewer/download_train_images.py` |
| `index/` | per-split metadata and the caption–mask check reports | `prepare_data.py`, then `verify.py` |
| `anchor_gt/{train,val}/` | per-image predicates and rule answers for every subject (385 MB) | `tools/anchor_gt/generate.py --split val` / `--split train`, then `--sample` |
| `eval/jobs.jsonl`, `eval/renders/` | model jobs and the rendered, marked images (1.2 GB) | `tools/eval/build_jobs.py` (jobs are also in the release asset) |
| `eval/runs/<model>/` | raw model outputs (`responses.jsonl`) and logs | release asset `anchor-eval-bundle.tar.gz` (21 MB), built by `tools/eval/pack_outputs.sh`. It also holds `jobs.jsonl` and the GT records of the 2,596 Test images (`anchor_gt/test_records/`, via `slim_records.py`). Or re-run `tools/eval/run.py` |

To recompute the result tables, you only need the files in git plus the release asset: extract it at the root of a fresh clone and run `tools/eval/consistency.py` / `analyze.py`. You need the images and masks only to regenerate items or renders.

**License.** COCONut-PanCap and COCO annotations are for research use only. The images keep their Flickr licenses. The captions behind M-text were drafted by GPT-4V and edited by humans.
