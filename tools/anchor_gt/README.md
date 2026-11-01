# ANCHOR ground-truth generator and viewer

Builds the rule-anchored selection items of [design_proposals.md](../../design_proposals.md) §4 from COCONut-PanCap, and lets you inspect them.

```bash
python tools/anchor_gt/generate.py --split val          # ~15 s
python tools/anchor_gt/generate.py --split train        # ~6 min on 15 workers
python tools/anchor_gt/generate.py --sample             # draft Test sample + summary.json
python tools/anchor_gt/server.py --port 8778            # open http://localhost:8778
python tools/anchor_gt/lexicon.py                       # regex lexicon self-check on val captions
```

| file | role |
|---|---|
| `rules.py` | thresholds, 3-valued predicates, the 16 templates of design v0.3.3 (program, primitives/cell order, NOT/EXCEPT clauses, Q-matrix components, twins, 3 paraphrases), the build-time checks (`check_template_set`), prompt rendering incl. the shared all-primitive probe |
| `lexicon.py` | regex: phrase–category consistency, group/partial flags, scene tags (indoor/outdoor, settings, food, people), physical-size classes, caption-stated contact/occlusion between tags (`contact_pairs`) |
| `generate.py` | subject funnel → candidates → predicates → GT set S\*, ignore set U, per-candidate truth-table cells, opportunity count (`fire`) and counter-heuristic ids per (subject, template); nested draft sampler |
| `server.py`, `viewer.html` | simple viewer at `/`: Overview (filter steps), Rules (16 rules in plain words), Images (answer + why per object) |
| `viewer_detailed.html` | every statistic, Q-matrix and design checks, at `/detailed` |

The build fails if a §4.1 constraint is violated (NOT/EXCEPT family coverage, Q-matrix full rank, ≥2 twin pairs).

Outputs in `data/anchor_gt/`: `{split}/details.jsonl` (one record per image), `{split}/index.json`, `{split}/stats.json`, `summary.json`, `test_draft.jsonl`, `test_diag.jsonl`, `regex_audit_haiku.json` (LLM spot-check of the regex labels on 150 images).

Naming: the predicate `touch` means outlines meeting in the image (overlap counts), so prompts, probe keys (`rules.PROBE_KEY`) and the viewer call it *borders*; `border` is shown as *reaches the image edge*. Code identifiers are unchanged.

Choices not fixed by the design doc:
- **M-text** subjects are unmarked; **M-contour** subjects are outlined magenta with no number; **M-mark** subjects carry a number.
- `border` has an ignore band like the other geometric predicates: edge distance ≤ 1 px → yes, > 1% of the diagonal → no.
- `CLOSURE_touch(same_cat)` is one hop: same-kind candidates plus candidates touching one of them.
- Phrases naming only another COCO category (mis-tags) drop the subject entirely; phrases naming no category ("gray jacket") keep the subject for M-contour/M-mark only.
- An item is **counter-heuristic** when answering one of its candidates by a shortcut (typical physical size for `larger`, box overlap for `touch`, box edge for `left`/`above`) would change that candidate's GT label.
- Draft Test sample (`sample_nested`, design §3.2): pool = val + a 10% hash slice of train, text_ok subjects only (every item runs in M-text and M-contour). Each subject gets a target L2/L3 template, then its twin, its L1 companions, and further L2/L3 targets sharing a primitive (≤3). |S\*| quotas (15% empty, 17% each for 1…5+) apply to L2/L3; ≥70% of NOT/EXCEPT/CLOSURE items have a candidate in the opportunity cell.

**Caption-contact diagnostic (`C01`, outside the 16 templates).** `contact` is physical contact as the caption states it. It is True only when two consecutive tags in one sentence are joined by nothing but a contact/occlusion cue ("holds", "rests on", "seated on", "partially covered by", …) *and* the outlines meet (gap ≤ 1.5 px). It is False when the masks are far apart (> 2% of the diagonal), and ignored otherwise, because captions omit most contacts: only 3–5% of pairs whose outlines meet are stated. `C01` uses everyday wording ("physically touching") and is paired with `T01` (operational "borders") on the same anchor, so the two definitions can be compared on the same candidates. Results are stored per anchor under `diag` (not `rules`, so the funnel, quotas, Q-matrix and viewer are unchanged), with `contact_stated` listing the cue kind per candidate. `sample_diag` draws 300 `C01` items (15% empty, 85% with ≥ 1 stated contact), preferring anchors whose `T01` is already in the main draft, and adds a `T01` item (`role: diag_pair`) only where it is missing. `test_draft.jsonl` is unchanged by this slice. Audit counts are in `summary.json → diagnostic.contact_audit`.
