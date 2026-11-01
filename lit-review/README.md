# Literature review

What exists around **rule-based set grounding**, and where ANCHOR sits. The search ran on 2026-10-27, and every arXiv ID was checked against arxiv.org/abs. The consistency line (SQuINT, generator–validator consistency, ConBench) was added on 2026-11-01, when the paper's headline moved to relation–set consistency.

## Reading order

1. **[reports/subject-conditioned-object-reasoning.md](reports/subject-conditioned-object-reasoning.md)** (10 min). This is the synthesis: what each neighbouring line of work holds, the design consequences, the reviewer checklist and the venue limits.
2. **[research_notes/subject-conditioned-object-reasoning/](research_notes/subject-conditioned-object-reasoning/)**: five notes, about 220 works in total. Each note answers four or five questions with cited findings, implications and gaps. Each also has a *closest prior work / novelty threats* table and a verified bibliography.
3. **[papers.md](papers.md)**: one table of every cited work, by theme, with arXiv links, local PDFs and the paper's reference numbers. It is generated; do not edit it by hand.

> The report predates the v1.0 pivot. It argues for H1 ("pairwise relation answers, executed, beat the direct set answer"), and H1 failed. The paper is now positioned on relation–set consistency (see [../paper_scaffold.md](../paper_scaffold.md) §1–2). The report's design consequences still hold.

## The five notes

| Note | Question it answers | Takeaway for ANCHOR |
|---|---|---|
| [01 Referring, grounding, visual prompting](research_notes/subject-conditioned-object-reasoning/01_referring_grounding_and_visual_prompting.md) | Do REC/GREC/reasoning-segmentation benchmarks condition on a given object and a stated rule? How does the way you point change answers? | They return sets, but from free-form expressions, with no anchor input. Answer with **mark IDs over GT segments**, so no segmenter enters the score. Treat the pointing style as a factor. |
| [02 Spatial and relational benchmarks](research_notes/subject-conditioned-object-reasoning/02_spatial_relational_region_benchmarks.md) | Which spatial primitives are tested, and how is GT made? | They are pairwise and multiple-choice. Rädsch 2025 is the closest generator. Copy its margin practice (explicit undecided band) and use counter-heuristic splits and blind baselines. |
| [03 Rule adherence, priors, logic](research_notes/subject-conditioned-object-reasoning/03_rule_adherence_priors_and_logic.md) | Do VLMs follow prompt-defined rules? What is known about negation and exceptions? | VLM-Fix already shows that rule names override definitions. No VLM benchmark tests exceptions over object sets, and no study compares thinking with non-thinking on visual rules. |
| [04 Distractors, editing, data source](research_notes/subject-conditioned-object-reasoning/04_distractors_editing_and_data_source.md) | Does distractor or removal or editing work pick targets by a checkable rule? Is COCONut-PanCap usable? | No: targets come from preference or pixels. Keep editing out of the claims. COCONut-PanCap is usable, with a research-only license and LLM-drafted captions. |
| [05 Benchmark methodology and statistics](research_notes/subject-conditioned-object-reasoning/05_benchmark_methodology_and_statistics.md) | What do 2025–26 reviewers check, and which statistics and set metrics are credible? | Use an image-clustered bootstrap, paired tests with Holm correction, a three-way GT (in / out / ignore), a GT audit with CIs, and a primary set metric with secondary metrics. |

## Position in one table

| Line of work | Representative | What it has | What ANCHOR adds |
|---|---|---|---|
| Mask-derived spatial QA | Rädsch 2025; CV-Bench; SpatialRGPT | Programmatic GT, margins | Set answers, anchored to a given subject and composed by rules; the pairwise answer becomes one of two answers being compared |
| Generalized referring | gRefCOCO/GREx; MUSE; MMR | Multi- and no-target sets | Executable rules instead of free text, so the GT does not depend on how annotators read a phrase |
| Concept segmentation | SAM 3 (+ Agent) | Every instance of a concept | Relations to the subject (size, contact, position) and exceptions. SAM 3 Agent is run as a baseline |
| Consistency across ways of asking | SQuINT; GV-consistency; ConBench | Agreement between answers | GT for every sub-answer, so error *coupling* can be measured against chance; an exact executor; two reference pairs |
| Rule following and negation | VLM-Fix; NegBench; Tumu 2025; Exception Chain Collapse | Rule drift; ignored "not"; nested exceptions (text) | Exceptions over object sets in natural images, with the error direction measured |
| Distractor and editing | SADL; SimpSON; Progressive Simplification | Subject-aware removal motivation | No attention or aesthetic claims; computed GT |

Novelty threats are listed per note. The two to re-check before submission:
- **GeoRefer-Bench** (2609.29541). It uses executable logical forms with set-valued GT, on aerial imagery. It was withdrawn on 2026-09-28.
- **Any 2026-Q4 preprint** that combines programmatic GT, set output and a given anchor object.

## Paper reference list (≤ 20, IEEE MultiMedia)

The list with reasons is at the end of [../paper_scaffold.md](../paper_scaffold.md). In [papers.md](papers.md) the **Ref** column carries each entry's number. Ref 17 (temporal attention) has no arXiv ID; its PDF is in `papers/`.

## PDFs

The 118 PDFs in `papers/` (1.3 GB) are not in git. [papers/MANIFEST.tsv](papers/MANIFEST.tsv) records their file names. To restore them:

```bash
python3 lit-review/build_index.py --fetch   # downloads every missing arXiv PDF (3 s apart); non-arXiv files are listed to add by hand
python3 lit-review/build_index.py           # regenerate papers.md after editing a note's bibliography
```
