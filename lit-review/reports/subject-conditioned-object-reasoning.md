# The Rule Is the Missing Variable: Where a Rebuilt SADL Can Still Be New

**Can a rebuilt SADL still make a novel contribution, and where?** Yes, but only if it gives up the idea that "distraction" can be annotated. The contribution has to rest on a variable no existing benchmark controls: an **explicitly defined, compositional selection rule, anchored to a referenced object in a natural image, scored as a set against deterministic ground truth.**

Each neighbouring line of work holds one or two of these pieces but none holds them all:
- **Pairwise relation probes.** Benchmarks built from masks and annotations ask pairwise questions with programmatic answers but never return sets ([Rädsch et al. 2025](https://arxiv.org/abs/2502.15563); CV-Bench, [Tong et al. 2024](https://arxiv.org/abs/2406.16860)).
- **Multi-target referring.** These benchmarks return sets, including empty ones, but the expressions are written by humans and no reference object is given as input ([gRefCOCO](https://arxiv.org/abs/2306.00968); [GREx](https://arxiv.org/abs/2601.05244)).
- **SAM 3.** It returns every instance of a concept, so its exemplar prompts already solve "same category as the subject". It has no notion of size, contact, position or exception relative to an anchor ([Carion et al. 2025](https://arxiv.org/abs/2511.16719)).
- **VLM-Fix (headline risk).** The obvious headline, that VLMs follow a rule's *name* rather than its *definition*, is already shown for VLMs. Neutral aliases shrink the standard-vs-inverse gap to 2.29 points, and loaded aliases reopen it, across 14 models ([Alam 2026](https://arxiv.org/abs/2604.12119)). That result is on board-game outcomes with single answers, so our version is an extension, not the lead.
- **The leftover gap.** No VLM benchmark tests exception or complement outputs ("X unless Y"), and no study compares thinking and non-thinking modes on prompt-defined visual rules. Both gaps are where SADL's one robust observation fits: exclusions over-applied "at scene-level plausibility".
- **Venue.** IEEE MultiMedia is a *magazine*, with 6,500 words in total, each float counting as 200 words, and at most 20 references. The design must compress to one story or move to IEEE TMM.

Sources: five research notes in [`research_notes/subject-conditioned-object-reasoning/`](../research_notes/subject-conditioned-object-reasoning/) (about 240 cited works, IDs verified on arXiv, 100+ PDFs in [`papers/`](../papers/)). The design built on this report is in [`design_proposals.md`](../../design_proposals.md).

## Referring and grounding benchmarks return sets, but nobody hands the model an anchor and a rule

Classic REC maps one expression to one object, and its labels are noisy. Ref-L4 measures annotation error at about 14% for RefCOCO, 24% for RefCOCO+ and 5% for RefCOCOg (note 01). Generalized REC (gRefCOCO/GRES/GREC, Ref-ZOM, D³, OmniLabel, HumanRef, OpenRef) adds multi-target and no-target answers. The expressions are still free language written by humans or LLMs, and ground truth depends on how annotators read them, which is exactly the reliability problem SADL had.

Reasoning segmentation (LISA/ReasonSeg, MUSE, MMR, and successors Seg-Zero, VisionReasoner, Sa2VA) moves to implicit queries such as "something to cut fruit". None of them conditions on a given reference object with a stated rule ([MUSE](https://arxiv.org/abs/2312.02228); [MMR](https://arxiv.org/abs/2503.13881)).

The nearest design match found was **GeoRefer-Bench**: executable logical forms, relation-defined referent sets and exact set scoring, on aerial imagery. It was **withdrawn on 2026-09-28** ([2609.29541](https://arxiv.org/abs/2609.29541)). It shows the idea is in the air, and that a final pre-submission search is mandatory.

Scoring of set outputs is not standardized:
- GREC uses Precision at F1 = 1.
- GRES uses gIoU/cIoU on a union mask.
- D³ and OmniLabel use AP.
- PointArena uses point-in-mask ([2505.09990](https://arxiv.org/abs/2505.09990)).

A new benchmark should therefore publish a primary metric and secondary metrics, and show its conclusions survive the choice (note 05).

**Design consequences.**
- The anchor is given, not described in free text. The rule is a program, and ground truth comes from executing it on COCONut masks.
- **SAM 3 Agent is a mandatory baseline.** Showing it succeed on `same_cat` and fail on size, contact, position and EXCEPT is the cleanest way to prove the task is not concept search.

## How you point at an object changes the answer, sometimes more than the model does

VP-Bench compares eight visual-prompt shapes across 28 MLLMs ([2511.11438](https://arxiv.org/abs/2511.11438)). Regular shapes (box, oval, tag) beat irregular ones (mask, contour, point, scribble).

Marker *style* alone can reorder leaderboards. Changing colour, size or label placement moves accuracy by up to about 13% ([Feng et al. 2025](https://arxiv.org/abs/2512.17875)). Set-of-Mark can also *hurt* open models: on What'sUp, LLaVA-1.6-34B scores 33.0 with SoM against 54.0 on the raw image ([CRG](https://arxiv.org/abs/2403.02325)).

Some frontier models cannot emit usable coordinates at all; GroundingME had to drop GPT-5, Claude and Grok from its coordinate track (note 01).

SADL's own Open Detection collapse fits this picture. Most of its lost F1 came from label-to-mask conversion, and when SoM is given ground-truth masks instead of segmenter output, it gains +14.5 mIoU on RefCOCOg (note 01).

**Design consequences.**
- The primary protocol uses **mark IDs over ground-truth segments**, with no segmenter in the scoring loop.
- Points scored by point-in-mask against GT masks are a secondary protocol.
- Free-text labels are dropped.
- Reference modality (text / mark / contour / box) is a *factor*, and marker style is swept on a subset with rank ranges reported.

## Spatial benchmarks are pairwise and multiple-choice; set retrieval over an anchor is open

Spatial evaluation has gone through four generations (note 02):
1. Image-text matching probes (VSR, What'sUp), now near saturation.
2. Programmatic "vision-centric" VQA from annotations (CV-Bench, SpatialRGPT-Bench, EmbSpatial, 3DSRBench). Questions are almost always pairwise and multiple-choice.
3. Small hand-built suites (OmniSpatial, MMSI, SpatiaLab), where frontier models score 40–57%.
4. Shortcut diagnostics (3DSRBench FlipEval; "Why Far Looks Up", [2605.30161](https://arxiv.org/abs/2605.30161)).

On EmbSpatial, 80.9% of depth items agree with a simple heuristic. One model falls to 24% on the remaining counter-heuristic items.

The closest generator is **Rädsch et al. 2025**. It derives 25 tasks per image from COCO/COCONut/KITTI masks, including mask-touching, relative size and depth with thresholds, and uses no LLM in the ground truth. Its answers are binary or 4-way about single objects or pairs, over 1,704 images ([2502.15563](https://arxiv.org/abs/2502.15563)).

**EarthSpatialBench** gives references as text, overlays or coordinates, and includes "group" queries scored by F1. It is limited to aerial imagery, to distance/direction relations, and has no composition ([2602.15918](https://arxiv.org/abs/2602.15918)).

Blind baselines matter here. On SpatialRGPT-Bench's qualitative questions, text-only GPT-4 scores 57.83 against GPT-4V's 58.14 (note 02).

**Coverage of rule primitives.**
- **Covered:** pairwise left/right/above/below, pairwise depth, size comparison, counting.
- **Under-tested:** category equality relative to a subject, mask-adjacency "touching" (which humans also find hard to label), "larger than" with a controlled ratio margin, subject-vs-all-others selection, and any negation, conjunction or override.

**Design consequences.**
- Copy the margin practices: the CV-Bench interval rule for depth, and an explicit undecided band for every geometric predicate.
- Tag every item as heuristic-consistent or counter-heuristic.
- Always report blind and class-list-only baselines.

## Rule adherence: the name-vs-definition effect exists; exceptions and set selection do not

Multimodal instruction-following benchmarks (MIA-Bench, VisIT-Bench, MM-IFEval) test format and style constraints. They do not test whether a model uses a *user-given definition* to decide which image content qualifies (note 03).

Text rule benchmarks (RuLES, RuleArena, ComplexBench, Inverse IFEval) have programmatic checkers, but they judge text. Only three works put novel rules over visual content:
- **MMCL-Bench**: 102 rubric-scored tasks ([2605.12703](https://arxiv.org/abs/2605.12703)).
- **RuleMaze**: maze planning ([2608.20237](https://arxiv.org/abs/2608.20237)).
- **VLM-Fix**: inverse game rules, 14 VLMs ([2604.12119](https://arxiv.org/abs/2604.12119)). It is the direct predecessor of our "name" manipulation. Standard-rule accuracy is 67.1% against 52.5% for the inverse rule. Neutral aliases nearly close the gap, and loaded aliases reopen it. The paper also covers fine-tuning (rule-aligned transfer) and late-layer steering.

Negation and logical composition are the open flank:
- Contrastive VLMs show affirmation bias ([NegBench](https://arxiv.org/abs/2501.09425)).
- REC models ignore "not". Tumu 2025 looked only at negated expressions that every model got wrong. In 92.9–93.8% of those cases, at least one model failed because it ignored the negation ([2511.06146](https://arxiv.org/abs/2511.06146)). This is a selected sample, not an overall rate.
- Text LLMs collapse on nested exceptions ([Simpson 2026](https://arxiv.org/abs/2607.23386)).
- **No VLM benchmark with complement or exception outputs was found, and SADL's over-applied exclusion has no prior in VLMs.**

The two documented error directions are opposite. SADL found exclusions over-applied; REC models drop the "not". A *signed* over-application index can therefore separate the two.

Reasoning modes cut both ways:
- Explicit CoT can hurt text instruction following.
- More reasoning can raise multimodal hallucination (RH-Bench).
- Thinking helps relational and rejection subsets in grounding (RefBench-PRO, Rex-Thinker).
- Strict JSON constraints can lower small-model accuracy unless the model drafts first and then formats.

No controlled thinking vs. non-thinking study on prompt-defined visual rules was found (note 03).

**Design consequences.**
- Lead with execution-vs-perception decomposition and the signed exception bias.
- Keep the name manipulation as a cheap replication of VLM-Fix on object sets, adding a counter-intuitive-name arm and a prior-intrusion rate.
- Run thinking on/off as a factor.
- Allow a free-form draft before the JSON answer, and report format compliance.

## Distractor and editing work picks targets by preference or pixels, never by a checkable rule

The distractor line has three families (note 04):
1. **Human-marked "what distracts" detectors**: [Fried et al. 2015](https://openaccess.thecvf.com/content_cvpr_2015/html/Fried_Finding_Distractors_In_2015_CVPR_paper.html) and SimpSON ([2305.17624](https://arxiv.org/abs/2305.17624)). Fried itself concedes that the subject confounds saliency.
2. **Saliency-reduction editing**, scored by a saliency model plus user studies.
3. **VLM-planned decluttering**: Progressive Photorealistic Simplification ([2605.10409](https://arxiv.org/abs/2605.10409)) and PhotoAgent. Only this family reasons about objects, and none conditions on a subject plus a stated rule. The 2026 simplification paper has a Gemini VLM pick removals under a "distractor-first" taxonomy, evaluated on 93 images.

Removal and inpainting work is about *how* to erase a given mask. The one paper that reasons about *what else* must go is ICOR/REORM: 110 items, pixel metrics ([2602.01298](https://arxiv.org/abs/2602.01298)).

Reasoning-editing benchmarks are small and judged by an LMM on the generated image (Reason-Edit, RISEBench, KRIS-Bench, CompBench). The clearest object-level evaluator, **EdiVal-Agent**, checks edits with a detector across turns but scores the image, not the choice of objects ([2509.13399](https://arxiv.org/abs/2509.13399)). Multi-turn editing benchmarks measure pixel consistency across turns, never whether the target is correctly re-resolved after the user changes the rule.

**Design consequences.**
- Framing the paper as "edit planning" collides with these papers and invites demands for a user study. Keep editing as a qualitative downstream illustration.
- The open multi-turn slot is "re-resolve the selection after the anchor or rule changes". That is the honest home for the memo's "dynamic attention" idea.

The SADL reference list has several placeholder arXiv IDs ("XXXXX") and wrong author or title metadata (SimpSON, Fried). Fix these before reusing any of it (note 04).

## COCONut-PanCap is a good source with two caveats

COCONut-PanCap ([2502.02589](https://arxiv.org/abs/2502.02589), NeurIPS 2025 D&B) provides:
- 118K train + 25K val COCO images
- human-verified panoptic masks, about 13.2 per image
- dense captions drafted by GPT-4V over set-of-marks and then human-edited, with phrases tied to mask IDs

Our own profiling of the local copy:
- val has a mean of 9.5 thing segments per image.
- 88% of thing segments are cited by a caption tag.
- 3,402 of 5,000 val images contain at least one uniquely referable anchor.
- Crisp rules leave only 0–18% of candidate labels inside ambiguity margins (see `design_proposals.md` §3.1).

Caveats to state in the paper:
- The license is research-only and non-commercial, following COCO.
- No independent quality audit exists beyond the authors' 1,000-image study.

COCO images are almost certainly in pretraining corpora. The specific rule–object combinations are not, but the limitation still has to be stated.

## What 2026 reviewers will check, and how the design answers

Reviewers' complaints fall into six families (note 05):
1. blind-solvable items
2. construct validity
3. fragility to setup knobs
4. saturation
5. label noise (3–24% in "gold" sets)
6. missing documentation

The post-2024 credible recipe, as adopted in the design:

| Reviewer check | Design response |
|---|---|
| Size and power | ≥20k test items; ≥300 per reported slice; power: a 3-pp paired difference needs ~0.9k–2.6k items at 10–30% discordance |
| Label noise | Stratified audit of ~1,200 items, Wilson/Clopper–Pearson CIs, 20% double-labelled with AC1/PABAK alongside κ, re-run on the clean subset |
| Ambiguity | Three-way GT (in / out / ignore), margin sweep, ignored fraction per rule |
| Shortcuts | Text-only, class-list-only, all/none/majority/random baselines; counter-heuristic split |
| Fragility | 3 rule paraphrases; marker-style sweep with rank ranges |
| Statistics | Image-clustered bootstrap (naive SEs can be up to 3× too small, [Miller 2024](https://arxiv.org/abs/2411.00640)); paired tests; Holm on pre-registered contrasts; confidence set for the top model |
| Efficiency | Stratified Lite subset validated by rank agreement. Not IRT, which is unreliable with 8–12 models |
| "So what?" | Three pre-registered hypotheses with effect sizes. **H1 (headline):** a model's own pairwise relation answers, executed by the rule program, beat its direct set answer, and the gap grows with clutter and rule depth. This makes Rädsch/CV-Bench-style pairwise accuracy the thing under test. **H2:** signed NOT/EXCEPT bias. **H3:** how-to-point effects |
| Reproducibility | Public generator, seeds, prompts, model versions with API dates, datasheet and Croissant metadata |

## Venue fit is a decision the team must make now

IEEE MultiMedia's official author page asks for a "conversational style" for an "educated general audience", with a "significant focus on aspects unique to multimedia systems and applications". The limits are:
- 6,500 words in total, including the abstract, references and biographies
- each figure or table counts as 200 words
- at most 20 references (30 for surveys)

It does publish dataset and benchmark-style articles: MangaUB (2025), a person re-identification benchmark (2026), and an AI-generated-image detection dataset (2026). Even so, the full design here (main table, decomposition, signed bias, clutter, modality, naming) only fits as a single story with about six floats and a large supplement.

If "IEEE Multimedia journal" in the memo meant **IEEE Transactions on Multimedia**, the full design fits as written. The [design document](../../design_proposals.md) §8 gives a float-by-float budget for the magazine option and a 20-reference shortlist.

## Open questions this review could not close

- **Cross-check search before submission.** Is any 2026-Q4 preprint combining programmatic GT, set output and a given anchor object? GeoRefer-Bench shows the space is moving.
- **Thinking vs. non-thinking.** Does thinking mode help composition but not prior intrusion? This is predicted from mixed evidence and has never been tested.
- **Size of the name effect on sets.** How large is it when the answer is a set rather than a single label? VLM-Fix's effect sizes may not transfer.
- **Tier-B audit outcome.** Will an audit of the depth- and colour-based rules pass a 5% error bar? If not, the memo's foreground/background idea stays in the supplement.
