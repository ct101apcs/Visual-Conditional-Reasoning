# Object-Relational, Spatial, and Region-Conditioned Reasoning Benchmarks for VLMs: What Exists, How Ground Truth Is Built, and Where "Subject + Programmatic Rule -> Set" Is Still Open

Scope note: every arXiv ID below was checked against its arxiv.org/abs page on 2026-10-27 (title, first author, and date match). Numbers were read from the paper PDFs or arXiv abstracts unless marked "(unverified)" or "(abstract-level)". PDFs saved under `lit-review/papers/` (PDFs are not tracked in git; see `lit-review/README.md`). This session downloaded the files marked [DL]. Files marked [DL-other] were already in the folder from parallel researchers. Entries with no marker were read at abstract/web level only. Target of our rebuild: a large COCONut-PanCap-derived benchmark ([DL-other] `2502.02589_Deng2025_coconut-pancap.pdf`), where a VLM (1) grounds a subject given a reference (text / mask / contour / box / numbered mark), (2) applies a prompt-specified, programmatically checkable rule relative to the subject, and (3) outputs the *set* of matching objects. Search note: the "Osprey-Bench" and "RegionBench" names in the brief could not be located as standalone benchmarks; Osprey (2312.10032) reports region classification/captioning evaluations only (abstract-level).

## Q1. Spatial-relation and geometric benchmarks: reference style, GT construction, ambiguity handling, frontier accuracy

### Takeaway
The field splits into four generations:
1. **Image-text-matching probes** (VSR, What'sUp, COCO/GQA-spatial). These are binary or 4-way, built from captions or controlled photos, and now near saturation for MLLMs.
2. **Programmatic "vision-centric" VQA from existing annotations** (CV-Bench, SpatialRGPT-Bench, SpatialVLM, EmbSpatial, 3DSRBench). GT comes from boxes, 3D cuboids or depth. Questions are almost always *pairwise* (A vs B, or A vs anchor) and multiple-choice.
3. **Hand-built cognitive-science suites** (OmniSpatial, SpaCE-10, SITE, MMSI, SpatiaLab, VSI-Bench). These are hard (frontier models 40-57%) but small and human-authored.
4. **Bias/shortcut diagnostics** (3DSRBench FlipEval, "Why Far Looks Up", COMFORT frame-of-reference, SpatialUncertain). They show that headline accuracy on the older sets is partly shortcut-driven.

Only three works touch our design directly: reference-type variation with visual marks (CV-Bench boxes, SpatialRGPT region tokens/SoM, EarthSpatialBench overlays vs coordinates), *group/aggregate* spatial queries (EarthSpatialBench), and mask-derived rule metadata (Rädsch et al.). None combines all three on natural images with a Boolean rule over a subject and set-valued output.

### Cited Findings

**A. Caption-/photo-matching probes (saturating)**
- **Liu et al. 2023, VSR** (2205.00363) [DL `2205.00363_Liu2022_visual-spatial-reasoning-vsr.pdf`]. 10,972 COCO image-caption pairs after filtering (true/false), 66 relation types, including "touching". Filter: keep items where at least 2 of 3 validators agree with the original label. Human ceiling 95.4%; Fleiss' kappa 0.895 on 500 samples; SOTA models ~70% at publication. Per-relation annotator disagreement is highest for reference-frame-dependent ("at the side of", "in front of") and vague ("near", "close to") relations, lowest for part-whole/in-out. A zero-shot concept split drops all models. — [arXiv 2205.00363](https://arxiv.org/abs/2205.00363)
- **Kamath et al. 2023, What'sUp** (2310.19785) [DL `2310.19785_Kamath2023_whats-up-vlm-spatial-reasoning.pdf`]. 820 tripod-controlled photos in 205 sets of 4 (same objects, only relation changes: on/under/left/right/front/behind); Subset A has real-world priors, Subset B does not. Also COCO-spatial (2,687 images; only mutually exclusive relations; each object >=3% of image) and GQA-spatial (1,451 images). Chance 25%/50%; humans ~97-100%; 18 VLMs mostly a few points above chance; best on What'sUp 42.1% (XVLM-COCO) vs 71.0% on COCO-spatial; pair/set accuracy near zero. States that >92% of GQA validation questions mix spatial reasoning with other skills. — [arXiv 2310.19785](https://arxiv.org/abs/2310.19785)
- **Rajabi & Kosecka 2024, GSR-Bench** (2406.13246) [DL `2406.13246_Rajabi2024_gsr-bench-grounded-spatial-reasoning.pdf`]. Extends What'sUp with bounding boxes and depth to separate grounding from relation reasoning. LLaMA-3-LLaVA-NeXT-8B reaches 86.1% on What'sUp, versus 60.4% for XVLM-COCO. So the original What'sUp is now close to saturated for MLLMs. Grounding scored as IoU >= 0.5 against GroundingDINO pseudo-GT. — [arXiv 2406.13246](https://arxiv.org/abs/2406.13246)
- **Liu et al. 2025, SpatialMQA** (2505.19015). Human-annotated, COCO2017-based, 5,392 samples, motivated by benchmarks that rely on boxes, ignore perspective substitution, or are answerable from priors. Best MLLM 48.14% vs 98.40% human (abstract-level). — [arXiv 2505.19015](https://arxiv.org/abs/2505.19015)

**B. Programmatic benchmarks from annotations (closest in spirit to ours)**
- **Tong et al. 2024, Cambrian-1 / CV-Bench** (2406.16860) [DL `2406.16860_Tong2024_cambrian-1-cv-bench.pdf`]. 2,638 manually inspected examples repurposed from ADE20K, COCO, Omni3D: 2D Spatial Relationship (650; left/right, top/bottom relative to an *anchor*), 2D Count (788), 3D Depth Order (600), 3D Relative Distance (600). Questions are generated programmatically, then unclear/ambiguous/erroneous ones are removed by hand. **Objects are referenced with colored bounding boxes** so duplicates of a category are not ambiguous. **Depth ambiguity margin**: A is closer than B only if A's *farthest* vertex is closer than B's *nearest* vertex by an offset (same rule vs anchor for relative distance). Reported (2D / 3D): GPT-4V 64.3 / 73.8; Cambrian-1-34B 74.0 / 79.7; LLaVA-NeXT-34B 73.0 / 74.8. — [arXiv 2406.16860](https://arxiv.org/abs/2406.16860)
- **Cheng et al. 2024, SpatialRGPT / SpatialRGPT-Bench** (2406.01584) [DL `2406.01584_Cheng2024_spatialrgpt-grounded-spatial-reasoning.pdf`]. 657 qualitative + 749 quantitative VQA pairs, 88 classes, from Omni3D 3D cuboids (nuScenes, KITTI, SUN RGB-D, ARKitScenes, Hypersim). 1,500 synthesized samples were randomly selected and then human-filtered. Qualitative relations: below/above, left/right, big/small, tall/short, wide/thin, behind/front. Regions are referenced as "Region [i]" via masks/boxes; baselines add Set-of-Mark. Grading uses GPT-4 as judge; quantitative success = within +/-25% of GT. Qualitative averages: GPT-4V 58.14, GPT-4 *blind with only the object class in text* 57.83, LLaVA-v1.6-34B 43.98, SpatialRGPT-VILA-1.5-8B 92.69. Blind GPT-4 matches GPT-4V on width/height quantities via world knowledge. Set-of-Mark gives LLaVA-v1.6-34B no gain (qualitative average 42.31 vs 43.98 without SoM) and lowers its quantitative success (for example direct distance 12.8 vs 24.3). — [arXiv 2406.01584](https://arxiv.org/abs/2406.01584)
- **Chen et al. 2024, SpatialVLM** (2401.12168) [DL `2401.12168_Chen2024_spatialvlm.pdf`]. Data engine: 2B spatial VQA on 10M internet images with lifted 3D point clouds. Reference disambiguation: FlexCap variable-length captions plus a CLIP-similarity pass that augments or rejects ambiguous captions (e.g., two "cake"s). Questions involve no more than two objects. Binary-predicate accuracy: GPT-4V 68.0%, LLaVA-1.5 71.3%, SpatialVLM 75.2%. The human-labelled benchmark was not released (SpatialRGPT says so). — [arXiv 2401.12168](https://arxiv.org/abs/2401.12168)
- **Du et al. 2024, EmbSpatial-Bench** (2406.05756) [DL `2406.05756_Du2024_embspatial-bench.pdf`]. Auto-derived from MP3D/AI2-THOR/ScanNet; 6 egocentric relations; 4 options. Filter: drop triplets whose object box has a side <50 or >half the image dimension; then human check that each object is uniquely identifiable, the relation holds, and negatives are truly wrong. Best LVLM 49.11% (generation) vs human 90.33%. — [arXiv 2406.05756](https://arxiv.org/abs/2406.05756)
- **Ma et al. 2024, 3DSRBench** (2412.07825) [DL `2412.07825_Ma2024_3dsrbench.pdf`]. 2,772 manually annotated QA, 12 types. **FlipEval** (horizontal-flip the image; left/right answers flip) removes left/right priors (for example drivers sit on the left); CircularEval for option-order. "Random++" baseline. Uncommon-viewpoint accuracy drops 13.5% (GPT-4o), 32.2% (Gemini-1.5-Pro); InternVL2.5-72B is still >40 points below human. — [arXiv 2412.07825](https://arxiv.org/abs/2412.07825)
- **Fu et al. 2024, BLINK** (2404.12390) [DL `2404.12390_Fu2024_blink-multimodal-llms-see-but-not-perceive.pdf`]. Relative depth (124 test items; two *marked points* from Depth in the Wild), spatial relation (143), counting (120). Test accuracy: relative depth - human 99.19, GPT-4o 64.52, GPT-4V 58.87, Gemini Pro 50.00 (chance 50); spatial relation GPT-4o 76.92; counting GPT-4V 60.83. Best models are only 13.17 points over random on the whole suite. — [arXiv 2404.12390](https://arxiv.org/abs/2404.12390)
- **Wang et al. 2025, Spatial457** (2502.08636) [DL `2502.08636_Wang2025_spatial457.pdf`]. Synthetic (CLEVR-style) 6D benchmark: 7 question types, 5 difficulty levels, Relative Performance Dropping Rate; declines with complexity especially in 3D/6D (abstract-level). — [arXiv 2502.08636](https://arxiv.org/abs/2502.08636)

**C. Hard hand-built suites (frontier numbers)**
- **Jia et al. 2025, OmniSpatial** (2506.03135) [DL `2506.03135_Jia2025_omnispatial.pdf`]. 8.4K+ manually annotated QA, 4 categories / 50 subcategories (dynamic reasoning, complex spatial logic, spatial interaction, perspective-taking). The authors show o3 and Gemini-2.5-Pro exceed 90% on SpatialBot-Bench and EmbSpatial ("basic tasks approach saturation"); on OmniSpatial VLMs peak near 57% vs human 92.63%. — [arXiv 2506.03135](https://arxiv.org/abs/2506.03135)
- **Yang et al. 2025, MMSI-Bench** (2505.23764) [DL `2505.23764_Yang2025_mmsi-bench.pdf`]. 1,000 hand-crafted multi-image MCQs, 37 MLLMs; GPT-5 ~40%, best open ~30%, humans 97%; error taxonomy leads with grounding errors. — [arXiv 2505.23764](https://arxiv.org/abs/2505.23764)
- **Wasi et al. 2026, SpatiaLab** (2602.03916) [DL `2602.03916_Wasi2026_spatialab-spatial-reasoning-in-the-wild.pdf`]. 1,400 real-image QA, 6 categories x 5 subcategories, MCQ and open-ended, 3-tier review. MCQ best InternVL3.5-72B 54.93% vs human 87.57%; open-ended best GPT-5-mini 40.93% vs human 64.93%. Depth & Occlusion MCQ best 58.30 (o4-mini; human 74.13); closed models "collapse to near-chance" on Relative Size Comparison. — [arXiv 2602.03916](https://arxiv.org/abs/2602.03916)
- **Yang et al. 2024, VSI-Bench** (2412.14171) [DL `2412.14171_Yang2024_vsi-bench-thinking-in-space.pdf`]. 5,000+ video QA; human 79.2 vs Gemini-1.5 Pro 45.4; chain-of-thought and self-consistency do not help, cognitive maps help distance questions. — [arXiv 2412.14171](https://arxiv.org/abs/2412.14171)
- **Gong et al. 2025, SpaCE-10** (2506.07966): 10 atomic -> 8 compositional capabilities, 5k+ QA on 811 indoor scenes; counting shortfall limits compositional performance (abstract-level; the "GPT-4o 37.4% vs human 72.0%" figure seen on a search snippet is unverified). — [arXiv 2506.07966](https://arxiv.org/abs/2506.07966)
- **Wang et al. 2025, SITE** (2505.05456): 8,068 MCQ tasks from 31 datasets, single/multi-image/video; models trail experts most in spatial orientation (abstract-level). — [arXiv 2505.05456](https://arxiv.org/abs/2505.05456)

**D. Shortcut and ambiguity diagnostics**
- **Min et al. 2026, "Why Far Looks Up" / SpatialTunnel** (2605.30161) [DL `2605.30161_Min2026_why-far-looks-up-spatialtunnel.pdf`]. VLMs entangle vertical image position with distance. On EmbSpatial depth questions 80.9% are "consistent" with the heuristic (farther object higher in image) and 10.7% counter; on CV-Bench-3D 60.5% consistent, 10.8% counter, and **28.7% flagged ambiguous (|delta y| < 5% of image height)**. Qwen2.5-VL fine-tuned on 2M spatial samples: 60.9% on consistent vs 24% on counter EmbSpatial items (36.9-point gap); the gap grows with spatial fine-tuning data. Size can also conflict with depth as a shortcut. — [arXiv 2605.30161](https://arxiv.org/abs/2605.30161)
- **Zhang et al. 2024, COMFORT** (2410.17385) [DL `2410.17385_Zhang2024_spatial-frame-of-reference-ambiguities.pdf`]. Nine VLMs show poor robustness/consistency under reference-frame ambiguity (camera vs object-intrinsic vs relative) and cannot flexibly switch frames (abstract-level). — [arXiv 2410.17385](https://arxiv.org/abs/2410.17385)
- **Zhang et al. 2026, SpatialUncertain** (2605.30557) [DL `2605.30557_Zhang2026_seeing-isnt-knowing-spatialuncertain.pdf`]. Under perspective conflict, judgments follow projected 2D appearance rather than 3D relation; models do not become more cautious as occlusion removes evidence (abstract-level). — [arXiv 2605.30557](https://arxiv.org/abs/2605.30557)
- **Vo et al. 2026, Depth-Ordinal Prompting** (2607.11173): showing a depth map can *hurt* VLMs; an ordinal text cue at the queried objects helps (abstract-level). — [arXiv 2607.11173](https://arxiv.org/abs/2607.11173)

### Implications for our design
- **Well covered (do not claim novelty):** pairwise left/right/above/below, pairwise "closer to camera", pairwise size comparison (big/small, tall/short, wide/thin), touching/contact as a binary pair question (VSR, Rädsch), and counting. For each there is a reported frontier number to cite as context.
- **Ambiguity-margin strategies to copy:**
  - CV-Bench's conservative interval rule: "A nearer than B only if A's far extent is nearer than B's near extent by an offset". For us, a depth-order rule should require non-overlapping per-mask depth percentiles plus a margin.
  - Min et al.'s ambiguity band (|delta y| < 5% of image height) as a *reported tag*, not only a filter.
  - What'sUp COCO-spatial's >=3% area floor and "mutually exclusive relations only".
  - EmbSpatial's bbox size filter (side <50 or >half the image) and uniqueness check.
  - VSR's 2/3 agreement filter and per-relation disagreement report.
  - SpatialRGPT's +/-25% tolerance for any metric rule.
- **Debias like 3DSRBench:** use FlipEval-style horizontal flips for left/right rules (the answer set must flip accordingly, which is trivial to verify programmatically with masks). Balance "consistent vs counter-heuristic" cases for nearer/farther rules (report accuracy on each).
- **Blind baseline is mandatory.** SpatialRGPT-Bench shows a text-only GPT-4 with class names (57.83) equals GPT-4V (58.14) on qualitative relations. Our benchmark should report a no-image / shuffled-image / class-names-only baseline and a "prior-only" baseline (always-predict-common-category).
- **Reference-type axis:** CV-Bench uses boxes; SpatialRGPT uses regions plus SoM; SoM hurt LLaVA-34B. A within-benchmark ablation of text vs mask overlay vs contour vs box vs numbered mark on identical questions is, as far as found, only partly done (see Q3 VP-Bench and EarthSpatialBench).

### Gaps
- Nearly every item is *pairwise* (A vs B or A vs anchor) with an MCQ/binary answer. Subject-vs-all-others set retrieval is absent from the natural-image spatial literature.
- No natural-image benchmark composes rules (AND/NOT, exceptions) over spatial predicates; OmniSpatial's "complex spatial logic" is hand-written puzzle-style, not programmatic.
- Depth GT in real-image sets is from human point labels (BLINK/DIW), 3D cuboids (Omni3D) or monocular estimates; few report an explicit margin and its effect on accuracy.
- Frontier "near chance" cases are documented for relative depth (BLINK Gemini Pro 50.0) and relative size (SpatiaLab), but with 100-300 items; no large-N confirmation.

## Q2. Size, counting, contact/adjacency, occlusion, depth order: documented failure patterns

### Takeaway
Counting and depth ordering are the best-documented weak points, but evidence on *size comparison* and *contact* is thin and partly contradictory (blind LLMs do well on size from priors; humans find contact/occlusion labeling hard). Occlusion hurts models modestly (about 5-6 points of sMAPE) while the chief failure is counting itself.

### Cited Findings
- **Paiss et al. 2023, "Teaching CLIP to Count to Ten"** (2302.12066) introduces CountBench, the standard small counting probe for CLIP-style models. — [arXiv 2302.12066](https://arxiv.org/abs/2302.12066)
- **Deitke et al. 2024, Molmo and PixMo** (2409.17146) [DL-other `2409.17146_Deitke2024_molmo-pixmo-points.pdf`]. PixMo-Count: detector-generated counting QA from web images (class with most strong detections), 120 human-verified samples per count for counts 2-10, giving 540-image val and test sets; ~36k train images with point annotations. Pointing + chain-of-thought "point-and-count" leads: Molmo-72B 91.2 CountBenchQA / 85.2 PixMo-Count vs GPT-4o 87.9 / 59.6. Pointing is a *set-valued output* format: a model emits all instance points. — [arXiv 2409.17146](https://arxiv.org/abs/2409.17146)
- **Acharya et al. 2018, TallyQA** (1810.12440): counting questions "that involve relationships between objects, attribute identification, reasoning" (complex-counting subset), the earliest relation-conditioned counting set (abstract-level; size unverified). — [arXiv 1810.12440](https://arxiv.org/abs/1810.12440)
- **Pothiraj et al. 2025, CAPTURe** (2504.15485) [DL `2504.15485_Pothiraj2025_capture-occluded-object-counting.pdf`]. Count objects in a pattern behind an occluder; sMAPE error. GPT-4o best at 14.75% (real) and 9.71% (synthetic); average models are 6.28% (real) and 4.85% (synthetic) worse with occlusion; Molmo 32.5% on real-occluded; humans near zero. Supplying object coordinates as text helps, so error comes from both counting and occlusion inference. — [arXiv 2504.15485](https://arxiv.org/abs/2504.15485)
- **Rahmanzadehgervi et al. 2024, "VLMs are Blind"** (2407.06581) [DL-other `2407.06581_Rahmanzadehgervi2024_vlms-are-blind.pdf`]. Four SOTA VLMs average 58.07% on 7 trivially simple geometric tasks (line intersections, overlapping circles, etc.); best Claude-3.5-Sonnet 77.84%. Suggests that "touching/overlap" at fine granularity is a real weakness. — [arXiv 2407.06581](https://arxiv.org/abs/2407.06581)
- **Rädsch et al. 2025** (2502.15563) [DL `2502.15563_Radsch2025_scalable-vlm-benchmark-generation-framework.pdf`]. Generates 25 tasks per image from instance masks, including Object Touching (mask-adjacency rule), Size Comparison, Depth Comparison (Depth Anything V2 with a minimum depth difference), Horizontal/Vertical Comparison. Human raters (6 per task, early stop at 4-agreement) found "Object Occlusion Detection" and "Object Touching Detection" the *hardest* tasks for humans, so mask-adjacency contact is ambiguous. — [arXiv 2502.15563](https://arxiv.org/abs/2502.15563)
- **Wasi et al. 2026, SpatiaLab** (2602.03916) [DL, above]: Depth & Occlusion MCQ 50-58% for top models (human 74%), Size & Scale 41-55% (human 89%); closed models near chance on Relative Size Comparison. — [arXiv 2602.03916](https://arxiv.org/abs/2602.03916)
- **Fu et al. 2024, BLINK** [DL, above]: relative depth 50.00 (Gemini Pro) to 64.52 (GPT-4o) vs 99.19 human; chance 50. — [arXiv 2404.12390](https://arxiv.org/abs/2404.12390)
- **Min et al. 2026** [DL, above]: depth order inflated by the vertical-position heuristic; counter-heuristic accuracy as low as 24% for a 2M-sample spatially fine-tuned model. — [arXiv 2605.30161](https://arxiv.org/abs/2605.30161)
- **Cheng et al. 2024, SpatialRGPT-Bench** [DL, above]: blind GPT-4 with class names is competitive on width/height (size comes from priors). — [arXiv 2406.01584](https://arxiv.org/abs/2406.01584)
- **Gong et al. 2025, SpaCE-10**: counting shortcoming limits compositional spatial scores (abstract-level). — [arXiv 2506.07966](https://arxiv.org/abs/2506.07966)

### Implications for our design
- **Counting is a hidden dependency of set-output tasks.** A set answer with k elements is also a counting problem. Report set size (k) as a difficulty axis and show exact-match/F1 vs k. PixMo-Count's per-count balancing (counts 2-10 x 120) is a template.
- **Size rules:** a "larger than subject" predicate should use mask area (or box area) with a ratio margin (for example >=1.25x) and be tagged against a "prior-only" baseline (people vs cups) because blind LLMs already exploit category-size priors. Include *counter-prior* items (small instance of a large class) as a stress split, mirroring Min et al.'s consistent/counter split.
- **Contact rules:** define "touching" as mask adjacency with an explicit pixel gap and minimum shared-boundary length; expect inherent ambiguity (Rädsch: humans struggle). Keep it as a rule with a tolerance sweep rather than a single ground-truth claim; report model accuracy as a function of boundary length.
- **Depth-order rules:** use predicted depth only where the per-object depth intervals are disjoint by margin; report the heuristic-consistent vs counter split, and an occlusion-ordering variant (if mask A occludes B per overlap in panoptic annotation) as an alternative depth proxy.

### Gaps
- No benchmark studies "larger than X" on *all* objects with a controlled ratio margin, nor with category-prior controls.
- Contact/support relations are mostly single binary items (Rädsch T4.1, VSR "touching"); no set-valued contact (list all things touching X).
- Depth Anything-type pseudo-GT is used without validation against metric depth in the COCO domain (Rädsch uses threshold only).
- Counting benchmarks rarely require *which* instances (points/boxes), only how many.

## Q3. Region/object-conditioned QA and relation benchmarks; set retrieval vs yes/no/MCQ

### Takeaway
Scene-graph-generated VQA (GQA, Visual Genome QA, CLEVR) set the template for programmatic questions but answer *single* values; relation-hallucination sets (MMRel, R-Bench, Reefknot) are yes/no or MCQ. Set-valued answers exist in (a) generalized referring expression segmentation/comprehension (gRefCOCO, MC-Bench, MMR), (b) pointing/counting (PixMo), and (c) one remote-sensing benchmark with aggregate group queries (EarthSpatialBench). None of these gives an explicit, composable, programmatically verified rule relative to a *given subject*.

### Cited Findings
**A. Scene-graph and synthetic-programmatic VQA**
- **Krishna et al. 2016, Visual Genome** (1602.07332): dense crowd-sourced region/object/relationship annotations, the substrate for GQA, ProVision and Reefknot. — [arXiv 1602.07332](https://arxiv.org/abs/1602.07332)
- **Hudson & Manning 2019, GQA** (1902.09506) [DL `1902.09506_Hudson2019_gqa.pdf`]. 22M questions generated from scene graphs with functional programs (select -> filter -> relate(subject, on) -> query), including verify/query/choose/logical/compare types, answer-distribution smoothing; blind LSTM 42.1%, best VQA 54.1%, human 89.3%. Criticisms to carry: the paper itself notes some question types are "more ambiguous and subjective" with no clear conclusive answer; What'sUp reports >92% of GQA validation questions mix spatial and non-spatial skills, which is why it carved out a GQA-spatial subset. — [arXiv 1902.09506](https://arxiv.org/abs/1902.09506)
- **Johnson et al. 2016, CLEVR** (1612.06890): synthetic scenes with programmatic questions and "minimal biases" including spatial-relation, comparison, count and logical operations. — [arXiv 1612.06890](https://arxiv.org/abs/1612.06890)
- **Li et al. 2022, Super-CLEVR** (2212.00259): factorizes domain shift (visual complexity, question redundancy, concept distribution, compositionality) in a controlled generator; the model for "controlled factor" ablations. — [arXiv 2212.00259](https://arxiv.org/abs/2212.00259)
- **Yang et al. 2022, PSG** (2207.11247): panoptic scene graph generation on 49k COCO/VG images with mask-level relations instead of boxes, because box-based labels include redundant classes and omit background (stuff). The only mask-grounded relation annotation on COCO images; a natural source of "touching / on / beside" labels. — [arXiv 2207.11247](https://arxiv.org/abs/2207.11247)

**B. Relation understanding / hallucination benchmarks (yes/no or MCQ)**
- **Nie et al. 2024, MMRel** (2406.09121): 22,500 QA, ~400 relations, three domains, adversarial unusual-relation cases, 28 MLLMs (abstract-level). — [arXiv 2406.09121](https://arxiv.org/abs/2406.09121)
- **Wu et al. 2024, R-Bench** (2406.16449): image-level and instance-level relationship hallucination questions; LVLMs ignore visual content and over-rely on LLM commonsense; struggle with spatial relations from context (abstract-level). — [arXiv 2406.16449](https://arxiv.org/abs/2406.16449)
- **Zheng et al. 2024, Reefknot** (2408.09429): 20,000+ samples from Visual Genome scene graphs; notes earlier datasets are biased by systematic annotation; a confidence-based mitigation reduces hallucination rate by 9.75% on average (abstract-level). — [arXiv 2408.09429](https://arxiv.org/abs/2408.09429)
- **Zhao et al. 2022, VL-CheckList** (2207.00221): object/attribute/relation taxonomy; a CheckList-style capability decomposition. — [arXiv 2207.00221](https://arxiv.org/abs/2207.00221)

**C. Region-/visual-prompt-conditioned evaluation**
- **Yang et al. 2023, Set-of-Mark** (2310.11441) [DL-other `2310.11441_Yang2023_set-of-mark.pdf`]: numbered marks on SAM segments let GPT-4V ground by index. Yan et al. 2024 (2404.16375) report that other MLLMs struggle with these tags without special training (abstract-level). — [arXiv 2310.11441](https://arxiv.org/abs/2310.11441)
- **Xu et al. 2025, VP-Bench** (2511.11438) [DL-other `2511.11438_Xu2025_vp-bench.pdf`]. Stage 1: 30K visualized prompts, 8 shapes (tags, boxes, masks, contours, ovals, points, etc.), 355 attribute combinations; tests existence, enumeration, referencing, spatial localization. Stage 2: six VP-enabled downstream tasks. 28 MLLMs; Stage 1 data is drawn primarily from MS-COCO. Regular shapes (box, oval) are generally more effective than irregular ones (masks, contours); adding a text description of the VP shape helps. — [arXiv 2511.11438](https://arxiv.org/abs/2511.11438)
- **Cai et al. 2023, ViP-LLaVA / ViP-Bench** (2312.00784) [DL-other `2312.00784_Cai2023_vip-llava-vip-bench.pdf`]: free-form visual-prompt VQA with arrow/box/scribble marks (details unverified here).
- **Yuan et al. 2023, Osprey** (2312.10032): mask-text instruction tuning (724K); region classification/captioning evaluation (abstract-level). — [arXiv 2312.10032](https://arxiv.org/abs/2312.10032)
- **Xu et al. 2026, EarthSpatialBench** (2602.15918) [DL `2602.15918_Xu2026_earthspatialbench.pdf`]. 325K+ QA on Earth imagery with distance/direction/topological relations; single-object, object-pair, and **compositional aggregate group queries** (objects within a distance/direction range of another object; counting or localization); **object references as text, visual overlay, or explicit coordinates (boxes, polylines, polygons)**; localization scored with IoU (threshold 0.1) and macro-F1. GPT-5 and Claude degrade markedly on *exhaustive grounding* (all buildings); overlays help Qwen3-VL but not GPT-5, which prefers coordinates. — [arXiv 2602.15918](https://arxiv.org/abs/2602.15918)

**D. Set-valued grounding**
- **Liu et al. 2023, GRES / gRefCOCO** (2306.00968) [DL-other `2306.00968_Liu2023_gres-grefcoco.pdf`] and **Ding et al. 2026, GREx** (2601.05244): expressions can refer to multiple or zero targets; relational modeling is flagged as the key difficulty (abstract-level). Expressions are human-written natural language, not programmatic rules. — [arXiv 2306.00968](https://arxiv.org/abs/2306.00968), [arXiv 2601.05244](https://arxiv.org/abs/2601.05244)
- **Xu et al. 2024, MC-Bench** (2410.12332) [DL `2410.12332_Xu2024_mc-bench-multi-context-visual-grounding.pdf`]: multi-image, open-ended text prompts localizing all instances of interest, 2K samples, 20 skills including spatial relation understanding, instance-level AP50 (abstract-level). — [arXiv 2410.12332](https://arxiv.org/abs/2410.12332)
- **Jang et al. 2025, MMR** (2503.13881) [DL-other `2503.13881_Jang2025_mmr-multi-target-reasoning-seg.pdf`]: 194K multi-target, part-level reasoning-segmentation instructions on existing masks (abstract-level). — [arXiv 2503.13881](https://arxiv.org/abs/2503.13881)
- **Li et al. 2025, GroundingME** (2512.17495) [DL-other `2512.17495_Li2025_groundingme.pdf`]: 1,005 examples, Discriminative / Spatial / Limited / Rejection; best of 25 MLLMs 45.1%; most score 0% on rejection (ungroundable queries). — [arXiv 2512.17495](https://arxiv.org/abs/2512.17495)

### Implications for our design
- **Reuse GQA's idea, not its data:** a rule is a small program (select subject -> filter by predicate -> collect), but the GT must come from panoptic masks (cleaner than VG scene graphs) and from rules defined *in the prompt*, so no reliance on annotator semantics.
- **Rule primitives already well covered in text/MCQ form:** category equality ("same type" = GQA "same/different" style, Super-CLEVR's logical operations), spatial relative position, counting. **Open as set outputs with explicit subject:** "same category as subject", "touching subject", combinations with NOT, exceptions/overrides.
- **Reference-type axis:** VP-Bench says boxes/ovals beat masks/contours for *perception*; EarthSpatialBench shows model-dependent overlay-vs-coordinates preferences. Our same-question ablation across five reference types would give a cleaner causal read than either. Include a text-only reference baseline (category name + attribute) and a "no reference" control.
- **Scoring for sets:** copy EarthSpatialBench's F1/IoU-style scoring and MC-Bench's instance-level matching; also report exact set match, precision/recall, and "empty set" correctness (GroundingME's rejection and gRefCOCO's no-target show models fail at empty answers).
- **Mark visibility:** SoM does not reliably help models without mark training (SpatialRGPT-Bench LLaVA-34B+SoM; Yan et al. 2404.16375), so a mark-style comparison should include models trained for it.

### Gaps
- No natural-image benchmark pairs (a) a given subject, (b) a prompt-stated Boolean rule, (c) set output with F1.
- PSG gives mask-level relations but is a generation task for scene-graph models, not a VLM prompt-following benchmark.
- GQA/VG ambiguity (duplicate instances, vague relations) is never quantified with a margin analysis.
- Relation-hallucination sets are yes/no, so prior-driven "yes" bias dominates (R-Bench's LLM-commonsense finding).

## Q4. Programmatic benchmark generation from dense annotations: lessons on GT noise, shortcuts, blind baselines

### Takeaway
Programmatic generation from existing annotations is accepted practice (Task Me Anything, ProVision, SpatialRGPT/SpatialVLM, Rädsch). Their shared lessons: (1) generation from *clean structured labels* avoids LLM hallucination in GT; (2) human verification of a random subset is still needed, and agreement is task-dependent; (3) thresholds on geometric differences are the main ambiguity control; (4) text-only baselines expose prior leakage; (5) templated categories give unbalanced answer priors unless explicitly smoothed.

### Cited Findings
- **Zhang et al. 2024, Task Me Anything** (2406.11775) [DL-other `2406.11775_Zhang2024_task-me-anything.pdf`]. Taxonomy of 122,866 scene graphs (113,018 images, 9,848 videos) + 1,996 3D assets; 28 task types, 365 object categories, 335 relationships, 14 spatial positions; programmatic task generators; 13 open + 18 proprietary models over >1M instances. Two-annotator human check of a random subset: 92-100% accuracy per generator (100% for 2D how-many, 92% for 3D rotation), contrasted with GQA's 70-84% human performance. Findings: all models weak at spatial reasoning (especially "middle/bottom/back"), counting, 3D attributes; best open model beats best proprietary by ~8% on spatial. Limitation acknowledged: perception, not complex reasoning. — [arXiv 2406.11775](https://arxiv.org/abs/2406.11775)
- **Zhang et al. 2024, ProVision** (2412.07012) [DL `2412.07012_Zhang2024_provision-programmatic-vision-instruction-data.pdf`]. 24 single-image and 14 multi-image generators over Visual Genome scene graphs (extended with depth and segmentation) and auto-generated graphs for DataComp; ProVision-10M. Includes "CloserObject" and "CloserToAnchorObject" generators. Argues program-based data avoids hallucination "as long as the underlying scene graph is accurate"; for unlabelled images GT inherits detector/depth-model noise. Data improves CV-Bench 2D up to 7% and 3D 8%. — [arXiv 2412.07012](https://arxiv.org/abs/2412.07012)
- **Cheng et al. 2024, SpatialRGPT pipeline** [DL, above]: metric 3D scene graph from depth + SAM masks; human verification after random selection of 1,500 of ~10,000 samples took a day; OBB vs AABB geometry error noted as a limitation. — [arXiv 2406.01584](https://arxiv.org/abs/2406.01584)
- **Rädsch et al. 2025** [DL, above]: "No LLMs or VLMs were used for task generation" (to avoid injected hallucination); minimum thresholds per task; 162,946 human reference answers; model ranking differs from human ranking. Closest framework to ours (uses COCONut masks and KITTI; rule metadata such as segmask-touch, relative size, average/top-95/bottom-5 depth). — [arXiv 2502.15563](https://arxiv.org/abs/2502.15563)
- **Pothiraj et al. 2025, CAPTURe** [DL, above]: synthetic split gives full control and human subset of 100 per split confirms near-zero human error. — [arXiv 2504.15485](https://arxiv.org/abs/2504.15485)
- **Chen et al. 2024, MMStar ("Are We on the Right Way...")** (2403.20330) [DL-other `2403.20330_Chen2024_mmstar-right-way-evaluating-lvlms.pdf`]: many benchmark items are answerable without the image (GeminiPro 42.9% on MMMU with no visual input; over 24% above random on average across six benchmarks); motivates "vision-indispensable" curation. — [arXiv 2403.20330](https://arxiv.org/abs/2403.20330)
- **Deng et al. 2025, COCONut-PanCap** (2502.02589) [DL-other `2502.02589_Deng2025_coconut-pancap.pdf`]: COCONut panoptic masks plus VLM-drafted region captions refined by humans and merged into grounded dense captions, NeurIPS 2025 D&B. This is our annotation substrate; its caption phrases are linked to masks. — [arXiv 2502.02589](https://arxiv.org/abs/2502.02589)
- **Ma et al. 2024, 3DSRBench** [DL, above]: FlipEval/CircularEval/Random++ as debiasing protocol. — [arXiv 2412.07825](https://arxiv.org/abs/2412.07825)
- **Kamath et al. 2023, What'sUp** [DL, above]: caption-prior normalization improves COCO/GQA-spatial only slightly, so text priors do not fully explain poor performance. — [arXiv 2310.19785](https://arxiv.org/abs/2310.19785)

### Implications for our design
- **Adopt the "no-LLM-in-GT" rule** (Rädsch) and publish the rule program, thresholds, and a random-subset human audit (Task Me Anything's 2-annotator, 92-100% style) rather than a kappa on a fuzzy taxonomy. Because rules are checkable, the audit tests *mask quality and rule geometry*, not perception.
- **Use COCONut's higher-quality masks** for boundaries (touching) and areas (size) but keep a per-rule ambiguity band and report accuracy inside and outside the band.
- **Report four baselines per rule:** blind (question only), class-list-only (category names of instances present, no pixels), reference-only (cropped subject), and shuffled-image. SpatialRGPT-Bench and MMStar are the evidence that this is needed.
- **Task generators lesson:** Task Me Anything's taxonomy of 14 spatial positions is a small fixed set; we can go beyond it by making the rule itself the manipulated variable (parametrised thresholds, negation, exceptions).
- **Use PanCap captions only as a source of subject names/grounded phrases**, not GT for the rule; GT stays geometric.

### Gaps
- Generators emit single-valued answers (counts, yes/no, MCQ, closest object); none treats the *answer set* as the object of evaluation.
- Few papers quantify sensitivity of model accuracy to the generator's margin parameters (CV-Bench offset values are not reported as an ablation in the sources read; Min et al. only show ambiguity share).
- Human validation is usually for correctness, not for "rule application" difficulty.

## Closest prior work / novelty threats

| Paper | What overlaps | What it lacks vs our design |
|---|---|---|
| Rädsch et al. 2025, framework for scalable benchmark generation (2502.15563) | Generates tasks from instance masks of COCO/COCONut and KITTI; mask-touching, relative size, depth comparison, left/right metadata; thresholds for reliable GT; no-LLM generation; human ambiguity baseline | Binary / 4-way / color-index answers on single objects or pairs; no subject-conditioned Boolean rule; no set output; no reference-type ablation; 1,704 images only |
| EarthSpatialBench, Xu et al. 2026 (2602.15918) | Reference given as text / overlay / coordinates; aggregate group queries ("objects within distance/direction of another object"); counting + localization with F1/IoU; exhaustive grounding failure documented | Remote-sensing imagery, not natural COCO images; relations are distance/direction/topology only; no category-equality, size, depth, rule composition or overrides; dataset size/quality claims not verified for natural domains |
| CV-Bench / Cambrian-1 (2406.16860) | Programmatic from COCO/ADE20K/Omni3D; boxes as visual prompts to disambiguate; depth margin rule; anchor-based relative questions | Pairwise MCQ; 2,638 items; no set output; no reference-type variation; no rule composition |
| SpatialRGPT-Bench (2406.01584) | Region references (masks/boxes, SoM baseline); programmatic 3D GT with human filter; blind LLM baseline | Region-specific *metric/relational* questions about a pair; 1,406 items; GPT-4-judge grading; no set retrieval; no category-equality |
| VP-Bench (2511.11438) and ViP-Bench (2312.00784) | Varies reference type (8 shapes incl. masks/contours/tags, 355 attribute combos) on COCO images | Tests VP perception and downstream QA, not a subject-relative rule; answers single-valued; no set output |
| gRefCOCO/GREx (2306.00968, 2601.05244), MC-Bench (2410.12332) | Multi-target and no-target set answers on COCO-style images; relational expressions are a stated difficulty | Human-written natural-language expressions (not parametric rules); GT correctness depends on annotator interpretation; no subject-reference modalities; no rule-family sweep |
| Task Me Anything (2406.11775) / ProVision (2412.07012) | Programmatic generators from scene graphs (closer-object, anchor-relative); human audit; taxonomy-controlled | Perceptual single-answer tasks; VG/DataComp graphs (noisier) rather than panoptic masks; no subject rule or set output |
| Molmo / PixMo-Count (2409.17146) | Set-valued output (points for all instances), counting with verification | Counting one class across the image; no relation to a subject; pointing format, no rule |
| Why Far Looks Up (2605.30161), 3DSRBench (2412.07825) | Documents shortcuts and debiasing (heuristic-consistent vs counter; FlipEval) that we must adopt | Diagnostics for pairwise depth/orientation; not a set-retrieval benchmark |
| OmniSpatial (2506.03135), SpatiaLab (2602.03916), MMSI-Bench (2505.23764) | Show frontier models are far from human on harder spatial tasks and give reportable context | Small, hand-authored MCQ; no programmatic rule sets |

## Bibliography

Verified against arxiv.org/abs on 2026-10-27 (ID | first author | year | title).
- 2205.00363 | Liu | 2022 | Visual Spatial Reasoning
- 2310.19785 | Kamath | 2023 | What's "up" with vision-language models? Investigating their struggle with spatial reasoning
- 2406.13246 | Rajabi | 2024 | GSR-BENCH: A Benchmark for Grounded Spatial Reasoning Evaluation via Multimodal LLMs
- 2505.19015 | Liu | 2025 | Can Multimodal Large Language Models Understand Spatial Relations? (SpatialMQA)
- 2401.12168 | Chen | 2024 | SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities
- 2406.01584 | Cheng | 2024 | SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models
- 2406.16860 | Tong | 2024 | Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs
- 2404.12390 | Fu | 2024 | BLINK: Multimodal Large Language Models Can See but Not Perceive
- 2412.07825 | Ma | 2024 | 3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark
- 2406.05756 | Du | 2024 | EmbSpatial-Bench: Benchmarking Spatial Understanding for Embodied Tasks with Large Vision-Language Models
- 2506.03135 | Jia | 2025 | OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models
- 2506.07966 | Gong | 2025 | SpaCE-10: A Comprehensive Benchmark for Multimodal Large Language Models in Compositional Spatial Intelligence
- 2505.05456 | Wang | 2025 | SITE: towards Spatial Intelligence Thorough Evaluation
- 2412.14171 | Yang | 2024 | Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces (VSI-Bench)
- 2502.08636 | Wang | 2025 | Spatial457: A Diagnostic Benchmark for 6D Spatial Reasoning of Large Multimodal Models
- 2505.23764 | Yang | 2025 | MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence
- 2602.03916 | Wasi | 2026 | SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?
- 2605.30161 | Min | 2026 | Why Far Looks Up: Probing Spatial Representation in Vision-Language Models
- 2605.30557 | Zhang | 2026 | Seeing Isn't Knowing: Do VLMs Know When Not to Answer Spatial Questions (and Why)?
- 2410.17385 | Zhang | 2024 | Do Vision-Language Models Represent Space and How? Evaluating Spatial Frame of Reference Under Ambiguities
- 2607.11173 | Vo | 2026 | When Depth Is Better Told Than Shown: Depth-Ordinal Prompting for Vision-Language Spatial Reasoning
- 2302.12066 | Paiss | 2023 | Teaching CLIP to Count to Ten
- 2409.17146 | Deitke | 2024 | Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models
- 1810.12440 | Acharya | 2018 | TallyQA: Answering Complex Counting Questions
- 2504.15485 | Pothiraj | 2025 | CAPTURe: Evaluating Spatial Reasoning in Vision Language Models via Occluded Object Counting
- 2407.06581 | Rahmanzadehgervi | 2024 | Vision language models are blind: Failing to translate detailed visual features into words
- 1602.07332 | Krishna | 2016 | Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations
- 1902.09506 | Hudson | 2019 | GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering
- 1612.06890 | Johnson | 2016 | CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning
- 2212.00259 | Li | 2022 | Super-CLEVR: A Virtual Benchmark to Diagnose Domain Robustness in Visual Reasoning
- 2207.11247 | Yang | 2022 | Panoptic Scene Graph Generation
- 2406.09121 | Nie | 2024 | MMRel: Benchmarking Relation Understanding in Multi-Modal Large Language Models
- 2406.16449 | Wu | 2024 | Evaluating and Analyzing Relationship Hallucinations in Large Vision-Language Models (R-Bench)
- 2408.09429 | Zheng | 2024 | Reefknot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models
- 2207.00221 | Zhao | 2022 | VL-CheckList: Evaluating Pre-trained Vision-Language Models with Objects, Attributes and Relations
- 2310.11441 | Yang | 2023 | Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V
- 2404.16375 | Yan | 2024 | List Items One by One: A New Data Source and Learning Paradigm for Multimodal LLMs
- 2511.11438 | Xu | 2025 | VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models
- 2312.00784 | Cai | 2023 | ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts
- 2312.10032 | Yuan | 2023 | Osprey: Pixel Understanding with Visual Instruction Tuning
- 2602.15918 | Xu | 2026 | EarthSpatialBench: Benchmarking Spatial Reasoning Capabilities of Multimodal LLMs on Earth Imagery
- 2306.00968 | Liu | 2023 | GRES: Generalized Referring Expression Segmentation
- 2601.05244 | Ding | 2026 | GREx: Generalized Referring Expression Segmentation, Comprehension, and Generation
- 2410.12332 | Xu | 2024 | MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs
- 2503.13881 | Jang | 2025 | MMR: A Large-scale Benchmark Dataset for Multi-target and Multi-granularity Reasoning Segmentation
- 2512.17495 | Li | 2025 | GroundingME: Exposing the Visual Grounding Gap in MLLMs through Multi-Dimensional Evaluation
- 2406.11775 | Zhang | 2024 | Task Me Anything
- 2412.07012 | Zhang | 2024 | ProVision: Programmatically Scaling Vision-centric Instruction Data for Multimodal Language Models
- 2502.15563 | Rädsch | 2025 | Bridging vision language model (VLM) evaluation gaps with a framework for scalable and cost-effective benchmark generation
- 2403.20330 | Chen | 2024 | Are We on the Right Way for Evaluating Large Vision-Language Models? (MMStar)
- 2502.02589 | Deng | 2025 | COCONut-PanCap: Joint Panoptic Segmentation and Grounded Captions for Fine-Grained Understanding and Generation
