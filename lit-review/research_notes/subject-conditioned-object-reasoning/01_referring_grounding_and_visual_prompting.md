# Referring Expression Comprehension/Segmentation, Multi-Target Grounding, Reasoning Segmentation, and Visual Prompting: What Exists, and Where Subject-Conditioned Rule Application Sits

Scope note: every arXiv ID below was checked against the arXiv API/abs page on 2026-10-27 (title, first author, and year match). PDFs are under `lit-review/papers/` (PDFs are not tracked in git; see `lit-review/README.md`). This session downloaded the files marked [DL]. Files marked [DL-other] were already in the folder from parallel researchers. Numbers come from abstracts or from the downloaded PDFs; anything I could not confirm in a paper's text is marked "(unverified)". Context: the rebuilt SADL benchmark (COCONut-PanCap images; a VLM grounds a *subject*, applies a prompt-specified, programmatically checkable rule relative to it, and outputs the *set* of matching objects). Note on one item: GeoRefer-Bench (2609.29541) was withdrawn by its authors on 2026-09-28. Its abstract is still readable and I cite it only as a design analogue, never as established evidence.

## Q1. What do classic and generalized REC/RES benchmarks test, how is GT made, and what is known to be wrong with them?

### Takeaway
Classic REC/RES (RefCOCO/+/g) is one expression -> one object, with human-written expressions. Its annotation noise is now measured and large. Generalized variants (gRefCOCO/GREC/GRES, Ref-ZOM, D3, OmniLabel, HumanRef, OpenRef) add multi-target and no-target outputs. Their expressions are still written by humans or LLMs as free language about the image content. None of them takes a *reference object* as input and applies a *stated rule* relative to it. The only benchmark I found that does both (relation-defined referent sets scored by exact set match, from executable logical forms) is GeoRefer-Bench, which is in overhead imagery and withdrawn. The nearest classic precedent for programmatic expression-to-GT binding is CLEVR-Ref+ (synthetic CLEVR scenes). Scoring multi-target outputs has no single standard: GREC uses Precision@(F1=1, IoU>=0.5), GRES uses gIoU/cIoU on a union mask, D3/OmniLabel use AP variants, HumanRef uses Recall/Precision/DensityF1.

### Cited Findings

**A. Classic REC/RES and their measured noise**
- **Yu et al. 2016, RefCOCO/RefCOCO+ ("Modeling Context in Referring Expressions")** (1608.00272) and **Mao et al. 2015, RefCOCOg ("Generation and Comprehension of Unambiguous Object Descriptions")** (1511.02283). The three datasets are built on COCO images, one referent per expression. I verified IDs only; I did not re-read these papers. — [arXiv 1608.00272](https://arxiv.org/abs/1608.00272), [arXiv 1511.02283](https://arxiv.org/abs/1511.02283)
- **Chen et al. 2024, Ref-L4 ("Revisiting REC Evaluation in the Era of LMMs")** (2406.16866) [DL `2406.16866_Chen2024_ref-l4-revisiting-rec.pdf`]. Manual audit finds labeling errors in 14% of RefCOCO, 24% of RefCOCO+, and 5% of RefCOCOg. Error types: non-unique expressions (an expression that fits several instances), inaccurate boxes, and expression-target misalignment. CogVLM reaches 92.44% on RefCOCO. Ref-L4 itself has 45,341 annotations, 365 categories, mean expression length 24.2 words, and 22,813 unique words; 24 large models are evaluated. — [arXiv 2406.16866](https://arxiv.org/abs/2406.16866)
- **Dong et al. 2026, Ref-Adv (ICLR 2026)** (2602.23898) [DL `2602.23898_Dong2026_ref-adv.pdf`]. RefCOCO/+/g expressions are about 3 words long, images have few same-category distractors (most cases only 1), and redundant descriptors allow shortcuts. Ref-Adv uses minimally sufficient, nontrivial expressions with hard distractors and reasoning-facet tags (including negation). The released subset Ref-Adv-s has 1,142 cases, and 13 MLLMs are evaluated. Models "drop markedly" relative to RefCOCO. — [arXiv 2602.23898](https://arxiv.org/abs/2602.23898)
- **Qiu et al. 2022, RefCrowd** (2206.08172). Find the target person in a crowd by subtle attribute differences from similar persons; I did not download it. A search snippet gives 16.8 persons per image (unverified in the PDF). — [arXiv 2206.08172](https://arxiv.org/abs/2206.08172)
- **Liu et al. 2019, CLEVR-Ref+** (1901.00850) [DL `1901.00850_Liu2019_clevr-ref-plus.pdf`]. A synthetic diagnostic REC dataset. Object locations and attributes are known, and each expression is paired with a *functional program*. This allows bias control and intermediate-reasoning GT with no human annotators. Models can also be tested on false-premise (no-target) expressions. This is the methodological ancestor of programmatic GT, but it is CLEVR toy scenes, with single-target outputs. — [arXiv 1901.00850](https://arxiv.org/abs/1901.00850)

**B. Generalized (multi-target / no-target) REC and RES**
- **Liu et al. 2023, GRES / gRefCOCO** (2306.00968) [DL `2306.00968_Liu2023_gres-grefcoco.pdf`]. gRefCOCO has 278,232 expressions: 80,022 multi-target and 32,202 no-target, covering 60,287 instances in 19,994 images. Masks and boxes are given for every target. Annotation was by humans with an online tool (following ReferIt), and part of the single-target set is inherited from RefCOCO. The paper names "complex relationship modeling" as the main challenge. The ReLA baseline reaches gIoU 63.60 / cIoU 62.42 / N-acc 56.37 on val. — [arXiv 2306.00968](https://arxiv.org/abs/2306.00968)
- **He et al. 2023, GREC** (2308.16182) [DL `2308.16182_He2023_grec-grefcoco.pdf`]. The box version of the same data. It argues that top-1 Precision@0.5 is meaningless when the number of GT and predicted boxes is unbounded. It sets two metrics: Precision@(F1=1, IoU>=0.5) (the share of samples whose per-sample F1 is 1, where a predicted box is a true positive if it matches a GT box at IoU>=0.5) and No-target accuracy (N-acc). — [arXiv 2308.16182](https://arxiv.org/abs/2308.16182)
- **Ding et al. 2026, GREx** (2601.05244) [DL `2601.05244_Ding2026_grex.pdf`]. The extended (journal) version covering GRES, GREC, and a new generation task (GREG). It states that existing datasets "commonly support single-target expressions only". It reports that zero-shot MLLMs can fail to grasp which subset is meant (per the secondary search summary; unverified in the PDF text). — [arXiv 2601.05244](https://arxiv.org/abs/2601.05244)
- **Xia et al. 2023, GSVA** (2312.10103) [DL `2312.10103_Xia2023_gsva-generalized-segmentation.pdf`]. Its Table 1 shows that a model trained on single-target data collapses on empty targets: zero-shot LISA-Vicuna-7B gets gIoU 32.21 and N-acc 2.71 on gRefCOCO val. GSVA adds a rejection token; the finetuned GSVA-Llama2-13B gets gIoU 70.04, N-acc 66.02. — [arXiv 2312.10103](https://arxiv.org/abs/2312.10103)
- **Hu et al. 2023, Ref-ZOM** (2308.13853). Collects a dataset with one-to-one, one-to-many, and zero-target ("zero or more") image-text pairs, because previous methods "fail when the expressions refer to either no objects or multiple objects". Verified by abstract; I did not download it. — [arXiv 2308.13853](https://arxiv.org/abs/2308.13853)
- **Xie et al. 2023, Described Object Detection (D3)** (2307.12813). It annotates *all* described objects on all images. REC methods "struggle with confidence scores, rejecting negative instances, and multi-target scenarios". — [arXiv 2307.12813](https://arxiv.org/abs/2307.12813)
- **Schulter et al. 2023, OmniLabel** (2304.11463) [DL `2304.11463_Schulter2023_omnilabel.pdf`]. More than 28K unique descriptions on over 25K images. Descriptions refer to one, multiple, or no objects, and the metric is a modified average precision. — [arXiv 2304.11463](https://arxiv.org/abs/2304.11463)
- **Jiang et al. 2025, HumanRef / RexSeek ("Referring to Any Person")** (2503.08507) [DL `2503.08507_Jiang2025_referring-to-any-person-humanref.pdf`]. 103,028 referring statements, an average of 2.2 target instances per expression, and a 6,000-expression benchmark over six subsets. Expressions cover attributes, positions, *interactions*, reasoning, and celebrity. Metrics: Recall, Precision, DensityF1 (a prediction is correct if its IoU with any GT box exceeds a threshold). SOTA models that do well on RefCOCO/+/g "struggle with HumanRef due to their inability to detect multiple individuals". The paper notes that Molmo-7B-D outputs points. Scope is persons only. — [arXiv 2503.08507](https://arxiv.org/abs/2503.08507)
- **Jiang et al. 2025, Rex-Thinker** (2506.04034). Frames object referring as CoT over candidate instances: identify all instances of the referred category, then judge each against the expression, with abstention. It uses the HumanRef-CoT data. This is the closest method-level analogue to "enumerate candidates, then test a rule". — [arXiv 2506.04034](https://arxiv.org/abs/2506.04034)
- **Wu et al. 2026, OpenRef** (2605.25706) [DL-other `2605.25706_Wu2026_openref-open-world-rec.pdf`]. A REC benchmark with none-, single-, and multi-target samples. It uses F1 for grounding and proposes N3R for rejection reliability. — [arXiv 2605.25706](https://arxiv.org/abs/2605.25706)
- **Yang et al. 2026, RC-GRPO** (2608.04698) [DL `2608.04698_Yang2026_rc-grpo-grec.pdf`]. RL for GREC. SFT/RL that improves refusal usually degrades positive-sample localization; "Refusal-Calibrated GRPO" balances the two. This is evidence that the empty-set case needs to be a first-class test. — [arXiv 2608.04698](https://arxiv.org/abs/2608.04698)

**C. 2025-26 MLLM-era benchmarks with relational or hard-distractor expressions (single-target outputs)**
- **Li et al. 2025, GroundingME (CVPR 2026)** (2512.17495) [DL `2512.17495_Li2025_groundingme.pdf`]. 1,005 samples: Discriminative 204, Spatial 300, Limited 300, Rejection 201. The best of 25 models (Qwen3-VL-235B-A22B) gets 45.1%, and most models score 0% on Rejection. Test-time scaling adds up to 4.5%; data-mixture training lifts rejection from 0% to 27.9%. The Spatial dimension covers relational descriptions, but each query has one target. — [arXiv 2512.17495](https://arxiv.org/abs/2512.17495)
- **Gao et al. 2025, RefBench-PRO** (2512.06276) [DL `2512.06276_Gao2025_refbench-pro.pdf`]. 6,000 pairs in six tasks (Attribute, Position, Interaction, Relation, Commonsense, Rejection), 1,000 each, with a fully automated generation pipeline (MLLM-written expressions, then corrected by ten annotators). The Position and Relation tasks are relational. One target per query. — [arXiv 2512.06276](https://arxiv.org/abs/2512.06276)
- **Liu et al. 2024, FineCops-Ref** (2409.14750). Controllable difficulty over categories, attributes, and multi-hop relations, plus fine-grained edited *negative* texts and images. Single-target. — [arXiv 2409.14750](https://arxiv.org/abs/2409.14750)

### Implications for our design
- **Position on the map.** A benchmark that gives the model a subject, states a rule, and demands the exact set sits between CLEVR-Ref+ (programmatic, single-target, toy scenes) and gRefCOCO/HumanRef (set outputs, human-written free text). Real images plus rules checkable from panoptic masks and categories is the unoccupied cell.
- **Avoid the RefCOCO noise sources by construction.** Ref-L4's three error types (non-unique expressions, wrong boxes, mismatch) are exactly what a rule evaluated on COCONut masks removes. Report this as a design property, not a claim of human validation.
- **Empty and large sets must be in the mix.** GSVA shows N-acc 2.71 for a model never trained on empty targets; GroundingME shows 0% rejection for most MLLMs; RC-GRPO shows rejection and localization trade off. Include rules that legitimately return the empty set, and report empty-set accuracy separately.
- **Metric.** Adopt GREC-style set-level success (exact set / F1=1 at an IoU or point-in-mask criterion) as headline, with set-F1 as secondary. Also report a "relation-blind baseline" (for example, always answer "all objects of the subject's category") to show that overlap alone does not certify rule application (a point GeoRefer-Bench also made).
- **Reuse Ref-Adv's shortcut ablations** (bag-of-words, descriptor deletion): for us the analogue is a "rule-deleted" prompt that should drop accuracy to a computed baseline.

### Gaps
- No real-image benchmark with a *given reference object* plus a *stated rule* over a *set output*.
- No REC/RES benchmark reports accuracy as a function of the size of the correct set, or by rule family, in a controlled way; gRefCOCO only splits single/multi/no-target.
- Multi-target benchmarks (gRefCOCO, HumanRef) rely on human-written free text, which brings paraphrase variance and ambiguity (the same noise class Ref-L4 measured on RefCOCO). I found no published audit of ambiguity rates for gRefCOCO (searched, not found).

## Q2. Reasoning segmentation and implicit-query grounding: which methods and benchmarks reason relative to a given reference object and output sets?

### Takeaway
**None of the reasoning-segmentation benchmarks I found conditions on a given reference object with a stated rule.** LISA/ReasonSeg is single-target, with implicit text queries. MUSE and MMR are multi-target but text-only, and their targets are chosen by world-knowledge or function questions generated by GPT-4V ("something to cut fruit", "turn on the TV"), not by a relation to a named or marked subject. SAM 3 is the biggest practical threat on the *method* side: its Promptable Concept Segmentation returns *all* instances matching a noun phrase or an image exemplar, which already implements the "same category as the subject" rule family; and the SAM 3 Agent (MLLM plus SAM 3) handles complex language queries. 2025-26 successors (Seg-Zero, VisionReasoner, StAR, Sa2VA, Rex-Omni) improve the same single/multi-target text-query settings.

### Cited Findings

**A. Reasoning segmentation**
- **Lai et al. 2023, LISA / ReasonSeg** (2308.00692) [DL `2308.00692_Lai2023_lisa-reasoning-segmentation.pdf`]. Defines reasoning segmentation (mask from a complex implicit query). ReasonSeg has 1,218 image-instruction-mask samples (239 train, 200 val, 779 test) from OpenImages and ScanNetv2, in short-phrase and long-sentence forms. Output is a single mask via a `<SEG>` token. LISA-7B zero-shot gets gIoU 44.4 on val (table row, unverified against the main text). — [arXiv 2308.00692](https://arxiv.org/abs/2308.00692)
- **Ren et al. 2023, PixelLM / MUSE** (2312.02228) [DL `2312.02228_Ren2023_pixellm-muse.pdf`]. MUSE: 246k question-answer pairs, 0.9M instances (from 910k LVIS masks, via a GPT-4V-aided pipeline), an average of 3.7 targets per answer. Splits: 239k train / 2.8k val / 4.3k test, and the test set is split by whether the question involves fewer or more targets. This is the first multi-target reasoning-segmentation benchmark, but queries are open-world text ("How to make fruit salad?") with no reference object. — [arXiv 2312.02228](https://arxiv.org/abs/2312.02228)
- **Jang et al. 2025, MMR (ICLR 2025)** (2503.13881) [DL `2503.13881_Jang2025_mmr-multi-target-reasoning-seg.pdf`]. 194,398 question-answer pairs, 57,643 images and masks from PACO-LVIS, splits 154,127 / 8,194 / 32,077. Multi-target and part-level; an average of 1.8 targets per answer, maximum 16. Questions are generated with GPT-4V and filtered by inspectors (a pair is removed if flagged by two or more). The M2SA baseline shows that "the existing reasoning segmentation model still has room for improvement". — [arXiv 2503.13881](https://arxiv.org/abs/2503.13881)
- **Rasheed et al. 2023, GLaMM / GranD** (2311.03356). Grounded Conversation Generation; GranD has 7.5M concepts grounded in 810M regions with masks. GLaMM accepts optional region (visual prompt) input, but the tasks are captioning/conversation, not rule-based set selection. — [arXiv 2311.03356](https://arxiv.org/abs/2311.03356)
- **Liu et al. 2025, Seg-Zero** (2503.06520). Decoupled reasoning model plus SAM-style segmenter, RL with GRPO, Seg-Zero-7B gets 57.5 zero-shot on ReasonSeg. — [arXiv 2503.06520](https://arxiv.org/abs/2503.06520)
- **Liu et al. 2025, VisionReasoner** (2505.12081). Unified reasoning with "multi-object cognitive learning"; relative gains over Qwen2.5-VL of 29.1% on COCO detection, 22.1% on ReasonSeg, 13.2% on CountBench. — [arXiv 2505.12081](https://arxiv.org/abs/2505.12081)
- **Yun et al. 2026, StAR** (2603.14382). RL design-space study and the ReasonSeg-X benchmark; evaluated on ReasonSeg, MMR, and MUSE. Uses 5k training samples. — [arXiv 2603.14382](https://arxiv.org/abs/2603.14382)
- **Zhu et al. 2025, LENS** (2508.14153). RL reasoning segmentation; the search snippet gives 58.0% cIoU on ReasonSeg-test (unverified). — [arXiv 2508.14153](https://arxiv.org/abs/2508.14153)
- **Yuan et al. 2025, Sa2VA** (2501.04001). SAM-2 plus MLLM; adds Ref-SAV (over 72k object expressions, 2k manually validated video objects). Referring/conversation focus, single object per expression. — [arXiv 2501.04001](https://arxiv.org/abs/2501.04001)

**B. Concept-prompted segmentation and detection (set-producing by design)**
- **Carion et al. 2025, SAM 3 / SA-Co** (2511.16719) [DL `2511.16719_Carion2025_sam3-concepts-sa-co.pdf`]. Promptable Concept Segmentation: a short noun phrase, image exemplars, or both, returns masks *and unique identities for all matching instances*. 4M unique concept labels with hard negatives. SA-Co benchmark: 207K unique phrases, 121K images and videos, over 3M media-phrase pairs. SA-Co/Gold has three annotators per image-phrase pair (to estimate human performance). Metrics: pmF1, image-level MCC, and cgF1 = 100 x pmF1 x IL_MCC. SAM 3 reaches about 74% of estimated human performance on SA-Co/Gold and more than doubles the cgF1 of the strongest baseline. The SAM 3 Agent (MLLM calls SAM 3 with noun phrases and verifies masks) reports ReasonSeg gIoU 77.0 val / 74.0 test with Gemini 2.5 Pro and 74.6 / 70.8 with Qwen2.5-VL 72B, zero-shot. — [arXiv 2511.16719](https://arxiv.org/abs/2511.16719)
- **Zhang et al. 2023, PerSAM** (2305.03048) and **Amini-Naieni et al. 2024, CountGD** (2407.04619); **Ranjan et al. 2021, FSC-147 / "Learning To Count Everything"** (2104.08391). One-shot or exemplar-conditioned segmentation and class-agnostic counting: given a marked exemplar, find all others of that kind. CountGD accepts text, visual exemplars, or both. These solve the "same category as subject" rule, in a count-or-box form, and not any other rule. Verified by titles/abstract; not downloaded. — [arXiv 2305.03048](https://arxiv.org/abs/2305.03048), [arXiv 2407.04619](https://arxiv.org/abs/2407.04619), [arXiv 2104.08391](https://arxiv.org/abs/2104.08391)

**C. Closest to "relation to a given reference" with sets**
- **Cao et al. 2026, GeoRefer-Bench** (2609.29541) [no PDF; **paper withdrawn 2026-09-28**]. Per the abstract: relational referring segmentation in UAV imagery ("buildings north of the road"), each query is an executable logical form over a metric scene graph, scored by Exact Query Success (EQS) = predicted instance set exactly equals the denoted set. 700 scenes, 26,217 instances, 142,796 relations, 20,916 queries over five levels, 24.0% unanswerable, 2,477 counterfactual pairs. Relation-blind strategies keep non-trivial mIoU but reach at most 22.7 EQS. This is the same *evaluation idea* as ours. Treat as unreliable because of the withdrawal, and cite cautiously or not at all. — [arXiv 2609.29541](https://arxiv.org/abs/2609.29541)
- **Rex-Thinker / HumanRef** (see Q1): relational and interaction expressions over person sets, but no reference-object input.
- **Xu et al. 2024, MC-Bench** (2410.12332) [DL-other text only]. Multi-context grounding across image pairs, 2K manually labeled samples; prompts in "referring", "comparison", and "reasoning" styles (20 skills). Instance-level AP50; where a prompt names multiple groups, a Hungarian assignment matches predicted to GT groups. — [arXiv 2410.12332](https://arxiv.org/abs/2410.12332)

### Implications for our design
- **Differentiate on conditioning and verifiability, not on "multi-target".** MUSE/MMR already own "multi-target reasoning segmentation". The new element is the *subject-relative* rule with *programmatic* GT. Say this explicitly in the related-work section and include MUSE/MMR in the comparison table.
- **Treat SAM 3 as both threat and tool.** Its exemplar prompt is an off-the-shelf solver for "same category as subject". Run it as a strong specialist baseline for that rule family and show where it fails (rules involving size, touching, left-of, negation, exceptions). That turns the threat into a result.
- **Include an agentic baseline** (MLLM plus SAM 3 Agent style) so reviewers see that the benchmark is not saturated by tool use; SAM 3 Agent's high ReasonSeg scores (77.0 gIoU) suggest that single-object reasoning segmentation is nearly saturated, while rule-over-set is untested.
- **Chain-of-thought candidate enumeration** (Rex-Thinker style) is the natural reasoning-mode ablation.

### Gaps
- No reasoning-segmentation work uses a subject given by mask, box, or mark as the anchor of the query, and no work varies the *rule* while keeping the image and subject fixed.
- No reasoning-segmentation benchmark scores exact set equality (they use gIoU/cIoU, which reward partial overlap; GeoRefer-Bench argued this and then was withdrawn).
- I found no published evaluation of Seg-Zero/VisionReasoner/Sa2VA on subject-relative rules (search 2026-10-27; absence of evidence only).

## Q3. Visual prompting and region referencing: how does the way you point at an object change VLM accuracy?

### Takeaway
There are two families: (i) overlay methods that edit pixels (red circle, SoM numbered masks, boxes, arrows, scribbles), which work zero-shot on strong closed models; and (ii) region-token models (GPT4RoI, Osprey, Ferret, Ferret-v2, DAM) that inject RoI features or masks. Evidence on reference *modality* is now direct: VP-Bench compares eight visual-prompt shapes across 28 MLLMs; regular shapes (boxes, ovals, tags) beat irregular ones (masks, contours, points, scribbles). There is also strong evidence that marks can *hurt* or *destabilize*: SoM lowers open-source LLaVA-1.6-34B accuracy on What'sUp, and Feng et al. show that changing marker color, size, shape, or label placement shifts accuracy by up to about 13-15% and reorders leaderboards.

### Cited Findings

**A. Overlay-based prompting**
- **Shtedritski et al. 2023, "What does CLIP know about a red circle?"** (2304.06712) [DL `2304.06712_Shtedritski2023_red-circle-clip.pdf`]. Drawing a red circle around an object directs CLIP's attention while preserving global context. Zero-shot REC top-1 accuracy with red circles: RefCOCO val 49.8 / testA 58.6 / testB 39.9; RefCOCO+ 55.3 / 63.9 / 45.4; RefCOCOg 59.4 / 58.9 (table row). It beats the crop baseline and a ReCLIP variant on most splits. The paper also raises ethical concerns, showing red circles change the classification of persons into "criminal" categories. — [arXiv 2304.06712](https://arxiv.org/abs/2304.06712)
- **Yang et al. 2023, Set-of-Mark (SoM)** (2310.11441) [DL `2310.11441_Yang2023_set-of-mark.pdf`]. SEEM/SAM partitions the image; numbered marks (and masks/boxes) are overlaid. GPT-4V on RefCOCOg REC: 25.7 raw vs 86.4 with SoM; RES 75.6 mIoU (table 2; evaluated on sub-sampled validation sets because of the GPT-4V interface). Mark type matters: on Flickr30K, Number&Mask gives R@1 84.4 and Number&Mask&Box gives 89.2 (table 3). Using ground-truth masks instead of MaskDINO on RefCOCOg raises mIoU from 75.6 to 90.1 (+14.5), so the segmenter is a major error source; the authors also observed that many failures with GT masks "are not the GPT-4V's problem, but the annotation itself is ambiguous or incorrect". The mark-placement algorithm handles overlaps by placing marks for smaller regions first inside distance-transform maxima. — [arXiv 2310.11441](https://arxiv.org/abs/2310.11441)
- **Yan et al. 2024, SoM-LLaVA ("List Items One by One")** (2404.16375). Open-source MLLMs do not get SoM for free; the authors train on a tag-listing task to teach numeric tags. Verified by title only; I read no numbers from it. — [arXiv 2404.16375](https://arxiv.org/abs/2404.16375)
- **Cai et al. 2023, ViP-LLaVA / ViP-Bench** (2312.00784) [DL `2312.00784_Cai2023_vip-llava-vip-bench.pdf`]. Overlays arbitrary visual prompts (arrow, box, circle, scribble) directly on the RGB image with no region encoder; strong on Visual7W, PointQA, VCR. ViP-Bench has 303 image-question pairs (from MM-Vet, MMBench, Visual Genome) with synthesized (tight boxes) and human-drawn prompts, across six capabilities including object-relationship reasoning. — [arXiv 2312.00784](https://arxiv.org/abs/2312.00784)

**B. Systematic comparisons of reference modality and evidence that marks can hurt**
- **Xu et al. 2025, VP-Bench (AAAI 2026)** (2511.11438) [DL `2511.11438_Xu2025_vp-bench.pdf`]. Stage 1: 30k visualized prompts, eight shapes (tag, bounding box, arrow, mask, contour, oval, point, scribble), 355 attribute combinations; Stage 2: six downstream tasks. 28 MLLMs. Findings:
  - Regular shapes (tag, arrow, box, oval) are recognized at about 80% average accuracy, versus under 70% for irregular ones (mask, point, scribble). Models also show a preference for boxes and tags and a heightened sensitivity to red prompts.
  - Adding a text description of the prompt shape helps a lot for masks and points: InternVL3-78B gains +29.24% (mask) and +12.38% (point); Qwen2.5-VL-72B gains +27.3% (mask) but -3.05% (point). For arrows and contours the gain is under 1.4%.
  - The prompt a model perceives best is usually, but not always, the best for downstream tasks: DeepSeek-VL2's best-VP choice is -19.43% on SeeClick versus a random-best-VP choice.
  - Training on VP data (ViP-LLaVA) lifted Stage-1 accuracy (64.33% -> 73.01% for 7B) but did not help downstream tasks and degraded at 13B.
  — [arXiv 2511.11438](https://arxiv.org/abs/2511.11438)
- **Feng et al. 2025, "Visually Prompted Benchmarks Are Surprisingly Fragile"** (2512.17875) [DL-other `2512.17875_Feng2025_visually-prompted-benchmarks-fragile.pdf`]. Nine VLMs on two BLINK-derived visually prompted tasks (relative depth, semantic correspondence). Changing marker size, shape, color, or label position swings Qwen2.5-VL accuracy by up to 13% on BLINK relative depth and can reorder the leaderboard. Slightly enlarging the marker puts InternVL3-8B alongside or above Gemini 2.5 Pro. JPEG compression level in API calls also changes the lineup. They release VPBench with 16 marker variants. — [arXiv 2512.17875](https://arxiv.org/abs/2512.17875)
- **Wan et al. 2024, Contrastive Region Guidance (CRG)** (2403.02325) [DL `2403.02325_Wan2024_contrastive-region-guidance.pdf`]. LLaVA-1.6-34B on What'sUp (Indiv / Pairs / Set of 4): original image 86.8 / 76.8 / 54.0; overlay with boxes 82.2 / 69.4 / 39.7; with masks 84.2 / 82.7 / 49.6; with SoM 77.9 / 63.9 / 33.0. The authors conclude SoM "does not transfer well to open-source VLMs" and that overlays do not force attention to the marked region. — [arXiv 2403.02325](https://arxiv.org/abs/2403.02325)
- **Luo et al. 2025, "Connecting the Dots" (training-free grounding)** (2511.19516). Reports that SoM and overlaying boxes/masks do not help the model it tests; occlusion and detector merging are failure sources. Search snippet and text read; numbers not extracted. — [arXiv 2511.19516](https://arxiv.org/abs/2511.19516)
- **Wu et al. 2024, Visual Prompting in MLLMs: A Survey** (2409.15310). Survey of visual-prompt types and region-referencing methods; used only as an index. — [arXiv 2409.15310](https://arxiv.org/abs/2409.15310)

**C. Region-token and mask-conditioned models**
- **Zhang et al. 2023, GPT4RoI** (2307.03601). Replaces `<region>` references in the instruction with RoI features; 81.6% on VCR (second place 75.6%). — [arXiv 2307.03601](https://arxiv.org/abs/2307.03601)
- **Yuan et al. 2023, Osprey** (2312.10032). Mask-text instruction tuning (724K samples) with a mask-aware visual extractor. — [arXiv 2312.10032](https://arxiv.org/abs/2312.10032)
- **You et al. 2023, Ferret** (2310.07704) and **Zhang et al. 2024, Ferret-v2** (2404.07973). Hybrid region representation (discrete coordinates plus continuous features) accepting points, boxes, and free-form shapes; GRIT has 1.1M samples, including 95K hard negatives. — [arXiv 2310.07704](https://arxiv.org/abs/2310.07704), [arXiv 2404.07973](https://arxiv.org/abs/2404.07973)
- **Lian et al. 2025, Describe Anything (DAM)** (2504.16072) [DL `2504.16072_Lian2025_describe-anything.pdf`]. A mask-conditioned model with a *focal prompt* (high-resolution crop of the region) and a localized vision backbone; DLC-Bench evaluates localized captioning without reference captions. — [arXiv 2504.16072](https://arxiv.org/abs/2504.16072)
- **Cheng et al. 2024, SpatialRGPT** (2406.01584) [DL-other `2406.01584_Cheng2024_spatialrgpt-grounded-spatial-reasoning.pdf`]. Region-referenced spatial reasoning (distance, direction, size between regions), the closest region-conditioned *relation* work; it answers questions about a region pair and does not output sets. — [arXiv 2406.01584](https://arxiv.org/abs/2406.01584)

### Implications for our design
- **Make the reference modality an explicit independent variable** (text only, mask overlay, contour, box, numbered mark, point) crossed with rule family. VP-Bench shows modality alone moves accuracy by tens of points and is model-dependent. Our benchmark would be the first to measure this effect on a *downstream reasoning* task of this kind (VP-Bench Stage 2 uses six unrelated domain tasks).
- **Budget for fragility.** Feng et al. mean any single rendering style (color, thickness, label offset, JPEG level) is a confound. Fix a documented default, run a small sweep of styles (at least color, size, label position) on a subset, report rank stability (for example Kendall tau), and use lossless images.
- **Include a prompt-side description of the reference format** (VP-Bench: +29% for masks). Report with and without it, as a controlled factor.
- **SoM can hurt open-source models** (CRG: -8.9 on Indiv, -12.9 on Pairs, -21.0 on Set of 4 against original on What'sUp for SoM). So "marks" arms should be reported per model, not pooled, and should not be interpreted as a measure of reasoning alone. A text-only subject arm (class and attributes) is the clean control.
- **Mark placement and overlap matter.** SoM's own algorithm exists because center-placed numbers collide in dense scenes; COCO images are dense. Document the placement rule and report accuracy by object size and occlusion.

### Gaps
- No study crosses reference modality with *downstream relational rules*; VP-Bench tests perception and six domain tasks, and Feng et al. test depth and correspondence.
- Evidence that marks bias models toward answering about the marked object (or toward labeled objects) is scattered. I found no direct measurement of "mark-ID hallucination" (naming an ID that does not exist) or of "marked-object preference" in set-output tasks.
- No evidence on whether the effect of modality persists in thinking-mode models (VP-Bench does not stratify by reasoning mode).

## Q4. Output formats for localization from VLMs, and how benchmarks score set outputs

### Takeaway
Four output formats are in use: (1) mark IDs over an overlaid partition (SoM), (2) points (Molmo, Qwen3-VL, Gemini), (3) boxes (Qwen3-VL, Gemini, Rex-Omni), and (4) text labels or phrases fed to a segmenter (SAM 3 Agent, SADL's OD track). Scoring is unsettled for sets: Hungarian matching plus IoU threshold (MC-Bench, SADL), per-sample F1=1 (GREC), point-in-mask (PointArena), mask IoU on the union (GRES), and exact set equality (GeoRefer-Bench, withdrawn). Known pitfalls: models disagree on coordinate conventions, some frontier models cannot emit usable coordinates, duplicate boxes, segmenter errors on small objects, and GT mask or label granularity.

### Cited Findings

**A. Formats and their conventions**
- **Deitke et al. 2024, Molmo / PixMo** (2409.17146) [DL `2409.17146_Deitke2024_molmo-pixmo-points.pdf`]. Introduces a 2D *pointing* dataset (PixMo-Points) collected without external VLMs; PixMo-Count has 540-image val/test sets with 120 verified samples per count from 2 to 10, and its training set is annotated with object-center points. Molmo-72B outperforms Claude 3.5 Sonnet and Gemini 1.5 Pro on their evaluation set. — [arXiv 2409.17146](https://arxiv.org/abs/2409.17146)
- **Cheng et al. 2025, PointArena / Point-Bench** (2505.09990) [DL `2505.09990_Cheng2025_pointarena.pdf`]. 982 image-question pairs in five categories (Spatial, Affordance, Counting, Steerable, Reasoning), each with pixel-level target masks. The model predicts a *set* of image-space points; a point counts as correct if it lies inside a GT mask. Molmo-72B is the best model, and proprietary models are comparable; targeted pointing supervision helps, and added language reasoning does not improve pointing (secondary summary, unverified in the PDF). — [arXiv 2505.09990](https://arxiv.org/abs/2505.09990)
- **Bai et al. 2025, Qwen3-VL Technical Report** (2511.21631) [DL `2511.21631_Bai2025_qwen3-vl.pdf`]. Box and *point-based* grounding data (PixMo points plus synthesized points) and three counting formulations (direct, box-based, point-based). Qwen3-VL switches to coordinates normalized to [0, 1000] (Qwen2.5-VL used absolute coordinates) for robustness to resolution and aspect ratio. — [arXiv 2511.21631](https://arxiv.org/abs/2511.21631)
- **Jiang et al. 2025, Rex-Omni** (2510.12798) [DL `2510.12798_Jiang2025_rex-omni-next-point.pdf`]. Names MLLM detection pitfalls: low recall, *duplicate predictions*, coordinate misalignment. Quantizes coordinates to 0-999 tokens and uses GRPO with geometry-aware rewards; also supports pointing and visual prompting. — [arXiv 2510.12798](https://arxiv.org/abs/2510.12798)
- **GroundingME** (2512.17495; see Q1). Gemini-2.5 is "significantly more receptive" to [y1, x1, y2, x2] order, so the prompt format was changed per model family. The authors omit GPT-5, Claude-Sonnet-4.5, and Grok-4 from the reported results "due to issues with their output" (unusable coordinates). So box output cannot be a universal protocol across frontier models. — [arXiv 2512.17495](https://arxiv.org/abs/2512.17495)
- **SoM** (2310.11441; see Q3). Output is a list of mark IDs, which is robust to coordinate conventions but couples the result to the quality of the upstream partition.

**B. Scoring sets**
- **GREC / GRES** (2308.16182, 2306.00968): Precision@(F1=1, IoU>=0.5), N-acc, gIoU, cIoU (see Q1). GSVA's table defines gIoU as per-example IoU averaged, cIoU as accumulated over the dataset. — [arXiv 2308.16182](https://arxiv.org/abs/2308.16182)
- **MC-Bench** (2410.12332): AP50 at the instance level with Hungarian matching of predicted groups to GT groups. — [arXiv 2410.12332](https://arxiv.org/abs/2410.12332)
- **OmniLabel** (2304.11463): modified AP for one/multiple/no-object descriptions. **HumanRef** (2503.08507): Recall/Precision/DensityF1.
- **SA-Co / SAM 3** (2511.16719): pmF1 gated by image-level MCC (cgF1), with three annotators per phrase and an "oracle" score taken against all annotations to handle ambiguity.
- **SADL (own prior work)**: OD track converts model text labels to masks with SAM 3, then Hungarian matching at IoU@0.5 against GT masks decides TP/FP/FN; this exposes the pipeline to segmenter failure (local file `2606.30393_SADL_original.pdf`).

**C. Pitfalls with direct evidence**
- **Segmenter as hidden error source**: SoM shows +14.5 mIoU when GT masks replace the segmenter (75.6 -> 90.1). "Connecting the Dots" (2511.19516) reports degraded performance on sub-5% referents. Rex-Omni and RexSeek note duplicates and under-prediction of multiple instances.
- **Label granularity and GT ambiguity**: Ref-L4 (non-unique expressions), SoM (ambiguous "golden" annotations), SA-Co (three-annotator ambiguity handling).
- **Format sensitivity**: Feng et al. on marker design; GroundingME on coordinate order; VP-Bench on prompt shape.

### Implications for our design
- **Use GT from the panoptic masks directly, and score outputs without a learned segmenter in the loop.** For mark-ID outputs, scoring is a set comparison of IDs (the GT partition is COCONut's own). For point outputs, use PointArena's point-in-mask test against the GT masks, then match points to distinct objects one-to-one (a point set can otherwise hit one big object many times). For box outputs, use Hungarian matching at IoU>=0.5 against GT boxes derived from masks. This removes SADL's SAM 3 failure mode on small objects.
- **Offer two tracks.** A *mark-ID* track works for every model, including those that cannot emit coordinates (GroundingME). A *native-coordinate* track (Qwen3-VL [0,1000] boxes/points, Gemini [y,x] order, Molmo points) gives a grounding-specialist view. Tell the model the coordinate convention in the prompt, and report invalid-output rate per model as a metric.
- **Headline metric = exact-set success; secondary = set-F1 and over/under-selection rates.** Report duplicate-collapse rules for boxes and points explicitly (Rex-Omni's duplicate problem).
- **Small and crowd objects.** Define a minimum object area for inclusion in candidate sets (and treat COCO `iscrowd` regions separately), and report accuracy by object-size bin so small-object failures are a stratified result and not noise.

### Gaps
- No agreed standard for set-level scoring of point outputs (one-to-one matching, penalty for extra points); PointArena's per-point criterion does not penalize hitting the same object twice (reading of the rule as stated; I did not test it).
- I found no systematic comparison of the *same* task across ID, point, box, and mask output formats for the same models.
- No public measurement of how much a pipeline segmenter (SAM 3, SAM 2) changes set-level scores relative to GT masks beyond SoM's single RefCOCOg datapoint.

## Closest prior work / novelty threats

| Paper | What overlaps | What it lacks vs our design |
|---|---|---|
| **SAM 3 / SA-Co + SAM 3 Agent** (2511.16719) | Returns all matching instances of a concept; image-exemplar prompts implement "same category as subject"; Agent handles complex queries | Concept-only, no relation to a referenced subject; no rule families (size, touch, left-of), no exceptions/overrides; not a VLM-only task; no exact-set scoring |
| **gRefCOCO / GRES / GREC / GREx** (2306.00968, 2308.16182, 2601.05244) | Multi-target and no-target outputs on COCO images; set metrics (F1=1, N-acc) | Human-written free-text expressions (ambiguity); no reference-object input; no programmatic rule; no reference-modality factor |
| **MUSE (PixelLM) and MMR** (2312.02228, 2503.13881) | Multi-target reasoning segmentation, large-scale (246k / 194k), LVIS/PACO masks | GPT-4V-generated implicit text queries, no given subject or rule, mask-overlap metrics (gIoU/cIoU) rather than exact set |
| **HumanRef / RexSeek / Rex-Thinker** (2503.08507, 2506.04034) | Multi-instance referring including interaction and spatial relations; candidate-then-verify CoT | Persons only; free-text; no reference-object input; no programmatic checkability |
| **GeoRefer-Bench (withdrawn)** (2609.29541) | Executable logical forms, exact instance-set match, relation-defined referent sets, relation-blind baseline | Aerial imagery; withdrawn 2026-09-28 (cannot be relied on); no reference-modality factor; no COCO-style natural images |
| **CLEVR-Ref+** (1901.00850) | Programmatic expression-to-GT binding, bias control, no-target tests | Synthetic CLEVR scenes, single target, no VLM-era evaluation |
| **VP-Bench; Feng et al. VPBench; ViP-Bench** (2511.11438, 2512.17875, 2312.00784) | Systematic comparison of reference modalities; fragility to marker style | Perception or single-answer questions, not set outputs under rules; no subject-relative reasoning |
| **GroundingME; RefBench-PRO; Ref-Adv** (2512.17495, 2512.06276, 2602.23898) | 2025-26 relational/hard-distractor/rejection REC for MLLMs | Single target per query; text-only reference; human or MLLM-authored expressions |
| **PointArena / Point-Bench** (2505.09990) | Point-set outputs scored by point-in-mask on GT masks; spatial category | 982 items, text-only queries, no reference object or rule structure |
| **SADL original** (2606.30393) | Same task motivation, subject-conditioned set output, Hungarian IoU@0.5 | Fuzzy 5-factor taxonomy, small, kappa 0.53, segmenter in the loop |

## Bibliography

Verified on arXiv 2026-10-27 (title, first author, year). [DL] = downloaded this session; [DL-other] = present from other researchers; no marker = abstract verified only.
1. Yu et al. 2016, Modeling Context in Referring Expressions (RefCOCO/+), 1608.00272
2. Mao et al. 2015, Generation and Comprehension of Unambiguous Object Descriptions (RefCOCOg), 1511.02283
3. Liu et al. 2023, GRES: Generalized Referring Expression Segmentation, 2306.00968 [DL]
4. He et al. 2023, GREC: Generalized Referring Expression Comprehension, 2308.16182 [DL]
5. Ding et al. 2026, GREx, 2601.05244 [DL]
6. Xia et al. 2023, GSVA, 2312.10103 [DL]
7. Hu et al. 2023, Beyond One-to-One (Ref-ZOM), 2308.13853
8. Qiu et al. 2022, RefCrowd, 2206.08172
9. Chen et al. 2024, Revisiting REC Evaluation (Ref-L4), 2406.16866 [DL]
10. Dong et al. 2026, Ref-Adv, 2602.23898 [DL]
11. Xie et al. 2023, Described Object Detection (D3), 2307.12813
12. Schulter et al. 2023, OmniLabel, 2304.11463 [DL]
13. Liu et al. 2019, CLEVR-Ref+, 1901.00850 [DL]
14. Jiang et al. 2025, Referring to Any Person (HumanRef/RexSeek), 2503.08507 [DL]
15. Jiang et al. 2025, Rex-Thinker, 2506.04034
16. Jiang et al. 2025, Detect Anything via Next Point Prediction (Rex-Omni), 2510.12798 [DL]
17. Wu et al. 2026, Towards Open-World REC (OpenRef), 2605.25706 [DL-other]
18. Yang et al. 2026, Teaching MLLMs to Say No (RC-GRPO), 2608.04698 [DL]
19. Li et al. 2025, GroundingME, 2512.17495 [DL]
20. Gao et al. 2025, RefBench-PRO, 2512.06276 [DL]
21. Liu et al. 2024, FineCops-Ref, 2409.14750
22. Lai et al. 2023, LISA, 2308.00692 [DL]
23. Ren et al. 2023, PixelLM (MUSE), 2312.02228 [DL]
24. Jang et al. 2025, MMR, 2503.13881 [DL]
25. Rasheed et al. 2023, GLaMM, 2311.03356
26. Liu et al. 2025, Seg-Zero, 2503.06520
27. Liu et al. 2025, VisionReasoner, 2505.12081
28. Yun et al. 2026, StAR, 2603.14382
29. Zhu et al. 2025, LENS, 2508.14153
30. Yuan et al. 2025, Sa2VA, 2501.04001
31. Carion et al. 2025, SAM 3, 2511.16719 [DL]
32. Zhang et al. 2023, Personalize SAM, 2305.03048; Amini-Naieni et al. 2024, CountGD, 2407.04619; Ranjan et al. 2021, Learning To Count Everything, 2104.08391
33. Cao et al. 2026, GeoRefer-Bench, 2609.29541 (WITHDRAWN 2026-09-28; no PDF)
34. Xu et al. 2024, MC-Bench, 2410.12332 (text read from other researcher's extraction; not in the papers folder)
35. Shtedritski et al. 2023, What does CLIP know about a red circle?, 2304.06712 [DL]
36. Yang et al. 2023, Set-of-Mark Prompting, 2310.11441 [DL]
37. Yan et al. 2024, List Items One by One (SoM-LLaVA), 2404.16375
38. Cai et al. 2023, ViP-LLaVA, 2312.00784 [DL]
39. Xu et al. 2025, VP-Bench, 2511.11438 [DL]
40. Feng et al. 2025, Visually Prompted Benchmarks Are Surprisingly Fragile, 2512.17875 [DL-other]
41. Wan et al. 2024, Contrastive Region Guidance, 2403.02325 [DL]
42. Luo et al. 2025, Connecting the Dots: Training-Free Visual Grounding via Agentic Reasoning, 2511.19516
43. Wu et al. 2024, Visual Prompting in MLLMs: A Survey, 2409.15310
44. Zhang et al. 2023, GPT4RoI, 2307.03601
45. Yuan et al. 2023, Osprey, 2312.10032
46. You et al. 2023, Ferret, 2310.07704; Zhang et al. 2024, Ferret-v2, 2404.07973
47. Lian et al. 2025, Describe Anything, 2504.16072 [DL]
48. Cheng et al. 2024, SpatialRGPT, 2406.01584 [DL-other]
49. Deitke et al. 2024, Molmo and PixMo, 2409.17146 [DL]
50. Cheng et al. 2025, PointArena, 2505.09990 [DL]
51. Bai et al. 2025, Qwen3-VL Technical Report, 2511.21631 [DL]
52. Zheng et al. 2026, GeM-VG, 2601.04777 (read abstract only; not cited in findings)

Not found: arXiv ID for HC-RefLoCo (Wei et al., NeurIPS 2024 D&B) was not located; omitted. No standalone paper for "RefSAM-type" methods was cited because none I examined takes a reference object as an anchor.
