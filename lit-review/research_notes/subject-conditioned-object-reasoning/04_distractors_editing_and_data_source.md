# Distractor Detection, Subject-Conditioned Removal / Editing Benchmarks, and the COCONut-PanCap Data Source

Scope note: every arXiv ID below was checked on 2026-10-27 against its arxiv.org/abs page (title, first author, and date match; a small script read the `citation_title` / `citation_author` / `citation_date` meta tags). PDFs are under `lit-review/papers/` (PDFs are not tracked in git; see `lit-review/README.md`). Files marked [DL] were downloaded in this session. Files marked [no-DL] were verified but not downloaded. Non-arXiv works (Fried 2015, Aberman 2022 CVPR version, IEEE TMM 2018) were checked against CVF, Princeton, and Crossref pages instead. Anything I could not confirm is marked "(unverified)". Anchor: the rejected SADL paper (arXiv 2606.30393, 1,800 cases / 1,000 photos / 7 VLMs). Its related work cites only classical saliency, Fried 2015, SimpSON, InstructPix2Pix, and SAM. The 2023-2026 editing and removal literature below is therefore absent from it. **Reference-hygiene warning for the rebuild:** the SADL reference list has at least these defects, all of which a reviewer can see.
- Several entries carry placeholder IDs (`arXiv:2501.XXXXX` for "Bae, iDIS", Molmo2, and "Kumain, Deep Visual Saliency"; `2503.XXXXX`, `2502.XXXXX`, `2505.XXXXX`, `2412.XXXXX`).
- Fried et al. 2015 is listed with "Aaron Hertzmann" as the 4th author. The CVPR page lists Fried, Shechtman, Goldman, **Finkelstein**.
- SimpSON is listed as "Tri Huynh ... Single-Click Distractors Removal Using Vision Transformer". arXiv 2305.17624 is titled "SimpSON: Simplifying Photo Cleanup with Single-Click Distracting Object Segmentation Network", first author **Chuong Huynh**.
- "iDIS" is arXiv 2511.21397 (first author Jiyun Bae, not "Juyeon").

## Q1. What does the distractor detection / removal line evaluate, and does any of it condition on a subject?

### Takeaway
The line is small and has been quiet since 2023. Three families exist.
1. **Human-annotated "what distracts" detectors** (Fried 2015, SimpSON 2023). Ground truth comes from crowd workers or professional photographers marking "distracting" regions. The score is AP/PR on those marks.
2. **Saliency-reduction editing** (Aberman 2022, Miangoleh 2023). A pretrained saliency model is the objective, and success is saliency drop inside a mask plus a realism or user study.
3. **VLM-planned decluttering** (Progressive Photorealistic Simplification 2026, PhotoAgent 2026). A VLM picks what to remove or edit.

Only family 3 reasons about objects at all, and none conditions on a user-given subject plus a stated rule. Fried 2015 itself concedes that the subject confounds saliency. Recent "distractor" papers in VLM robustness (iDIS, Distract-Bench) and NeRF/NVS (IDDR-NGP, DF3DV-1K) use the word in a different sense. They are not an identification task.

### Cited Findings

**A. Classic detection datasets**
- **Fried, Shechtman, Goldman, Finkelstein 2015, "Finding Distractors in Images"** (CVPR 2015, pp. 1703-1712, no arXiv version; [CVF page](https://openaccess.thecvf.com/content_cvpr_2015/html/Fried_Finding_Distractors_In_2015_CVPR_paper.html)) [no-DL; text read from the Princeton PDF]. Two datasets.
  - A Mechanical Turk set: 403 images, 27.8 annotations per image on average (1,073 images / 11,244 annotations in the first-pass collection).
  - A mobile-app set: 376 images, 1 annotation each.
  - The model predicts a distractor map and the paper uses it for inpainting-based removal.
  - Key text: saliency cannot define distractors because "the main subject ... where people look first usually has a high saliency value", and distractors are looked at "after longer periods of time". The paper never conditions on a stated subject.
- **Huynh et al. 2023, SimpSON** (2305.17624, CVPR 2023) [DL `2305.17624_Huynh2023_simpson-photo-cleanup.pdf`]. One click selects a distractor, a Click Proposal Network finds similar instances, and a verification module filters them.
  - Data: Distractor20K, 21,821 photos (20,790 train with 178,815 instances; 1,031 val with 8,956 instances), masked by 3 professional photographers "that affect its overall aesthetic appeal". Plus synthetic sets (DistractorSyn14K) built from LVIS instances. Pretraining uses COCO Panoptic (118,287 images) and LVIS.
  - Evaluation: AP, APs/APm/APl, with and without click embedding (R101: 30.7 AP without, 35.5 with).
  - It is subject-agnostic (click-based) and its definition is aesthetic.
- **Zhang et al. 2018, "Detecting and Removing Visual Distractors for Video Aesthetic Enhancement"** (IEEE Trans. Multimedia, DOI 10.1109/tmm.2018.2790163 via Crossref; authors Zhang, Wu, Li, Wang, Zheng, Hu) [no-DL]. This is the video-distractor line SADL cites. Given names beyond the family name were not verified.

**B. Saliency-guided editing**
- **Aberman et al. 2022, "Deep Saliency Prior for Reducing Visual Distraction"** (2109.01980, CVPR 2022) [DL `2109.01980_Aberman2021_deep-saliency-prior.pdf`]. It back-propagates through a pretrained saliency model to drive recolor / warp / GAN-replace operators inside a *user-provided mask*. The metric is saliency reduction in the region plus a user study. There is no object reasoning and the mask is given.
- **Miangoleh et al. 2023, "Realistic Saliency Guided Image Enhancement"** (2306.06092, CVPR 2023) [DL `2306.06092_Miangoleh2023_realistic-saliency-guided-enhancement.pdf`]. It adds a learned realism network to saliency attenuation / amplification of a masked region. Evaluation uses professional photographers. The mask is again an input.
- "Distraction is All You Need" (attention-guided distractor removal): **I could not find a paper by this title** via web search or arXiv API (only patents and the works above). Treat it as unverified and do not cite it.

**C. 2024-2026 distractor-flavoured work**
- **Rosenthal et al. 2026, "Progressive Photorealistic Simplification"** (2605.10409) [DL `2605.10409_Rosenthal2026_progressive-photorealistic-simplification.pdf`]. The closest published system to the "decide what to remove" step.
  - A Gemini 3 Pro planner runs a Select-Remove-Verify loop. A fixed 4-level taxonomy orders removal: Distracting, Secondary, Primary-Subject, Background.
  - Its test set is 93 Open Images photos, with 3 raters and 530 validated objects in 68 images (7.8 objects per image). Reported ordering consensus: across-category "Distractor" removed first in 100% of cases, "Secondary" before "Primary" in 96.1%. Within-category pairs are ambiguous, with "doesn't matter" chosen 27.5% of the time (N about 1,600 pairs), against 3.7% for across-category pairs (N about 500).
  - It scores removal *order* (modified Kendall tau-b) on generated videos. It is **not** subject-conditioned: "primary subject" is a class, not a user input. The ground truth is human preference.
  - Note for our motivation: the paper itself reports that within-category order is only weakly determined, which supports "static taxonomies are ill-posed".
- **Yao et al. 2026, PhotoAgent** (2602.22809) [DL `2602.22809_Yao2026_photoagent.pdf`]. Autonomous editing as long-horizon planning with tree search. It ships UGC-Edit: 7,000 photos plus a learned aesthetic reward model, and a 1,017-photo test set. Evaluation is aesthetic score and instruction adherence. Distraction removal is one of many actions (from the project page; the paper abstract does not single it out).
- **Ye et al. 2026, Agent Banana** (2602.09084) [no-DL]. A planner-executor with layer isolation that resists over-editing. Not a distractor benchmark. Relevant as "edit planning" prior art.
- **Bae et al. 2025, iDIS / "Understanding the Effects of Distractors on Reasoning VLMs"** (2511.21397) [DL `2511.21397_Bae2025_distractors-reasoning-vlms.pdf`]. A VQA dataset that varies *added* distractors along semantic and numerical axes. Visual distractors reduce accuracy without lengthening reasoning. Here a distractor is a confound, not an identification target.
- **Sun et al. 2026, Distract-Bench** (2606.08894) [no-DL]. Semantic visual distractions are added while "preserving the ground-truth answer". 8 open and 2 closed VLMs. Reasoning VLMs are less robust to semantic distractions than their non-reasoning bases.
- Different sense of "distractor" (transient objects in 3D): IDDR-NGP (2601.11030) and DF3DV-1K (2604.13416, [no-DL, title only seen in search]). They should not be confused with this line.

### Implications for our design
1. Position the benchmark against the *selection step* that every remover assumes. SimpSON needs a click, Aberman needs a mask, and Progressive Simplification has a fixed taxonomy. None takes (subject, rule) and returns a set.
2. Drop "distractor" as the label (it carries the aesthetic, human-centred definition that sank SADL). Use a neutral name such as "rule-selected objects" so that ground truth is a *checkable predicate* over COCONut masks (spatial relation, size ratio, overlap, category, count), not a human judgment.
3. Cite Fried 2015 and Progressive Simplification only as evidence that labels in this area are subjective (27.5% "doesn't matter" within a category). That justifies moving to prompt-specified rules without making human-attention claims. Fried's observation that saliency and distraction come apart is usable as motivation for the temporal-dynamics paper.
4. Use set-level metrics (precision/recall/F1 over mask IDs, exact-set match, plus a false-removal rate for "must keep" objects). SimpSON's AP over instances is the nearest prior metric.

### Gaps
- No benchmark evaluates VLMs on subject-conditioned *set* selection with programmatic ground truth.
- No distractor paper reports per-rule breakdowns or hard negatives.
- I found no independent 2024-2026 follow-up to SimpSON besides Progressive Simplification and SADL, and no photo-cleanup benchmark with VLM baselines.
- Video distractors: only the 2018 TMM paper surfaced. I did not find a 2024-2026 successor.

## Q2. Object removal and inpainting: methods, benchmarks, and the "which object?" gap

### Takeaway
The removal literature (2024-2026) is overwhelmingly about *how* to erase a given mask (shadow / reflection side effects, background consistency). Targets arrive as masks or short object names. The one paper that reasons about *what else* must go (ICOR / REORM, 2026) uses 110 evaluation items and pixel metrics. Nobody evaluates the choice of removal targets under a relative rule.

### Cited Findings

- **Winter et al. 2024, ObjectDrop** (2403.18818) [DL `2403.18818_Winter2024_objectdrop.pdf`]. A counterfactual dataset of 2,500 photo pairs (scene before and after physically removing one object), then bootstrap supervision: it filters about 700,000 images to synthesize more for insertion. A 51-triplet held-out set was captured after the project. Mask given, human preference evaluation.
- **Li et al. 2025, RORem** (2501.00740, CVPR 2025) [DL `2501.00740_Li2025_rorem.pdf`]. Human-in-the-loop: 60K seed pairs, a discriminator trained on human selections, iterated to over 200K pairs. Reported gain: "more than 18%" success rate over prior removers, and 4-step distilled inference (under 1 s). Success rate is judged by a learned or human discriminator, not by object selection.
- **Wei et al. 2025, OmniEraser** (2501.07397) [DL `2501.07397_Wei2025_omnieraser.pdf`]. Video4Removal: 134,281 photorealistic 1920x1080 triplets from paired video frames; RemovalBench: 70 crafted pairs. Mask given.
- **Jiang et al. 2025, SmartEraser** (2501.08279) [DL `2501.08279_Jiang2025_smarteraser.pdf`]. Masked-Region Guidance plus Syn4Removal (about 1 million synthetic triplets from copy-paste of instance masks, so the "answer" is known by construction). Mask given.
- **Liu et al. 2025, Erase Diffusion** (2503.07026) [DL `2503.07026_Liu2025_erase-diffusion.pdf`]. A diffusion-pathway calibration method. Mask given.
- **Zhao et al. 2025, ObjectClear** (2505.22636) [no-DL]. OBER dataset, target-aware attention that separates removal from background reconstruction. **Zhu et al. 2025, GeoRemover** (2509.18538) [no-DL]. Removing objects and their causal artifacts via geometry. **Sun et al. 2024, Attentive Eraser** (2412.12974, AAAI 2025) [no-DL] is training-free attention redirection. PANDORA (2603.27555), DORS (2607.16656; dense scenes where similar instances confuse attention) and AdaEraser (2605.15921, ID seen in search only, unverified) are 2026 training-free variants. All take a mask.
- **Tudosiu et al. 2024, MULAN** (2404.02790) [no-DL]. A multi-layer annotated (RGBA) dataset built from COCO-derived images via automatic instance decomposition. It is useful as an example of deriving layer/occlusion structure from COCO. (ROVI, which I intended to check, was not located; unverified.)
- **Huang et al. 2026, ICOR / REORM** (2602.01298) [DL `2602.01298_Huang2026_interaction-consistent-object-removal.pdf`]. The only found paper where an MLLM decides *which additional elements* must be removed alongside the named target (lighting effects, connected objects, produced elements, contextually linked objects). The instruction names only the target and "do[es] not describe interaction details". ICOREval has **110 examples** (an initial version had 72), built from RORD, RemovalBench, synthetic Nano Banana insertions, and copy-paste composites. Metrics are DINO, LPIPS, PSNR, SSIM against a ground-truth output image (pixel metrics). GPT-4o is both "MLLM Simulator" and "Examiner". REORM beats GPT Image 1 and Nano Banana on all four metrics. It is a set-selection-flavoured task but with a causal/physical, not relative-rule, definition.
- **Rosenthal et al. 2026** (2605.10409, see Q1) is the other VLM "pick the next object to remove" system.
- No benchmark found for instructions like "remove everything except the main subject" or "remove the object the person is holding" with set-level scoring. Closest hits were instruction editors that include a removal task (CompBench "Object Removal"; I2EBench2.0 has an object-removal dimension; UnicEdit-10M treats leftover background as failure in extraction/removal edits, from the search summary, unverified in full text).

### Implications for our design
1. Position as the missing *front half* of removal: all of ObjectDrop / RORem / OmniEraser / SmartEraser / Erase Diffusion assume the mask, and their benchmarks (51, 70, 110 items) are small and judged on pixels. Our benchmark is large and judged on *which masks*.
2. The edit-planning application framing is defensible: ICOR shows that "which elements go" is an open reasoning problem and that MLLM reasoning helps. We differ by (a) the rule is relative to a user-named subject, (b) the answer is a set over panoptic IDs, (c) no inpainter is needed to score.
3. Use the removal literature for an *optional downstream validation*: feed gold vs predicted sets to one open remover (e.g. a RORem-class model) and report that wrong sets leave or destroy objects. Keep it secondary and automatic (mask-overlap against the COCONut panoptic map), so no pixel-quality human eval is required.
4. Copy SmartEraser's idea that construction determines the ground truth. Our ground truth is also by construction (rule over masks), which avoids annotator disagreement.

### Gaps
- No set-valued, subject-relative removal benchmark exists in what I found.
- The existing "reasoning removal" benchmark (ICOREval) is 110 items with pixel metrics and an LLM examiner.
- I could not verify ROVI. Also, no paper evaluates removers when given the *wrong* set.

## Q3. Reasoning-based and multi-turn editing benchmarks, and whether any score localization separately

### Takeaway
Reasoning editing benchmarks (Reason-Edit, RISEBench, KRIS-Bench, CompBench) are small (hundreds to about 1.3K items) and judged by GPT-4o / LMM-as-judge on the generated image. Only a few decompose the pipeline. EdiVal (ICLR 2026) is the clearest: object-centric, detector-checked instruction following, with a dynamic object pool across turns. Localization is rarely scored on its own. Multi-turn benchmarks (MagicBrush, EdiVal, MICE-Bench, I2EBench2.0, ImgEdit multi-turn) all measure *pixel consistency across turns*, never whether the model correctly re-resolves the target after the user changes the rule.

### Cited Findings

**A. Reasoning / complex-instruction editing**
- **Huang et al. 2023, SmartEdit** (2312.06739, CVPR 2024) [DL `2312.06739_Huang2023_smartedit.pdf`]. MLLM-enhanced editing with a Bidirectional Interaction Module. The **Reason-Edit** eval set has only **219** image-text pairs, covering complex understanding (location, color, relative size, "in or outside the mirror") and reasoning scenarios. This is the closest prior to "which object does the instruction mean".
- **Zhao et al. 2025, RISEBench** (2504.02826) [DL `2504.02826_Zhao2025_risebench.pdf`]. Temporal, Causal, Spatial, Logical reasoning. Judged by humans and LMM-as-judge on Instruction Reasoning, Appearance Consistency, Visual Plausibility. GPT-4o-Image reaches only **28.8%** accuracy.
- **Wu et al. 2025, KRIS-Bench** (2505.16707) [DL `2505.16707_Wu2025_kris-bench.pdf`]. 22 tasks, 7 reasoning dimensions, **1,267** instances, 10 models, with a "Knowledge Plausibility" metric.
- **Jia et al. 2025, CompBench** (2505.12200) [DL `2505.12200_Jia2025_compbench.pdf`]. Nine tasks (object addition / removal / replacement, multi-object editing, location editing, and others). Instructions are decoupled into location, appearance, dynamics, objects. The authors argue earlier benchmarks have oversimplified, object-centric scenes.
- **Ma et al. 2024, I2EBench** (2408.14180, NeurIPS 2024) [DL `2408.14180_Ma2024_i2ebench.pdf`]. 2,000+ images, 4,000+ instructions, 16 dimensions, user-study alignment. **I2EBench2.0** (2606.15570) [DL `2606.15570_Ma2026_i2ebench2.pdf`] adds multi-round (16 single-round + 7 multi-round dimensions, 8 models).
- **Ye et al. 2025, ImgEdit** (2505.20275) [DL `2505.20275_Ye2025_imgedit.pdf`]. 1.2M edit pairs built with a VLM + detector + segmenter pipeline. Its benchmark has a basic suite, a challenging single-turn suite (Understanding-Grounding-Editing, UGE) and a multi-turn suite. The paper reports a *grounding* capability (find the region to edit) but scores it through the edited image with GPT-4o, not as a mask.
- **Yang et al. 2025, Complex-Edit** (2504.13143) [no-DL]. GPT-4o "Chain-of-Edit" instructions of controlled complexity. Finding: decomposing a complex instruction into sequential atomic steps *degrades* performance.
- Older anchors: **Brooks 2022, InstructPix2Pix** (2211.09800) [no-DL]; **Zhang 2023, MagicBrush** (2306.10012) [DL `2306.10012_Zhang2023_magicbrush.pdf`]: 5,313 sessions / 10,388 turns, sampled from **MS COCO** images with class balancing over 80 classes, with single/multi-turn and mask/mask-free variants; **Sheynin 2023, Emu Edit** (2311.10089) [no-DL]: benchmark across seven tasks; **Wang 2022, EditBench** (2212.06909) [no-DL]: human evaluation of text-guided inpainting over objects / attributes / scenes. **Liu 2025, GEdit-Bench** is introduced inside Step1X-Edit (2504.17761) [no-DL] and rooted in real user instructions (606 images, per the EdiVal paper).

**B. Object-centric, checkable, multi-turn evaluation**
- **Chen et al. 2025, EdiVal-Agent** (2509.13399, ICLR 2026) [DL `2509.13399_Chen2025_edival-agent-multi-turn-editing.pdf`]. Pipeline: a VLM parses objects with attributes, Grounding DINO validates boxes, instructions are generated against a *dynamically updated object pool across turns*. Metrics: EdiVal-IF (detectors for symbolic checks plus VLM verification on detector-guided crops), EdiVal-CC (semantic similarity of unchanged objects), EdiVal-VQ. EdiVal-Bench has **1,716 instructions**, 3 turns, 9 instruction types, **572 images** (GEdit-Bench's 606 minus 34), and 16 models. Qwen-Image-Edit degrades 78.36 -> 59.56 -> 41.93 (EdiVal-O) over three turns, which the authors attribute to exposure bias. This is the closest *design* precedent for checkable object-level scoring and multi-turn state tracking. It scores the generated image, not the planner's object choice.
- **Ye et al. 2026, Edit-R2 / MICE-Bench** (2606.05950) [DL `2606.05950_Ye2026_edit-r2-mice-bench.pdf`]. Multi-turn in-context editing. Adds a "global awareness" metric for session-level constraints that the user stated earlier. Two named failure modes are long-context dilution and state contamination.
- **Zhou et al. 2025, Multi-turn Consistent Image Editing** (2505.04320) [no-DL], **Zhao et al. 2026, LME-Bench** (On-Policy Self-Distillation, 2609.35611; 10-turn sessions) [no-DL], **Dorfman 2026, ACEdit** (2610.22328) [no-DL], **Chen 2026, MT Forcing** (2610.25404) [no-DL]. All measure consistency across turns.

**C. Localization scored separately**
- **Zhang et al. 2026, PC-Edit** (2607.21318) [no-DL]: EditRegion-Bench, 484 cases (236 additions, 248 replacements), one human box per target, scored with mIoU / AP / PeakHit (from search summary; the ID was verified, the numbers were not read in the paper). **He et al. 2026, "Rethinking Where to Edit"** (2604.20258) [no-DL]: task-aware localization inside IIE models, evaluated on EdiVal-Bench. Both localize *inside the editor* and for add/replace tasks, not for rule-selected sets.

**D. Multi-turn grounding / dialogue**
- **Tian et al. 2024, ChatterBox** (2401.13307) [DL `2401.13307_Tian2024_chatterbox-multi-round-grounding.pdf`]. Multi-round referring and grounding (MRG). CB-300K is built on **Visual Genome**, with 339,877 dialogue threads / 717,075 QA pairs (counts from search snippet; the abstract does not give them). The benchmark requires "consistent reasoning" across rounds, with an accuracy metric covering both the referring and grounding sides.
- **Rasheed et al. 2023, GLaMM** (2311.03356) [DL `2311.03356_Rasheed2023_glamm-grand.pdf`]. Grounded Conversation Generation; GranD annotates 7.5M concepts over 810M regions with masks.
- Peripheral: MC-Bench (2410.12332, multi-image grounding), MT-Video-Bench (2510.17722, multi-turn video dialogue, includes topic shifting), MM-Snowball (2606.00622, hallucination snowballing in multimodal multi-turn dialogue) [all no-DL]. None tests a mid-conversation change of subject or rule with set-valued answers.

### Implications for our design
1. Adopt EdiVal's methodological stance (object-level, detector-checkable, dynamic object pool) but score the *selection* directly against COCONut IDs, so no generator, judge, or pixel metric is in the loop. This is the answer to the "avoid pixel-quality human eval" constraint.
2. For the dynamic-intent track, reuse the MagicBrush / EdiVal / MICE-Bench structure (sessions of 2-3 turns on one image) but make each turn change the subject or rule. Score three things: set correctness at every turn, whether stale rule content leaks into the new answer (carry-over error), and the set difference between turns. Add a "rule changed but gold set did not" control to separate rule following from copying.
3. MICE-Bench's "global awareness" and EdiVal's exposure-bias finding give a testable hypothesis: models that degrade over turns in generation will also carry stale constraints in selection.
4. Position against CompBench / Reason-Edit / ImgEdit-UGE: they contain "which object" sub-skills but fold them into an end-to-end generated image judged by GPT-4o, with 219 to about 1.7K items. We isolate that sub-skill with exact labels at much larger scale.

### Gaps
- No benchmark scores the planner's object choice against ground-truth masks for relative-rule instructions.
- No multi-turn benchmark changes the *subject* or *rule* and checks the new target set. They change the edit operation.
- Localization-only scoring exists for add / replace (EditRegion-Bench) but not for multi-object sets.
- I did not read RISEBench's per-item counts (only the 28.8% headline), so avoid quoting its size.

## Q4. The data source and its peers

### Takeaway
COCONut-PanCap is a good fit: 118K train + 25K val COCO-image panoptic masks (human-verified COCONut) plus a human-edited, GPT-4V-drafted dense caption whose phrases are tied to mask IDs. It is human-annotated, which helps us on label-quality arguments. The caveats we must carry into the paper are license (research / non-commercial) and a possible reliance on COCO's 133-class vocabulary. We found no independent audit of it.

### Cited Findings

**A. COCONut and COCONut-PanCap**
- **Deng et al. 2024, COCONut: Modernizing COCO Segmentation** (2404.08639, CVPR 2024) [DL `2404.08639_Deng2024_coconut-modernizing-coco.pdf`]. About **383K images and 5.18M human-verified panoptic masks**. Splits: COCONut-S (the 118K COCO train2017 images), -B (242K images / 2.78M masks), -L (358K images / 4.75M masks), and COCONut-val (5K images). Built by an assisted-manual pipeline in which neural nets propose masks, humans verify and relabel, and the model is retrained iteratively (relabeling rounds fall as more data is added).
- **Deng et al. 2025, COCONut-PanCap** (2502.02589; NeurIPS 2025 Datasets and Benchmarks Track per the NeurIPS proceedings page) [DL `2502.02589_Deng2025_coconut-pancap.pdf`].
  - **Size:** 118K training images (average caption length **203 words**, 11 sentences) and 25K validation images (233 words). "143K annotated images" in total. The validation set is the COCONut-val split of 25K images (COCO2017-val plus 20K Objects365 validation images). Average **13.2 masks per image** (GLaMM-GCG: 3.6).
  - **Provenance:** human-annotated COCONut panoptic masks, both "thing" and "stuff", are overlaid on the image with class names using set-of-marks. GPT-4V is prompted with the marked and original images to draft region captions. Human raters then review and correct them, and the edits are merged into one detailed caption whose phrases are tagged `<id: phrase>`. The class vocabulary is COCO's 133 panoptic classes (captions mention extra concepts such as "vegetable").
  - **Quality evidence (the authors' own):** a single-evaluator, 4-way choice on 1,000 sampled images against GPT-4V, Qwen2-VL, and InternVL-2 captions. GPT-4V was preferred in 87 cases, and 46 of those 87 were rated ties against PanCap. That is relative preference, not an error audit.
  - **Stated limitation:** "relatively smaller dataset size", with future work suggested on synthetic expansion.
  - **Known issues relevant to us (my inference, not published):** captions were drafted by a VLM, so residual hallucination is possible. The paper itself observes that GPT-4V *adds hallucinated sunlight* in its own tier examples. The mask-to-phrase links are the part we would rely on. I found no mask-caption misalignment audit and no third-party error analysis (searched 2025-2026).
  - **Release / license:** the HF dataset `xdeng77/coconut_pancap` (`license: other`, files `caption_train2017.tar`, `caption_val2017.tar`, `pancap_val.tar`) and the COCONut GitHub README state: "We follow the same license as COCO dataset for images. The dataset cannot be used for commercial purposes. ... created for research purposes." The README timeline says full PanCap was released 2026-02-12 on HF, after earlier 50K-caption and region-30K subsets. The arXiv paper page itself is CC BY 4.0, but that licenses the paper, not the data. The HF card's annotation example is a simplified COCO-style JSON (segments_info with `segment_id`, `category`, `caption`) and the real caption format should be checked after download.
  - COCO's own terms (Flickr image licenses, annotation license) were **not** verifiable from cocodataset.org in this session (JS-rendered), so state "COCO terms of use" without quoting a CC clause.

**B. Alternatives with dense masks + grounded text**
- **GranD** (inside GLaMM, 2311.03356, [DL]): 7.5M concepts, 810M regions with masks, built automatically (specialist models plus LLM), so noisier than PanCap but vastly larger. Not human-verified.
- **SA-1B** (Kirillov 2023, 2304.02643) [no-DL]: 11M licensed images, 1B+ masks, class-agnostic and caption-free. It needs separate captioning; DCI (8K human captions on SA-1B) exists but is small (counts from the PanCap comparison table).
- **DenseWorld-1M** (Li 2025, 2506.24102) [DL `2506.24102_Li2025_denseworld-1m.pdf`]: about 1M images with entity masks, object-level and relational dense captions, three-stage VLM pipeline, high-resolution. Largest and richest on relations, but model-generated.
- **PSG** (Yang 2022, 2207.11247) [no-DL]: 49K images from COCO and Visual Genome with panoptic masks and relation triplets. Good for *relational* ground truth.
- **Panoptic Narrative Grounding** (Gonzalez 2021, 2109.04988) [no-DL]: Localized Narratives transferred onto COCO panoptic regions. Baseline 55.4 absolute Average Recall. Noun-phrase to mask, with no relative-rule semantics.
- **EntitySeg** (Qi 2022, 2211.05776) [no-DL]: high-resolution in-the-wild dense entity masks, with no text.
- **LVIS** (Gupta 2019, 1908.03195) [no-DL], **Open Images V4** (Kuznetsova 2018, 1811.00982) [no-DL], **Visual Genome** (Krishna 2016, 1602.07332) [no-DL]: object / relationship annotations. Open Images masks and boxes are partial. Progressive Simplification uses Open Images.
- Peer uses of COCO for editing: MagicBrush (sampled from COCO, Q3) and SimpSON (pretrained on COCO Panoptic).

### Implications for our design
1. COCONut-PanCap is the right primary source: panoptic IDs are exhaustive (thing + stuff, 13.2 masks per image), so "the set of all other objects" is well-defined and rules can be executed programmatically. PanCap's grounded phrases let a VLM be prompted for the subject while the rule's gold set comes from masks and geometry.
2. Add a held-out validation layer on top of the dataset: sample a subset and have a second pass (rule execution plus a VLM/human sanity check) to *measure* the mask / caption error rate. This answers the "low reliability" reviewer critique, and nobody has published such an audit for PanCap.
3. Treat the 133-class vocabulary as a limit: rules should be phrased over classes, geometry, and counts that COCONut makes reliable. Consider a PSG or DenseWorld cross-check subset if you need relations beyond geometry.
4. Licensing: keep to research and non-commercial release. Release only IDs / prompts / gold sets, not images. Say explicitly that downstream users must obtain the images under COCO terms and the annotations from the COCONut-PanCap HF repository. The same applies to a possible "edit planning" demo.
5. COCO is heavily represented in VLM training and in editing benchmarks (MagicBrush). Run a contamination caveat. Prefer the COCONut-val / Objects365-val portion (which covers 20K non-COCO images) for the test split, and report the COCO/O365 split separately.

### Gaps
- No published audit of PanCap caption / mask correctness, no inter-annotator numbers beyond one user study.
- I did not locate how many PanCap images have 5+ mutually non-trivial objects per rule (this needs measurement on the actual data).
- The HF card is thin on format, and the dense-mask subset was "under final inspection" as of the 2025-05 README note.

## Q5. IEEE MultiMedia: author guidelines (official source)

### Takeaway
Verified on 2026-10-27 from the official IEEE Computer Society call-for-papers / author page (`computer.org/digital-library/magazines/mu/cfp-ieee-multimedia`). IEEE MultiMedia is a *magazine*: "conversational style", "educated general audience", and it wants "a significant focus on aspects unique to multimedia systems and applications". It is a poor fit for a long benchmark paper with many tables.

### Cited Findings
- **Length:** "should not exceed **6,500 words**, including all text, the abstract, keywords, bibliography, and biographies. **Each table and figure counts for 200 words.**"
- **References:** "limit the number of references to the **20 most relevant** (except for surveys, which may have up to 30)". A 2025 TCMC newsletter snippet said 12, but the official page says 20, so I use 20.
- **Abstract:** no more than **150 words**.
- **Submission:** via the IEEE Author Portal (ScholarOne). IEEE DataPort upload of the dataset is "encouraged" and linkable to the paper.
- **Article types:** the page does not enumerate types. The generic IEEE CS author page says magazines "publish both peer-reviewed and editorial content". The per-magazine "Author Information" page is JS-rendered and could not be read, so I could not list types (unverified).
- **Practical budget:** 6,500 words minus about 1,400-1,800 words for 7-9 figures/tables (200 each) leaves roughly 4,700-5,100 words of text, and the reference list is capped at 20 items.
- **Recent benchmark / dataset-style articles in the magazine** (found via the Crossref API for ISSN 1070-986X; coverage is sparse, so the list is not exhaustive):
  - Ikuta et al., "MangaUB: A Manga Understanding Benchmark for Large Multimodal Models", IEEE MultiMedia, April 2025, DOI 10.1109/mmul.2025.3550451.
  - Wang et al., "Trajectory Recovery: A New Setting and Benchmark for Person Re-identification", 2026, DOI 10.1109/mmul.2025.3646857.
  - Truong et al., "Toward Generalizable AI-Generated Image Detection with a New Realistic Dataset: Performance Evaluation and Improvement", 2026, DOI 10.1109/mmul.2026.3710370.
  - Li et al., "Improving Scene Knowledge Referring Expression Comprehension With Large Language Models", Jan 2026, DOI 10.1109/mmul.2025.3635870 (a grounding paper in the venue).

### Implications for our design
- 20 references means the related-work coverage here must be compressed: pick about 12 core citations (Fried, SimpSON, Progressive Simplification, ICOR, EdiVal, Reason-Edit / RISEBench, MagicBrush, ChatterBox, COCONut, COCONut-PanCap, GLaMM, SAM) and cite the rest in an appendix or a data/code repository.
- The benchmark paper should be pitched as a "multimedia systems / application" article (edit planning, conversational editing). A purely model-evaluation story is a weaker match.

### Gaps
- Article-type list and review turnaround were not retrievable. Confirm on the Author Information page before submitting.
- The Crossref sample of 2024-2026 benchmark articles is small (3-4 items), which suggests few such papers appear in the magazine.

## Closest prior work / novelty threats

| Rank | Work | What it does | Why it threatens | Our differentiator |
|---|---|---|---|---|
| 1 | Progressive Photorealistic Simplification (2605.10409, 2026) | VLM picks which elements to remove in a Select-Remove-Verify loop, with a Distractor-first taxonomy and a 93-image eval | Same "VLM decides what to remove" claim and the word "distractor" | Not subject-conditioned or rule-prompted; the label is human preference; the test set is tiny; the output is an order, not a checkable set |
| 2 | ICOR / REORM (2602.01298, 2026) | MLLM infers which extra elements to remove alongside the target; ICOREval (110) | Reasoning about *what else goes* in an editing pipeline | Physical / causal, not relative-rule; pixel metrics with an LLM examiner; 110 items; the target is named |
| 3 | EdiVal-Agent (2509.13399, ICLR 2026) | Object-centric, detector-checked, multi-turn editing evaluation with a dynamic object pool | Same "checkable, object-level, multi-turn" methodology | It scores generated images (1,716 instructions, 572 images), not the selection; edits change operation, not subject / rule |
| 4 | CompBench / Reason-Edit / ImgEdit-UGE (2505.12200, 2312.06739, 2505.20275) | Complex or reasoning-based editing with spatial / object conditions | "Which object" reasoning exists inside them | Entangled with generation; judged by GPT-4o; 219 to about 1.7K items; no set-valued gold |
| 5 | SADL itself (2606.30393) and SimpSON (2305.17624) | Subject-aware identification, 1,800 cases; click-based distractor segmentation with Distractor20K | Our own rejected benchmark, plus the origin of the task | Larger, programmatic gold, no human-attention claims; must not reuse its aesthetic framing |
| 6 | ChatterBox / CB-300K (2401.13307) and GLaMM (2311.03356) | Multi-round referring and grounding; grounded conversation | Multi-turn grounding prior art for the dynamic-intent track | Single referent per query, not rule-relative sets; no subject/rule switch |
| 7 | PhotoAgent (2602.22809), Agent Banana (2602.09084) | Agentic edit planning | "Edit planning" framing | Aesthetic reward, not selection correctness; no per-object ground truth |
| 8 | iDIS (2511.21397), Distract-Bench (2606.08894) | VLM robustness to added distractors | Cited to reviewers as "distractor benchmarks" | Distractors are injected confounds, not an identification target (already stated in SADL) |
| 9 | COCONut-PanCap (2502.02589) | Source data with grounded captions and a PGC task | Defines the pixel-grounding task on the same masks | Our task is rule-relative set selection, not captioning or segmentation; a different output |

## Bibliography

1. Fried, O., Shechtman, E., Goldman, D. B., Finkelstein, A. "Finding Distractors In Images." CVPR 2015, pp. 1703-1712. DOI 10.1109/CVPR.2015.7298779. (No arXiv ID.)
2. Huynh, C., et al. "SimpSON: Simplifying Photo Cleanup with Single-Click Distracting Object Segmentation Network." CVPR 2023. arXiv:2305.17624.
3. Aberman, K., et al. "Deep Saliency Prior for Reducing Visual Distraction." CVPR 2022. arXiv:2109.01980.
4. Miangoleh, S. M. H., et al. "Realistic Saliency Guided Image Enhancement." CVPR 2023. arXiv:2306.06092.
5. Zhang, et al. "Detecting and Removing Visual Distractors for Video Aesthetic Enhancement." IEEE Trans. Multimedia, 2018. DOI 10.1109/tmm.2018.2790163.
6. Rosenthal, A., et al. "Progressive Photorealistic Simplification." arXiv:2605.10409 (2026).
7. Yao, M., et al. "PhotoAgent: Exploratory Visual Aesthetic Planning with Large Vision Models." arXiv:2602.22809 (2026).
8. Ye, R., et al. "Agent Banana: High-Fidelity Image Editing with Agentic Thinking and Tooling." arXiv:2602.09084 (2026).
9. Bae, J., et al. "Understanding the Effects of Distractors on Reasoning Vision-Language Models" (iDIS). arXiv:2511.21397 (2025).
10. Sun, Y., et al. "Are Reasoning Vision-Language Models Robust to Semantic Visual Distractions?" (Distract-Bench). arXiv:2606.08894 (2026).
11. Winter, D., et al. "ObjectDrop: Bootstrapping Counterfactuals for Photorealistic Object Removal and Insertion." arXiv:2403.18818 (2024).
12. Li, R., et al. "RORem: Training a Robust Object Remover with Human-in-the-Loop." CVPR 2025. arXiv:2501.00740.
13. Wei, R., et al. "OmniEraser: Remove Objects and Their Effects in Images with Paired Video-Frame Data." arXiv:2501.07397 (2025).
14. Jiang, L., et al. "SmartEraser: Remove Anything from Images using Masked-Region Guidance." arXiv:2501.08279 (2025).
15. Liu, Y., et al. "Erase Diffusion: Empowering Object Removal Through Calibrating Diffusion Pathways." arXiv:2503.07026 (2025).
16. Zhao, J., et al. "Precise Object and Effect Removal with Adaptive Target-Aware Attention" (ObjectClear). arXiv:2505.22636 (2025).
17. Zhu, Z., et al. "GeoRemover: Removing Objects and Their Causal Visual Artifacts." arXiv:2509.18538 (2025).
18. Sun, W., et al. "Attentive Eraser." AAAI 2025. arXiv:2412.12974.
19. Tudosiu, P.-D., et al. "MULAN: A Multi Layer Annotated Dataset for Controllable Text-to-Image Generation." arXiv:2404.02790 (2024).
20. Huang, C.-K., et al. "Interaction-Consistent Object Removal via MLLM-Based Reasoning" (REORM / ICOREval). arXiv:2602.01298 (2026).
21. Huang, Y., et al. "SmartEdit: Exploring Complex Instruction-based Image Editing with Multimodal Large Language Models" (Reason-Edit). arXiv:2312.06739 (2023).
22. Zhao, X., et al. "Envisioning Beyond the Pixels: Benchmarking Reasoning-Informed Visual Editing" (RISEBench). arXiv:2504.02826 (2025).
23. Wu, Y., et al. "KRIS-Bench: Benchmarking Next-Level Intelligent Image Editing Models." arXiv:2505.16707 (2025).
24. Jia, B., et al. "CompBench: Benchmarking Complex Instruction-guided Image Editing." arXiv:2505.12200 (2025).
25. Ma, Y., et al. "I2EBench: A Comprehensive Benchmark for Instruction-based Image Editing." NeurIPS 2024. arXiv:2408.14180; and "An Extensive Benchmark for Single-round and Multi-round Instruction-based Image Editing" (I2EBench2.0), arXiv:2606.15570 (2026).
26. Ye, Y., et al. "ImgEdit: A Unified Image Editing Dataset and Benchmark." arXiv:2505.20275 (2025).
27. Yang, S., et al. "Complex-Edit." arXiv:2504.13143 (2025).
28. Liu, S., et al. "Step1X-Edit" (GEdit-Bench). arXiv:2504.17761 (2025).
29. Zhang, K., et al. "MagicBrush." arXiv:2306.10012 (2023).
30. Brooks, T., et al. "InstructPix2Pix." arXiv:2211.09800 (2022). Sheynin, S., et al. "Emu Edit." arXiv:2311.10089 (2023). Wang, S., et al. "Imagen Editor and EditBench." arXiv:2212.06909 (2022).
31. Chen, T., et al. "EdiVal-Agent." ICLR 2026. arXiv:2509.13399.
32. Ye, Y., et al. "Edit-R2: Context-Aware Reinforcement Learning for Multi-Turn Image Editing" (MICE-Bench). arXiv:2606.05950 (2026).
33. Zhou, Z., et al. "Multi-turn Consistent Image Editing." arXiv:2505.04320 (2025). Zhao, L., et al. "On-Policy Self-Distillation for Multi-Turn Image Editing" (LME-Bench). arXiv:2609.35611 (2026).
34. Zhang, J., et al. "PC-Edit" (EditRegion-Bench). arXiv:2607.21318 (2026). He, J., et al. "Rethinking Where to Edit." arXiv:2604.20258 (2026).
35. Tian, Y., et al. "ChatterBox: Multi-round Multimodal Referring and Grounding." arXiv:2401.13307 (2024).
36. Rasheed, H., et al. "GLaMM: Pixel Grounding Large Multimodal Model." arXiv:2311.03356 (2023).
37. Deng, X., et al. "COCONut: Modernizing COCO Segmentation." CVPR 2024. arXiv:2404.08639.
38. Deng, X., et al. "COCONut-PanCap: Joint Panoptic Segmentation and Grounded Captions for Fine-Grained Understanding and Generation." NeurIPS 2025 D&B. arXiv:2502.02589.
39. Li, X., et al. "DenseWorld-1M." arXiv:2506.24102 (2025).
40. Kirillov, A., et al. "Segment Anything." arXiv:2304.02643 (2023).
41. Yang, J., et al. "Panoptic Scene Graph Generation." arXiv:2207.11247 (2022). Gonzalez, C., et al. "Panoptic Narrative Grounding." arXiv:2109.04988 (2021). Qi, L., et al. "High-Quality Entity Segmentation." arXiv:2211.05776 (2022).
42. Gupta, A., et al. "LVIS." arXiv:1908.03195. Kuznetsova, A., et al. "Open Images V4." arXiv:1811.00982. Krishna, R., et al. "Visual Genome." arXiv:1602.07332.
43. Xu, Y., et al. "MC-Bench." arXiv:2410.12332 (2024). Pan, Y., et al. "MT-Video-Bench." arXiv:2510.17722 (2025). Jiang, Y., et al. "MM-Snowball." arXiv:2606.00622 (2026).
44. IEEE Computer Society. "Call for Papers: IEEE MultiMedia" (author submission instructions). https://www.computer.org/digital-library/magazines/mu/cfp-ieee-multimedia (accessed 2026-10-27).
45. Ikuta et al., "MangaUB," IEEE MultiMedia 2025, DOI 10.1109/mmul.2025.3550451; Wang et al., DOI 10.1109/mmul.2025.3646857; Truong et al., DOI 10.1109/mmul.2026.3710370; Li et al., DOI 10.1109/mmul.2025.3635870.
46. COCONut GitHub README (terms of use) https://github.com/bytedance/coconut_cvpr2024; HF dataset card https://huggingface.co/datasets/xdeng77/coconut_pancap (`license: other`).
