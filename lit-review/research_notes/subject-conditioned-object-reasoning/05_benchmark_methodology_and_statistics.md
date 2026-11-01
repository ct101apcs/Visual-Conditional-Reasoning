# Benchmark Methodology for a Programmatic, Mask-Derived Set-Output VLM Benchmark: Credibility, Label Noise, Statistics, and Set Metrics

Scope note: every arXiv ID below was checked against its arxiv.org/abs page on 2026-10-27 (title, first author, and year match). PDFs are saved under `lit-review/papers/` (PDFs are not tracked in git; see `lit-review/README.md`). Files marked [DL] were downloaded in this session. Files marked [DL-other] were already in the folder from parallel researchers. Items marked "(no arXiv)" are cited from a web page or venue page only. Numbers are quoted from abstracts or from PDF text read in this session unless marked "(unverified)". Numbers marked "(own computation)" are ours (scipy, Clopper-Pearson / Wilson / normal approximations), not from a paper. Anchor: the rejected SADL paper (arXiv 2606.30393; 1,000 images, Cohen's kappa 0.53 on fuzzy labels, "no meaningful insights"). The planned replacement is a programmatic benchmark. Source data is COCONut-PanCap panoptic masks plus grounded captions. A VLM grounds a subject through one of five cues (text, mask overlay, contour, box, numbered marks) and returns the set of objects satisfying a checkable rule. Ground truth (GT) is computed from masks with ambiguity margins, and humans only audit a sample.

## Q1. What makes a VLM benchmark credible in 2025-2026 reviews?

### Takeaway
Reviewers' complaints fall into six families, and each maps to a reviewer comment SADL already received:
1. **Blind-solvable items and leakage.** Items can be answered without the image, or were seen in training.
2. **Construct validity.** The score may not measure the named capability.
3. **Fragility.** Rankings flip under trivial setup changes (marker color, option order, harness).
4. **Saturation and low discrimination.**
5. **Label noise.** Benchmarks with ~3-14% wrong labels are now routinely re-audited by outsiders.
6. **Missing documentation and reproducibility.** The NeurIPS 2026 track is now called "Evaluations & Datasets" (formerly Datasets & Benchmarks). It requires code and data to be public at camera-ready and Croissant metadata with Responsible-AI fields.

The post-2024 "credible" recipe is:
- Show a **vision-indispensable control**: a blind or text-only baseline at chance.
- Report **per-slice diagnostics with CIs**.
- Report **sensitivity to the setup knobs** (prompt, marker, resolution).
- **Audit the label noise** and publish the error rate with a CI.
- Document everything in a **datasheet**.

For our design the strongest answer to "so what?" is that the cue-comparison and rule-family breakdowns are the findings. The size increase only fixes the "small" complaint.

### Cited Findings

**A. Blind solvability, leakage, saturation, vision-centric design**
- **Chen et al. 2024, MMStar, "Are We on the Right Way for Evaluating Large Vision-Language Models?"** (2403.20330) [DL `2403.20330_Chen2024_mmstar-right-way-evaluating-lvlms.pdf`]. Two problems are identified:
  - Visual content is unnecessary for many items: GeminiPro scores 42.9% on MMMU with no image, and text-only models beat random choice by >24% on average across six benchmarks.
  - Training-set leakage: Sphinx-X-MoE scores 43.6% on MMMU without images, 17.9 points above its LLM backbone.
  - The fix is MMStar: 1,500 human-selected items over 6 capabilities and 18 axes, plus two metrics for leakage and "actual multimodal gain". — [arXiv 2403.20330](https://arxiv.org/abs/2403.20330)
- **Li et al. 2024, NaturalBench** (2410.14669) [DL `2410.14669_Li2024_naturalbench.pdf`]. 10,000 human-verified VQA items. Each question is paired with **two images that yield different answers**, so a blind answer cannot score. 53 VLMs lag humans (>90%) by 50-70 points. It exposes answer bias ("choose the same answer regardless of the image") and tags each item with 1-8 skill tags. This is the template for our paired controls. — [arXiv 2410.14669](https://arxiv.org/abs/2410.14669)
- **Fu et al. 2024, BLINK** (2404.12390). 3,807 multiple-choice items from 14 classic CV tasks, many with visual prompting. Humans score 95.70%. GPT-4V and Gemini score 51.26% and 45.72%, only 13.17 and 7.63 points above random. — [arXiv 2404.12390](https://arxiv.org/abs/2404.12390)
- **Tong et al. 2024, MMVP ("Eyes Wide Shut?")** (2401.06209). Builds the benchmark from "CLIP-blind pairs", images CLIP embeds similarly but which differ visually. This is an adversarial-pair construction like NaturalBench. — [arXiv 2401.06209](https://arxiv.org/abs/2401.06209)
- **Tong et al. 2024, Cambrian-1 (CV-Bench)** (2406.16860). "Critically examines existing MLLM benchmarks" and introduces the vision-centric CV-Bench. Numbers not re-read (unverified). — [arXiv 2406.16860](https://arxiv.org/abs/2406.16860)
- **DatologyAI 2026, DatBench** (2601.02316) [DL `2601.02316_DatologyAI2026_datbench.pdf`]. Proposes three desiderata: faithfulness, discriminability, efficiency. Failure modes it quantifies:
  - Blindly solvable questions are up to 70% of some evaluations.
  - Mislabeled or ambiguous samples are up to 42% of examples in some datasets.
  - Multiple-choice to generative conversion causes capability drops of up to 35%.
  - Filtering both improves discriminability and cuts compute. — [arXiv 2601.02316](https://arxiv.org/abs/2601.02316)
- **Akhtar et al. 2026, "When AI Benchmarks Plateau"** (2602.16763) [DL `2602.16763_Akhtar2026_when-ai-benchmarks-plateau-saturation.pdf`]. 60 language-model benchmarks, 14 properties. Nearly half are saturated, with the rate rising with age. Expert curation, not public test data, predicts resilience. ICML 2026. Caveat: LLM benchmarks only. — [arXiv 2602.16763](https://arxiv.org/abs/2602.16763)

**B. Construct validity, agreement testing, documentation**
- **Bean et al. 2025, "Measuring what Matters"** (2511.04703) [DL `2511.04703_Bean2025_measuring-what-matters-construct-validity.pdf`]. 29 expert reviewers examined 445 LLM benchmarks from top venues. Patterns in phenomena, tasks, and scoring metrics undermine validity. It gives eight recommendations. These are the closest thing to a reviewer checklist (define the construct, check the task measures it, justify the metric). — [arXiv 2511.04703](https://arxiv.org/abs/2511.04703)
- **Raji et al. 2021, "AI and the Everything in the Whole Wide World Benchmark"** (2111.15366). Position paper: general benchmarks are framed as measures of broad progress, but their construct validity does not support that framing. Use this to scope claims narrowly. — [arXiv 2111.15366](https://arxiv.org/abs/2111.15366)
- **Perlitz et al. 2024, BenchBench / Benchmark Agreement Testing** (2407.13696). Over 40 benchmarks analysed. Overlooked choices (reference-benchmark set, rank-correlation type, model subset) change BAT conclusions. It proposes best practices and a package. For us, "agreement" means showing our scores correlate with, yet add information beyond, RefCOCO/gRefCOCO-style grounding and general VLM leaderboards. — [arXiv 2407.13696](https://arxiv.org/abs/2407.13696)
- **Gebru et al. 2018/2021, "Datasheets for Datasets"** (1803.09010). Documentation of motivation, composition, collection, and recommended uses. — [arXiv 1803.09010](https://arxiv.org/abs/1803.09010)
- **NeurIPS 2026 Evaluations & Datasets Track CFP** (no arXiv). Track renamed from Datasets & Benchmarks. Abstracts were due 2026-05-04 and notification was 2026-09-24. Code and datasets must be public by camera-ready, and Croissant files carry Responsible-AI metadata. — [neurips.cc/Conferences/2026/CallForEvaluationsDatasets](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets). The 2026 reviewer criteria were not retrievable (unverified).

**C. Fragility of visually prompted benchmarks (directly on our cue axis)**
- **Feng et al. 2025, "Visually Prompted Benchmarks Are Surprisingly Fragile"** (2512.17875) [DL `2512.17875_Feng2025_visually-prompted-benchmarks-fragile.pdf`]. Nine VLMs on BLINK-style visually prompted tasks.
  - Changing a marker from red to blue can reorder the leaderboard.
  - Slightly enlarging the marker puts open-source InternVL3-8B alongside or above Gemini 2.5 Pro.
  - JPEG compression level in API calls changes the lineup.
  - Variance is much larger than on semantic VLM tasks.
  - VPBench has 16 marker variants and 35,088 images, versus 224 samples in the BLINK relative-depth and correspondence tasks.
  - The paper runs a **paired bootstrap** on accuracy differences and a variance ratio R, marker-induced variance over dataset variance. — [arXiv 2512.17875](https://arxiv.org/abs/2512.17875)

### Implications for our design
1. Include **blind controls**: text-only, caption-only, and "wrong-image" (the same prompt paired with a different image). Report them with CIs and make them near chance. Use NaturalBench-style **paired items** (same rule, two images or two subjects with different GT) where possible.
2. Treat the **five grounding cues** (text, mask, contour, box, marks) as a first-class factor. Per Feng et al., sweep marker style (color, thickness, opacity, mark size, font) and report ranking stability. Do not pick one rendering and claim ranks.
3. Pre-register the construct: "set-valued rule evaluation conditioned on a grounded subject". Provide an agreement analysis (rank correlation with RefCOCO-family grounding and a general VLM leaderboard) following Perlitz-style best practices (several reference benchmarks, a stated correlation type).
4. Report **slice-level insight**: per rule family, subject cue, set-size bucket, crowdedness, and area bucket. This is the rebuttal to "no meaningful insights".
5. Ship a **datasheet** (Gebru) and a **Croissant** file with RAI fields. Release the generator, seeds, and prompts for the E&D-style reproducibility bar.
6. Use a benchmark "life plan": a hidden or rotating split (the saturation paper says public/hidden alone does not help; an expert-curated audit core does).

### Gaps
- No meta-study exists yet that is specific to **visually prompted, set-valued** benchmarks. VPBench is the nearest and it is multiple-choice.
- Saturation and construct-validity studies are LLM-centric (Akhtar 60, Bean 445). Transfer to VLMs is assumed.
- The 2026 E&D reviewer rubric text could not be retrieved. Check the live call page before submission.
- IEEE MultiMedia's own review criteria are not covered here.

## Q2. Programmatic/synthetic GT: validation, label-noise reporting, and don't-care regions

### Takeaway
Programmatic benchmarks are credible when they can show three things:
1. **Controlled biases**: the generator is checked by blind or answer-prior baselines.
2. **Measured label noise**: a human audit gives an error rate with a CI, preferably per stratum.
3. **Explicit ignore semantics**: unlabeled, ambiguous, or crowd regions are neither positives nor negatives.

Prior art shows both failure modes. Templated hard negatives can be exploited by blind models (ARO). Human-annotated "gold" sets can still carry 3-24% wrong labels. Programmatic GT removes annotator disagreement but moves the risk to **upstream mask and caption errors** and to **rule ambiguity**. An audit must therefore target the pipeline's failure points, not only a uniform sample.

### Cited Findings

**A. Procedural generators and their bias controls**
- **Johnson et al. 2016, CLEVR** (1612.06890). A diagnostic dataset "with minimal biases" and per-question reasoning-type annotations. The point that matters for us: generating from a scene structure lets you report accuracy per reasoning type and test for exploitable priors. — [arXiv 1612.06890](https://arxiv.org/abs/1612.06890)
- **Hudson & Manning 2019, GQA** (1902.09506). Scene-graph question engine with 22M questions, each with a functional program. It uses a smoothing technique to tune the answer distribution. A blind LSTM scores 42.1%, strong VQA models 54.1%, humans 89.3%. It adds consistency, grounding, and plausibility metrics. This shows the reporting norm: always publish the blind baseline. — [arXiv 1902.09506](https://arxiv.org/abs/1902.09506)
- **Zhang et al. 2024, Task Me Anything** (2406.11775) [DL `2406.11775_Zhang2024_task-me-anything.pdf`]. A benchmark-generation engine over 113K images, 10K videos, 2K 3D assets, 365 categories, 655 attributes, and 335 relationships. It can generate 750M QA pairs and answers user queries under a **compute budget**. This is a template for cost-aware evaluation of programmatic tasks. — [arXiv 2406.11775](https://arxiv.org/abs/2406.11775)
- **Zhang et al. 2024, ProVision** (2412.07012). Scene-graph plus human-written programs (24 single-image and 14 multi-image generators) to synthesize vision-centric data, claiming interpretability and factual accuracy by construction. The abstract does not describe an explicit label-error audit (not read beyond the abstract; unverified). — [arXiv 2412.07012](https://arxiv.org/abs/2412.07012)
- **Yuksekgonul et al. 2022, ARO** (2210.01936) vs **Hsieh et al. 2023, SugarCrepe** (2306.14610) [DL `2306.14610_Hsieh2023_sugarcrepe-hackable-benchmarks.pdf`]. SugarCrepe shows that compositionality benchmarks built with rule-based negatives (including ARO's family) are "hackable": **blind models with no image access outperform state-of-the-art VLMs**. Fixes: LLM-generated fluent negatives plus an adversarial refinement step to reduce bias. After re-evaluation, improvements from compositionality-inducing training were "hugely overestimated". — [arXiv 2306.14610](https://arxiv.org/abs/2306.14610), [arXiv 2210.01936](https://arxiv.org/abs/2210.01936)
- **Thrush et al. 2022, Winoground** (2204.03162) and **Diwan et al. 2022, "Why is Winoground Hard?"** (2211.00768). Winoground has two images and two captions with the same words in different order. Diwan et al. re-annotate with fine-grained tags and show that solving it needs more than compositional language (commonsense, locating small or out-of-focus objects in low-resolution images). The lesson: hardness may come from confounds such as small objects. Our "area bucket" and "crowdedness" tags control for these confounds. — [arXiv 2211.00768](https://arxiv.org/abs/2211.00768)

**B. Estimating label noise in existing benchmarks (audit protocols and sample sizes)**
- **Northcutt et al. 2021, "Pervasive Label Errors in Test Sets"** (2103.14749). Ten datasets, average error of at least 3.3%, at least 6% for ImageNet validation. Method: confident-learning flags candidates and crowdsourcing validates them, with 51% of flagged items confirmed. Rankings can flip after correction: ResNet-18 beats ResNet-50 if mislabeled prevalence rises by 6 percentage points. A model-assisted audit therefore finds errors efficiently, but the confirmation rate (51%) shows that a flag is not an error. — [arXiv 2103.14749](https://arxiv.org/abs/2103.14749)
- **Beyer et al. 2020, "Are we done with ImageNet?"** (2006.07159). A more robust human re-annotation protocol shows reported accuracy gains are substantially smaller on the new labels. — [arXiv 2006.07159](https://arxiv.org/abs/2006.07159)
- **Gema et al. 2024, MMLU-Redux** (2406.04127) [DL `2406.04127_Gema2024_are-we-done-with-mmlu.pdf`]. A two-stage **error annotation protocol** (is the instance unambiguous, then is there a single correct answer, then does it match the dataset label) with typed errors (wrong ground truth, multiple correct, no correct, bad options clarity, bad question clarity). 5,700 re-annotated questions. It estimates a 6.49% error rate overall, 57% in the Virology subset (n=100 sampled in that analysis; the 100-sample figure is from the paper's Figure 3 text). Inter-annotator reliability is reported with Cohen's kappa among three annotators on five subjects. — [arXiv 2406.04127](https://arxiv.org/abs/2406.04127)
- **Chen et al. 2024, Ref-L4** (2406.16866) [DL-other `2406.16866_Chen2024_ref-l4-revisiting-rec.pdf`]. Manual examination of val+test sets: **14% error in RefCOCO (3,054 of 21,586), 24% in RefCOCO+ (5,201 of 21,373), 5% in RefCOCOg (675 of 14,498)**. Errors include typos, expression-target misalignment, and inaccurate boxes. Removing them raises model accuracy. It is a close analog for visual-grounding noise. — [arXiv 2406.16866](https://arxiv.org/abs/2406.16866)
- **DatBench 2026** (2601.02316, above). Mislabeled or ambiguous samples reach up to 42% of items in some VLM datasets.
- **Angelopoulos et al. 2023, Prediction-Powered Inference** (2301.09633). Valid CIs from a small gold-labeled set plus many model-predicted labels. Applicable if we want to extrapolate audit findings to the whole benchmark using an automatic validity checker (e.g., a mask-consistency script) as the "predictor". — [arXiv 2301.09633](https://arxiv.org/abs/2301.09633)
- **Own computation (Clopper-Pearson / Wilson, 95%)** for a binary "GT is wrong" audit:
  - Zero errors in n=299 / 149 / 99 / 59 items gives a one-sided 95% upper bound of 1% / 2% / 3% / 5%.
  - n=400 with 8 errors (2%): CI [0.9%, 3.9%]. n=500 with 10 errors: [1.0%, 3.7%]. n=1,000 with 20 errors: [1.2%, 3.1%].
  - Half-width ±2% at an expected 5% error needs n=457. ±1% needs n=1,825.
  - Wilson and Clopper-Pearson agree to within about 0.2 points at these sizes.

**C. Ignore / don't-care regions and non-exhaustive labels**
- **Lin et al. 2014, COCO** (1405.0312). Dense crowds are not split into instances after roughly 10-15 instances; the group is labeled as a crowd region. The COCO evaluation protocol ignores detections matched to crowd regions (the evaluator behavior is from the cocoapi, not re-verified here). — [arXiv 1405.0312](https://arxiv.org/abs/1405.0312)
- **Gupta et al. 2019, LVIS** (1908.03195). A **federated** dataset: exhaustive annotation holds only within each per-category constituent set. For an image i and category c not in the positive set Pc or negative set Nc, a detection of c "is not evaluated". This is the cleanest formal precedent for "not-exhaustively-labeled" semantics. — [arXiv 1908.03195](https://arxiv.org/abs/1908.03195)
- **Liu et al. 2023, GRES / gRefCOCO** (2306.00968) [DL-other `2306.00968_Liu2023_gres-grefcoco.pdf`]. 278,232 expressions: 80,022 multi-target and 32,202 no-target, over 19,994 images and 60,287 instances. Provides masks and boxes for all targets. No-target is a first-class label. — [arXiv 2306.00968](https://arxiv.org/abs/2306.00968)
- **Chen et al. 2024, ROPE** (2407.06192) [DL `2407.06192_Chen2024_rope-multi-object-hallucination.pdf`]. Multi-object probing uses visual referring prompts "to eliminate ambiguity" and accounts for the class distribution within each image. Hallucination rises with several objects at once. — [arXiv 2407.06192](https://arxiv.org/abs/2407.06192)
- **Li et al. 2023, POPE** (2305.10355). Notes that evaluation can be affected by instruction and generation style. — [arXiv 2305.10355](https://arxiv.org/abs/2305.10355)

### Implications for our design
1. **Three-way GT label per candidate object**: `in-set`, `out-of-set`, `ignore`. An object is `ignore` when its rule-relevant measurement lies within a margin of the rule threshold (e.g., area ratio within ±δ of the cutoff, distance within ±δ), when its mask is crowd/tiny/heavily occluded, or when its caption is ambiguous. Score predictions on `ignore` objects as neutral (neither TP nor FP). This is the LVIS and COCO-crowd convention. Publish δ and the fraction ignored per rule.
2. **Report a margin sweep**: show that model rankings are stable as δ varies (e.g., δ in {0, 5, 10, 20}% of the threshold).
3. **Audit protocol, stratified**: sample at least 300 items per major stratum (rule family x cue). With zero errors that bounds the error below 1.2% (95% one-sided, own computation); at an expected 2-5% error the CI half-width is ±1.6-2.5 points. Overall n of about 1,000-1,500 audited items is affordable and sufficient. Report Clopper-Pearson or Wilson CIs, not the normal approximation, and give the CI for each stratum.
4. **Typed error taxonomy** (MMLU-Redux style): upstream mask error, missing instance in the panoptic annotation, rule ambiguity, subject-grounding ambiguity (cue does not uniquely identify the subject), caption error, rendering error. Provide error type counts so readers see where noise comes from.
5. **Two annotators on a 20% overlap**, with raw agreement and a prevalence-robust statistic (Gwet AC1 or PABAK) alongside Cohen's kappa. With rare errors, kappa is unstable even at high agreement (general statistics advice, not from a cited paper). This pre-empts the reviewer pushback on kappa 0.53.
6. **Hard-negative and bias controls**: run text-only, answer-prior (majority-set-size), and caption-only baselines, as GQA and SugarCrepe do. Report the blind F1.
7. **Evaluate the "corrected" and "raw" splits** as Ref-L4 and ImageNet-ReaL do. Show that conclusions hold on the audited-clean subset.
8. If extrapolating the audit, use prediction-powered inference or a simple stratified estimator with explicit weights.

### Gaps
- None of the procedural-generator papers (CLEVR, GQA, TMA, ProVision) publishes an audited label-error rate with a CI from the abstracts. We did not read their full text for it (unverified). This makes "audited error with CI" a possible differentiator.
- No source found that gives a canonical sample-size recommendation for benchmark label audits. The numbers above are standard binomial arithmetic.
- COCONut-PanCap's own label quality (human-verified masks) is a prior for the audit. It is covered by another researcher's note, not here.

## Q3. Statistical rigor for model comparison, efficient subsets, and prompt/seed variance

### Takeaway
The 2024-2026 literature converges on six rules:
1. Report **standard errors / CIs** per model, not bare means.
2. Use **paired** inference for model comparisons (question-level differences).
3. Use **clustered** SEs when items are grouped (our items share an image or subject).
4. Avoid the naive CLT when n is small. Use Wilson, Clopper-Pearson, bootstrap, or Bayesian intervals.
5. Treat **the top-ranked model as a post-selection result** and provide a confidence set of plausible winners.
6. Quantify setup sensitivity (prompt format, option order, scoring method, rendering) and report ranges, not one number.

Efficient-evaluation methods (IRT-based, anchor selection) can cut cost by 10-25x on language benchmarks, but they need many pre-evaluated models to fit, and fragile items are the discriminative ones, so they are risky for our small model pool. Programmatic generation gives a cleaner path: stratified sampling plus sequential allocation.

### Cited Findings

**A. Intervals, paired tests, clustering, power**
- **Miller 2024, "Adding Error Bars to Evals"** (2411.00640) [DL `2411.00640_Miller2024_adding-error-bars-to-evals.pdf`]. Treats questions as draws from a super-population. Recommends:
  1. CLT-based SEs.
  2. **Clustered SEs** when questions are grouped.
  3. Variance reduction by resampling answers and using next-token probabilities.
  4. **Paired differences** when comparing two models.
  5. **Power analysis**.

  In its Table 4 on real evals, clustered SEs are up to **3.05x larger** than naive SEs (DROP: 1.34 vs 0.44; RACE-H: 0.51% vs 0.46%; MGSM: 1.62% vs 0.86%). It also recommends reporting the number of clusters alongside the number of questions. — [arXiv 2411.00640](https://arxiv.org/abs/2411.00640)
- **Bowyer et al. 2025, "Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints"** (2503.01747) [DL `2503.01747_Bowyer2025_dont-use-clt-llm-evals.pdf`]. In simulations with known truth, CLT intervals under-cover at small n and can extend beyond [0,1] or collapse (shown at N=20). It recommends Wilson, Clopper-Pearson, bootstrap, or Bayesian intervals, with a Python library (bayes_evals). It covers single-model, two-model, IID, and clustered cases. — [arXiv 2503.01747](https://arxiv.org/abs/2503.01747)
- **Leaderboard Lottery (ICML 2026)** (no arXiv found; page: [icml.cc/virtual/2026/80012](https://icml.cc/virtual/2026/80012)). Calibrated to the Open LLM Leaderboard v2, the reported winner is the population-level best only **0-58%** of the time under prompt resampling (nominal 95%). It adapts paired-bootstrap Multiple Comparisons with the Best (MCB) to return a family-wise-error-controlled set of plausible winners. The 95% sets contain **8-35 models**. Needs only per-prompt scores. Read from the venue abstract only (unverified beyond that).
- **Menendez et al. 2026, Prompt-Dependent Ranking with Uncertainty Quantification** (2603.03336). Argues that ranks inferred from marginal per-model intervals are generally invalid; builds simultaneous intervals for pairwise utility differences. Only the search-result summary was read (unverified in detail). — [arXiv 2603.03336](https://arxiv.org/abs/2603.03336)
- **Feng et al. 2025, VPBench** (2512.17875, above). Uses a **paired bootstrap** over items to test whether differences across marker variants are significant (test whether 0 lies in the 95% bootstrap CI of the paired accuracy difference), and a variance ratio R to separate setup variance from dataset variance. This is exactly what we need for the cue comparison.
- **Madaan et al. 2024, "Quantifying Variance in Evaluation Benchmarks"** (2406.10229). Defines seed variance across initializations and monotonicity during training. Finds that continuous metrics and framing choice tasks as completions can reduce variance for ~7B models, while item analysis and IRT "struggle to meaningfully reduce variance". — [arXiv 2406.10229](https://arxiv.org/abs/2406.10229)

**B. Ranking fragility and prompt/format sensitivity**
- **Alzahrani et al. 2024, "When Benchmarks are Targets"** (2402.01781). Minor perturbations (option order, answer-selection method) shift LLM rankings by up to **8 positions** on MMLU-type benchmarks. It recommends a hybrid scoring method. — [arXiv 2402.01781](https://arxiv.org/abs/2402.01781)
- **Parupudi 2026, "There Is No Neutral Harness"** (2608.21382) [DL `2608.21382_Parupudi2026_no-neutral-harness-config-fragile-items.pdf`]. 12 open-weight LLMs, 3,679 items, 26 harness configurations. One model's score ranges 31% to 89% depending only on configuration. Four of 12 models reach rank one under some configuration. Config-fragile items explain 95.7% of the gap between adjacent models; item discrimination correlates with fragility at 0.28 (95% CI 0.25-0.30), so compression methods that keep discriminative items keep fragile ones. Scoring method (generated text vs per-option likelihood) matters most. — [arXiv 2608.21382](https://arxiv.org/abs/2608.21382)
- **Sclar et al. 2023, FormatSpread** (2310.11324). Up to **76 accuracy points** difference from prompt formatting for LLaMA-2-13B. Format performance correlates only weakly across models. Recommends reporting a range across plausible formats, not a single format. — [arXiv 2310.11324](https://arxiv.org/abs/2310.11324)
- **Voronov et al. 2024, "Mind Your Format"** (2401.06766). Template-sensitivity of in-context-learning comparisons, with a call for consistent evaluation. Abstract not re-read (unverified). — [arXiv 2401.06766](https://arxiv.org/abs/2401.06766)
- **Zhuo et al. 2024, ProSA** (2410.12405). Introduces PromptSensiScore (instance-level) and finds higher decoding confidence correlates with prompt robustness, and larger models are more robust. — [arXiv 2410.12405](https://arxiv.org/abs/2410.12405)
- **VLM-specific:** **Schmalfuss et al. 2025, PARC** (2506.14808): 11 prompt variations x 22 VLMs x 7 datasets; InternVL2 most robust (details from search summary, unverified). **Dumpala et al. 2024** (2410.13030): paraphrased prompts and reordered options shift accuracy for GPT-4o and BakLLaVA on SugarCrepe++. — [arXiv 2506.14808](https://arxiv.org/abs/2506.14808), [arXiv 2410.13030](https://arxiv.org/abs/2410.13030)

**C. IRT and efficient subsets**
- **Polo et al. 2024, tinyBenchmarks** (2402.14992) [DL `2402.14992_Polo2024_tinybenchmarks.pdf`]. About 100 curated examples suffice to estimate performance on MMLU (14K items). IRT-based estimators; tiny versions of Open LLM Leaderboard, MMLU, HELM, and AlpacaEval 2.0. — [arXiv 2402.14992](https://arxiv.org/abs/2402.14992)
- **Kipnis et al. 2024, metabench** (2407.12844). Using n > 5,000 LLMs, six benchmarks with 28,632 items are distilled to under 3% of the size. Reconstructs individual scores with 1.24% RMSE on average, the total with 0.58% RMSE, and the common factor correlates r=0.94 with the total score. Needs a large model population. — [arXiv 2407.12844](https://arxiv.org/abs/2407.12844)
- **Uebayashi et al. 2026, M3IRT** (2603.02663) [DL `2603.02663_Uebayashi2026_m3irt-multimodal-item-response-theory.pdf`]. Splits ability and difficulty into image-only, text-only, and cross-modal components. 24 VLMs on three benchmarks. It de-prioritises shortcut items and preserves ranking fidelity even when 50% of items are artificially injected low-quality ones. ICLR 2026. — [arXiv 2603.02663](https://arxiv.org/abs/2603.02663)
- **Ding et al. 2025, AutoJudger** (2505.21389). IRT difficulty plus an agent that picks informative items adaptively. Uses 4% of MMT-Bench data to reach over 90% ranking accuracy. — [arXiv 2505.21389](https://arxiv.org/abs/2505.21389)
- **Zhu et al. 2024, LIME** (2409.06851). Filters uninformative and answer-leaking samples. 76% fewer samples, 77% less evaluation time, while distinguishing models better. Finds CIDEr inadequate for MLLM captioning. — [arXiv 2409.06851](https://arxiv.org/abs/2409.06851)
- **Huang et al. 2025** (2508.11847): dropping about 0.003% of Chatbot Arena preferences can change the top-ranked model; a reminder that the top rank is the least stable quantity. — [arXiv 2508.11847](https://arxiv.org/abs/2508.11847)

### Implications for our design
1. **Unit of resampling = image (or subject)**. Items from the same image or subject are correlated. Use cluster bootstrap by image, and report the number of clusters next to the number of items. Naive SEs can be up to 3x too small (Miller).
2. **Per-model CI**: cluster-bootstrap percentile intervals (10,000 resamples) on primary metric. Use Wilson/Clopper-Pearson for exact-match proportions. Never plain CLT intervals on slices below a few hundred items.
3. **Model comparison**: paired cluster bootstrap on per-item score differences (the same item set for all models). For binary exact-set-match, McNemar's exact test gives an equivalent check; for continuous F1 use a paired bootstrap or a paired permutation test. Report effect size with a CI, not only p.
4. **Multiple comparisons**: with 8-12 models there are 28-66 pairs. Control FWER or FDR (Holm for the primary pairs; Benjamini-Hochberg for exploratory cue-by-model contrasts), or report a **MCB confidence set** for the top rank, as in Leaderboard Lottery. Report rank intervals from bootstrap, not only point ranks.
5. **Power planning (own computation, paired design, 80% power, 5% two-sided)**: to detect a 3-point accuracy difference when 10% / 20% / 30% of items are discordant between two models, one needs roughly 870 / 1,740 / 2,610 items; for a 5-point difference roughly 310 / 630 / 940 (normal approximation, paired). Tens of thousands of instances comfortably support all of these, including slices of about 1,000.
6. **Sensitivity grid, reported as ranges**: prompt wording (3-5 paraphrases of the rule statement and output format), marker rendering (color, thickness, mark size), image resolution/compression, and output-format variants. Report each model's min-max and rank range (FormatSpread, VPBench). Use a fixed decoding config (temperature 0) and, for proprietary models, record the API date, version string, and run repeated samples on a subset to measure nondeterminism.
7. **Cost-efficient evaluation**: rather than IRT (needs many fitted models; fragile items are the discriminative ones), use (a) stratified random sampling with Neyman allocation to a pilot-variance estimate, (b) a nested evaluation: a "lite" core (about 2-3K items) for all models, the full set only for the top candidates, and (c) a sequential stopping rule: stop sampling a pair when the paired CI excludes zero or the effect is below a pre-set equivalence margin. Validate the lite subset by rank correlation (Spearman / Kendall tau) with the full-set ranking over bootstrap draws and by top-1 agreement. Cost figure for the budget section should come from token counts, not from the papers.
8. **IRT as an analysis tool only**: fit a 2PL IRT on the model-by-item matrix (even with 8-12 models) to flag items with negative discrimination (probable label errors) for the audit queue, not to select the benchmark. This links to Q2 (Northcutt-style model-assisted flagging; expect about half of flags to be real errors).

### Gaps
- IRT-based subset methods were validated with hundreds to thousands of models (metabench n>5,000); with 8-12 VLMs they are unreliable. No source addresses small-model-pool efficient evaluation.
- VLM-specific statistical-rigor guidance is thin. Most rigor papers are LLM-text (Miller, Bowyer, Madaan). The transfer to set-valued, clustered, per-image metrics has to be argued by us.
- The Leaderboard Lottery paper has no arXiv ID we could find. Cite the ICML 2026 page and flag it as venue-only.
- Prompt-sensitivity studies for visual-prompt cues are limited to VPBench (multiple-choice).

## Q4. Metrics for set-valued outputs and multi-target localization

### Takeaway
The multi-target grounding literature (gRefCOCO / GRES / GREC, D3, OmniLabel, OpenRef, ChartREG++) has settled, loosely, on this pattern:
- Per-sample TP/FP/FN under a **one-to-one assignment** (greedy or Hungarian) at a match criterion (box IoU >= 0.5, point-in-mask with a small tolerance, or mask IoU).
- **Per-sample F1 averaged over samples** (macro), not pooled micro-averages, so large sets do not dominate.
- **No-target is scored separately** (N-acc / rejection reliability) and also folded into an all-sample aggregate with a convention (correct empty = 1, wrong = 0).

None of this is standardized, so a new benchmark should publish both primary and secondary metrics and show its conclusions do not hinge on the choice.

### Cited Findings
- **GRES / gRefCOCO (Liu et al. 2023)** (2306.00968) [DL-other `2306.00968_Liu2023_gres-grefcoco.pdf`]. Metrics (from the PDF text):
  - **cIoU**: total intersection pixels over total union pixels. It favors large and multi-target samples.
  - **Pr@X**: share of samples with IoU above X. No-target samples are excluded, and the starting threshold is raised to 0.7 because multi-target regions are larger.
  - **gIoU**: mean of per-image IoU over *all* samples. A correctly empty no-target prediction counts as IoU 1, a false positive as 0.
  - **N-acc** = TP/(TP+FN), where a no-target sample is a TP if the prediction has no foreground pixels. **T-acc** = TN/(TN+FP), the share of target samples not wrongly labeled no-target.

  The authors argue cIoU "favors larger objects" and so introduce gIoU. — [arXiv 2306.00968](https://arxiv.org/abs/2306.00968)
- **GREC (He et al. 2023)** (2308.16182) [DL-other `2308.16182_He2023_grec-grefcoco.pdf`] and **GREx (Ding et al. 2026)** (2601.05244). The box-level (GREC) and journal-extended (GREx: segmentation, comprehension, generation) versions of the same benchmark. Both keep multi-target and no-target as first-class. — [arXiv 2308.16182](https://arxiv.org/abs/2308.16182), [arXiv 2601.05244](https://arxiv.org/abs/2601.05244)
- **Xie et al. 2023, Described Object Detection (D3)** (2307.12813) [DL `2307.12813_Xie2023_described-object-detection-d3.pdf`]. Annotates *all* described objects in all images "without omission". REC methods struggle with confidence scores, rejecting negative instances, and multi-target cases. — [arXiv 2307.12813](https://arxiv.org/abs/2307.12813)
- **Schulter et al. 2023, OmniLabel** (2304.11463) [DL-other `2304.11463_Schulter2023_omnilabel.pdf`]. More than 28K unique descriptions over >25K images. Each description can refer to one, multiple, or no objects (negative examples in free text). Uses a **modified average precision** that handles the large label space. AP needs confidence scores, which many proprietary VLMs do not expose. — [arXiv 2304.11463](https://arxiv.org/abs/2304.11463)
- **Wu et al. 2026, OpenRef** (2605.25706) [DL `2605.25706_Wu2026_openref-open-world-rec.pdf`]. 32,735 expressions over 17,586 images and 37,698 instances, including multi-target and none-target; tiny targets (<1% area) are 73.1% of targets. It defines per-sample TP, FP, FN with a one-to-one match at IoU >= 0.5 and uses per-sample F1 = 2TP/(2TP+FP+FN) averaged over samples (its Eq. 2). It adds **N3R**, a confidence-weighted rejection score for negative expressions: the mean over negatives of the product of (1 - confidence) over spurious boxes. For models without box scores it derives confidence from the mean probability of the coordinate tokens' logits. Note: Eq. 2 is defined on positive samples; the none-target handling is through N3R. A caveat is that N3R depends on token-probability access (not available for most closed APIs). — [arXiv 2605.25706](https://arxiv.org/abs/2605.25706)
- **Niu et al. 2026, ChartREG++** (2605.07415) [DL `2605.07415_Niu2026_chartreg-plus-multi-target-grounding.pdf`]. Supports **point and box outputs**. A predicted point matches a target if it falls inside the target's mask, or within 5 pixels of the nearest foreground pixel. Boxes match at IoU >= 0.5. In both cases, **Hungarian matching under a one-to-one constraint** gives the maximum matching, and **macro P/R/F1 per sample, averaged over all test samples**, is reported. This is the closest published precedent for output-style-agnostic scoring. — [arXiv 2605.07415](https://arxiv.org/abs/2605.07415)
- **Gao et al. 2025, RefBench-PRO** (2512.06276) [DL-other `2512.06276_Gao2025_refbench-pro.pdf`]. REC benchmark that decomposes referring into perception and reasoning dimensions. Not re-read for its metric (unverified). — [arXiv 2512.06276](https://arxiv.org/abs/2512.06276)
- **Chen et al. 2024, ROPE** (2407.06192). Automated multi-object probing protocol with visual referring prompts; counts per-object correctness inside a prompt that targets several objects. — [arXiv 2407.06192](https://arxiv.org/abs/2407.06192)
- **Chen et al. 2024, Ref-L4** (2406.16866): shows that benchmark noise (5-24% errors) can swamp small metric differences. See Q2.

### Implications for our design
1. **Primary metric: per-instance set F1** between the predicted set P and GT set G (with `ignore` objects removed from both), averaged **per instance, then macro-averaged within slice**. It is equal to 2|P∩G| / (|P|+|G|). Report precision and recall separately so the error direction (over- vs under-selection) is visible.
2. **Strict secondary metric: exact-set match** (P = G after removing `ignore`). It is easy to interpret but harsh for large sets, so always show it next to F1 and stratify by |G|.
3. **Empty-set convention** (follow gIoU): if G is empty and P is empty, score 1 for all metrics. If G is empty and P is non-empty, score 0 (F1 = 0). If G is non-empty and P is empty, score 0. Report *separately*: **N-acc** on empty-GT items and **T-acc**-style "wrongly-empty rate" on non-empty items (GRES). Also report the share of empty-GT items, since a model that always answers "none" earns that share. Balance or report the empty prior explicitly.
4. **Micro vs macro**: report macro per-instance F1 as primary (as OpenRef and ChartREG++ do), and micro (pooled TP/FP/FN) as a robustness check. Show that model ranking is stable (rank correlation between the two).
5. **Matching at the object level**: because the model must select objects that we know by ID (numbered marks) or by location, convert every output style to a set of object IDs before scoring:
   - Numbered marks / IDs: direct set.
   - Points: a point is assigned to the mask that contains it. If it falls in no mask or in several, apply a tolerance rule; use Hungarian one-to-one assignment so duplicates do not double count (ChartREG++).
   - Boxes: Hungarian assignment at IoU >= 0.5 against GT-object boxes, with a sweep over 0.3 / 0.5 / 0.7.
   - Text names or captions: out of scope unless the prompt asks for IDs or coordinates; if allowed, resolve via a fixed parser and report parse-failure rate separately.
6. **Output-style comparability**: score the same item under a fixed output protocol (e.g., "list marked IDs" or "list boxes") for all models, and keep an extra "native style" track only as supplementary. Report **format-compliance rate** (valid parse) per model, and compute metrics both over all items (parse failure = empty set) and over parsed items only. Pre-register the repair rules for malformed outputs.
7. **Duplicates and out-of-universe predictions**: define FP policy for duplicates, hallucinated IDs, and IDs of `ignore` objects (neutral). Count duplicates once.
8. **Confidence-free metrics**: do not rely on AP or N3R-style scores that need logits; most proprietary VLMs do not expose box-level confidence. Use F1 and exact match as the headline, with optional logit-based extras for open models only.
9. **Chance and trivial baselines**: report all-objects, none, random-k (matching the size prior), and largest-object baselines for each metric to show the floor.
10. **Size-conditional reporting**: bucket by |G| (0, 1, 2-3, 4-7, 8+), as gRefCOCO-style multi-target is where models fail (D3, OpenRef), and show F1 vs |G| curves.

### Gaps
- No standardized set-valued grounding metric exists. OpenRef (macro F1 at IoU 0.5), ChartREG++ (Hungarian macro F1), GRES (gIoU / N-acc) and OmniLabel (modified AP) all differ. We have to justify ours.
- Only mask-point tolerances (5 px in ChartREG++) are published for point outputs. A tolerance for panoptic masks with thin or occluded objects is unspecified.
- We found no paper that studies how metric choice (macro vs micro, exact vs F1) changes **model rankings** for multi-target grounding. That analysis would itself be a contribution.
- RefBench-PRO's metric and the SADL original's metric definitions were not compared here.

## Reviewer-proofing checklist

1. **Size and power.** At least 20K items in total, at least 300 per reported slice, with a power statement for the main contrasts (about 900-2,600 items for a 3-point paired difference at 10-30% discordance, own computation).
2. **Label-noise audit with a CI.** A stratified human audit of about 1,000-1,500 items (at least 300 per major stratum), typed error taxonomy, Clopper-Pearson or Wilson 95% CIs, and a "clean subset" re-run. A zero-error stratum of 300 bounds error below 1.2%.
3. **Inter-annotator reliability done properly.** Double-label about 20% of the audit sample. Report raw agreement and a prevalence-robust statistic (AC1/PABAK) with kappa, and explain the kappa 0.53 history in SADL.
4. **Three-way GT with ambiguity margins.** `in-set` / `out-of-set` / `ignore` (LVIS and COCO-crowd convention), margin sweep, and the ignored fraction per rule.
5. **Blind and shortcut controls.** Text-only, wrong-image, caption-only, majority-size, all/none/random baselines, plus NaturalBench-style paired items. Blind performance near the floor.
6. **Cue-rendering sensitivity.** Marker style and resolution sweeps with rank ranges (VPBench). State which ranking claims survive.
7. **Prompt/format sensitivity.** 3-5 paraphrases of the rule and output format, ranges per model, and a fixed decoding configuration (FormatSpread / ProSA norms).
8. **Cluster-aware statistics.** Image-level cluster bootstrap CIs, paired tests for model comparisons, multiple-comparison control, a confidence set for the top rank, and the number of clusters reported.
9. **Set-metric transparency.** Per-instance macro F1 (primary), exact-set match, precision/recall, N-acc/wrongly-empty rate, micro-F1 robustness, and format-compliance rates. Same metric conventions for all output styles.
10. **Insights, not only scores.** Pre-registered hypotheses (e.g., which cue helps, which rule family fails, set-size scaling), each with an effect size and CI. Answers "so what?".
11. **Reproducibility package.** Public generator, seeds, prompts, model versions and API dates, Croissant file with RAI metadata, datasheet, per-item outputs, and analysis scripts (E&D 2026 requirements).
12. **Efficiency story.** Lite core with validated rank agreement against the full set, cost per model, and a sequential-evaluation rule. Do not claim IRT subset validity from a pool of 8-12 models.
13. **Scoped claims.** State the construct narrowly (Raji, Bean) and compare to existing grounding benchmarks via agreement testing (BenchBench) rather than claiming general visual reasoning.

## Bibliography

(Status: [DL] downloaded this session, [DL-other] already present, [-] not downloaded.)

1. Chen, L. et al. 2024. *Are We on the Right Way for Evaluating Large Vision-Language Models?* (MMStar). arXiv 2403.20330. [DL]
2. Fu, X. et al. 2024. *BLINK: Multimodal Large Language Models Can See but Not Perceive.* arXiv 2404.12390. [-]
3. Tong, S. et al. 2024. *Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs* (MMVP). arXiv 2401.06209. [-]
4. Li, B. et al. 2024. *NaturalBench: Evaluating Vision-Language Models on Natural Adversarial Samples.* arXiv 2410.14669. [DL]
5. Tong, S. et al. 2024. *Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs.* arXiv 2406.16860. [-]
6. DatologyAI. 2026. *DatBench: Discriminative, Faithful, and Efficient VLM Evaluations.* arXiv 2601.02316. [DL]
7. Akhtar, M. et al. 2026. *When AI Benchmarks Plateau: A Systematic Study of Benchmark Saturation.* arXiv 2602.16763. [DL]
8. Bean, A. M. et al. 2025. *Measuring what Matters: Construct Validity in Large Language Model Benchmarks.* arXiv 2511.04703. [DL]
9. Raji, I. D. et al. 2021. *AI and the Everything in the Whole Wide World Benchmark.* arXiv 2111.15366. [-]
10. Perlitz, Y. et al. 2024. *Do These LLM Benchmarks Agree? Fixing Benchmark Evaluation with BenchBench.* arXiv 2407.13696. [-]
11. Gebru, T. et al. 2018. *Datasheets for Datasets.* arXiv 1803.09010. [-]
12. NeurIPS 2026 Evaluations & Datasets Track Call for Papers (no arXiv). https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets
13. Feng, H. et al. 2025. *Visually Prompted Benchmarks Are Surprisingly Fragile* (VPBench). arXiv 2512.17875. [DL]
14. Johnson, J. et al. 2016. *CLEVR.* arXiv 1612.06890. [-]
15. Hudson, D. A. and Manning, C. D. 2019. *GQA.* arXiv 1902.09506. [-]
16. Zhang, J. et al. 2024. *Task Me Anything.* arXiv 2406.11775. [DL]
17. Zhang, J. et al. 2024. *ProVision: Programmatically Scaling Vision-centric Instruction Data for Multimodal Language Models.* arXiv 2412.07012. [-]
18. Yuksekgonul, M. et al. 2022. *When and why vision-language models behave like bags-of-words, and what to do about it?* (ARO). arXiv 2210.01936. [-]
19. Hsieh, C.-Y. et al. 2023. *SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality.* arXiv 2306.14610. [DL]
20. Thrush, T. et al. 2022. *Winoground.* arXiv 2204.03162. [-]
21. Diwan, A. et al. 2022. *Why is Winoground Hard? Investigating Failures in Visuolinguistic Compositionality.* arXiv 2211.00768. [-]
22. Northcutt, C. G. et al. 2021. *Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks.* arXiv 2103.14749. [-]
23. Beyer, L. et al. 2020. *Are we done with ImageNet?* arXiv 2006.07159. [-]
24. Gema, A. P. et al. 2024. *Are We Done with MMLU?* (MMLU-Redux). arXiv 2406.04127. [DL]
25. Chen, J. et al. 2024. *Revisiting Referring Expression Comprehension Evaluation in the Era of Large Multimodal Models* (Ref-L4). arXiv 2406.16866. [DL-other]
26. Angelopoulos, A. N. et al. 2023. *Prediction-Powered Inference.* arXiv 2301.09633. [-]
27. Lin, T.-Y. et al. 2014. *Microsoft COCO: Common Objects in Context.* arXiv 1405.0312. [-]
28. Gupta, A. et al. 2019. *LVIS: A Dataset for Large Vocabulary Instance Segmentation.* arXiv 1908.03195. [-]
29. Chen, X. et al. 2024. *Multi-Object Hallucination in Vision-Language Models* (ROPE). arXiv 2407.06192. [DL]
30. Li, Y. et al. 2023. *Evaluating Object Hallucination in Large Vision-Language Models* (POPE). arXiv 2305.10355. [-]
31. Miller, E. 2024. *Adding Error Bars to Evals.* arXiv 2411.00640. [DL]
32. Bowyer, S. et al. 2025. *Position: Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints.* arXiv 2503.01747. [DL]
33. The Leaderboard Lottery: Why Top LLM Rankings Are Often Coin Flips, and How to Fix Them. ICML 2026 (no arXiv found). https://icml.cc/virtual/2026/80012
34. Menendez, A. R. A. et al. 2026. *Prompt-Dependent Ranking of Large Language Models with Uncertainty Quantification.* arXiv 2603.03336. [-]
35. Madaan, L. et al. 2024. *Quantifying Variance in Evaluation Benchmarks.* arXiv 2406.10229. [-]
36. Alzahrani, N. et al. 2024. *When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboards.* arXiv 2402.01781. [-]
37. Parupudi, V. S. R. 2026. *There Is No Neutral Harness: Modern LLM Leaderboards Are Manufactured by Config-Fragile Items.* arXiv 2608.21382. [DL]
38. Sclar, M. et al. 2023. *Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design.* arXiv 2310.11324. [-]
39. Voronov, A. et al. 2024. *Mind Your Format: Towards Consistent Evaluation of In-Context Learning Improvements.* arXiv 2401.06766. [-]
40. Zhuo, J. et al. 2024. *ProSA: Assessing and Understanding the Prompt Sensitivity of LLMs.* arXiv 2410.12405. [-]
41. Schmalfuss, J. et al. 2025. *PARC: A Quantitative Framework Uncovering the Symmetries within Vision Language Models.* arXiv 2506.14808. [-]
42. Dumpala, S. H. et al. 2024. *Sensitivity of Generative VLMs to Semantically and Lexically Altered Prompts.* arXiv 2410.13030. [-]
43. Polo, F. M. et al. 2024. *tinyBenchmarks: evaluating LLMs with fewer examples.* arXiv 2402.14992. [DL]
44. Kipnis, A. et al. 2024. *metabench: A Sparse Benchmark of Reasoning and Knowledge in Large Language Models.* arXiv 2407.12844. [-]
45. Uebayashi, S. et al. 2026. *Evaluating Cross-Modal Reasoning Ability and Problem Characteristics with Multimodal Item Response Theory.* arXiv 2603.02663. [DL]
46. Ding, X. et al. 2025. *AutoJudger: An Agent-Driven Framework for Efficient Benchmarking of MLLMs.* arXiv 2505.21389. [-]
47. Zhu, K. et al. 2024. *LIME: Less Is More for MLLM Evaluation.* arXiv 2409.06851. [-]
48. Huang, J. Y. et al. 2025. *Dropping Just a Handful of Preferences Can Change Top Large Language Model Rankings.* arXiv 2508.11847. [-]
49. Liu, C. et al. 2023. *GRES: Generalized Referring Expression Segmentation.* arXiv 2306.00968. [DL-other]
50. He, S. et al. 2023. *GREC: Generalized Referring Expression Comprehension.* arXiv 2308.16182. [DL-other]
51. Ding, H. et al. 2026. *GREx: Generalized Referring Expression Segmentation, Comprehension, and Generation.* arXiv 2601.05244. [-]
52. Xie, C. et al. 2023. *Described Object Detection: Liberating Object Detection with Flexible Expressions* (D3). arXiv 2307.12813. [DL]
53. Schulter, S. et al. 2023. *OmniLabel: A Challenging Benchmark for Language-Based Object Detection.* arXiv 2304.11463. [DL-other]
54. Wu, Z. et al. 2026. *Towards Open-World Referring Expression Comprehension: A Benchmark with Training-free Multi-task Consistency Checker* (OpenRef). arXiv 2605.25706. [DL]
55. Niu, T. et al. 2026. *ChartREG++: Towards Benchmarking and Improving Chart Referring Expression Grounding under Diverse referring clues and Multi-Target Referring.* arXiv 2605.07415. [DL]
56. Gao, T. et al. 2025. *RefBench-PRO: Perceptual and Reasoning Oriented Benchmark for Referring Expression Comprehension.* arXiv 2512.06276. [DL-other]
