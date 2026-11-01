# Rule Adherence, Semantic-Prior Override, and Compositional Logic (Negation, Exceptions) in (V)LMs

Scope note: every arXiv ID below was checked against its arxiv.org/abs page on 2026-10-27 (title, first author, and date match). PDFs saved under `lit-review/papers/` (PDFs are not tracked in git; see `lit-review/README.md`). Files marked [DL] were downloaded in this session. Files marked [DL-other] were already in the folder from parallel researchers. Papers without a marker were verified at abstract level only (abs page / og:description), and numbers taken from abstracts are tagged "abstract". Numbers I could not confirm in either the abstract or the PDF text are tagged "(unverified)". Project anchor: SADL (arXiv 2606.30393) found that VLMs asked for "distractors" w.r.t. a named subject over-applied prompt-injected exclusion rules at scene-level plausibility, and that removing the rules recovered recall. The new design is a COCONut-PanCap benchmark where a VLM must ground a subject, apply a programmatically checkable rule (category, size ratio, contact, left/right, composed with AND/NOT/exception), and output the set of matching objects. Candidate headline: "VLMs follow the name/prior of a rule rather than its definition." **Headline status after this review: a closely related claim has already been made for VLMs (Alam 2026, "semantic fixation", 2604.12119), but only on abstract board-game outcome tags and on VLMBias counting, not on object-set selection with relational predicates. So the idea is not novel in kind; it is novel in setting and in measurement.**

## Q1. Do multimodal and text instruction/rule-following benchmarks test definition-following with novel or arbitrary rules applied to visual content?

### Takeaway
Mainstream multimodal instruction-following (IF) benchmarks (MIA-Bench, VisIT-Bench, MM-IFEval, VC-IFEval) test *output-format and style constraints* (length, genre, keywords) and loosely visual constraints judged by an LLM. They do not test whether the model uses a *user-given definition of a category or relation* to decide *which image content* qualifies. Text-LLM rule benchmarks (RuLES, RuleArena, ComplexBench, FollowBench, Inverse IFEval) have programmatic checkers and composition, but the content being judged is text. Only three recent works put novel rules over visual content: MMCL-Bench (rules learned from visual context, rubric-scored, 102 tasks), RuleMaze (rules for maze planning, compiled validators), and Alam 2026's VLM-Fix (inverse board-game rules, Q2). None targets object-set selection on natural images with crisp relational predicates and an exclusion/exception algebra. The closest *crisp-predicate* language benchmark family is generalized referring expressions (gRefCOCO, OmniLabel), but those predicates are natural-language descriptions, not prompt-supplied definitions.

### Cited Findings

**A. Multimodal IF benchmarks**
- **Qian et al. 2024, MIA-Bench** (2407.01509) [DL `2407.01509_Qian2024_mia-bench.pdf`]. 400 image-prompt pairs with layered instructions (description, length, genre, grammar, etc.), scored by GPT-4o as judge. Best model GPT-4o scores 88.58. Constraints are about the *response*, not about a semantic rule over image content. — [arXiv 2407.01509](https://arxiv.org/abs/2407.01509)
- **Bitton et al. 2023, VisIT-Bench** (2308.06595). 70 instruction families, 592 test queries with human-authored instruction-conditioned captions; the top instruction-following model wins against the GPT-4 reference in just 27% of comparisons (abstract). Open-ended quality, not rule adherence. — [arXiv 2308.06595](https://arxiv.org/abs/2308.06595)
- **Ding et al. 2025, MM-IFEngine / MM-IFEval** (2504.07957) [DL `2504.07957_Ding2025_mm-ifengine.pdf`]. Compose-level (output) and perception-level (tied to image content) constraints; 32 constraint types; 400 problems (search summary of the ICCV version gives a 300/100 compose/perception split; the 32-constraint and 400-problem counts are from secondary summaries, (unverified) against the PDF). Rule-based plus LLM-judge scoring. Fine-tuning on MM-IFInstruct-23k gives +10.2% on MM-IFEval, +7.6% on MIA, +12.3% on IFEval (abstract). Perception-level constraints are things like "mention the object at the left"; they do not redefine a category. — [arXiv 2504.07957](https://arxiv.org/abs/2504.07957)
- **He et al. 2026, VC-IFEval** (2601.03198). Adds vision-dependent constraints to instruction design and shows fine-tuning gains (abstract); still response-level adherence. — [arXiv 2601.03198](https://arxiv.org/abs/2601.03198)

**B. Visual rule-induction / rule-application**
- **Chen et al. 2026, MMCL-Bench** (2605.12703) [DL `2605.12703_Chen2026_mmcl-bench-visual-rules.pdf`]. Multimodal context learning: learn task-local rules, procedures, and empirical patterns from visual/mixed teaching context and apply them to new visual instances. 102 tasks in three categories (rule system application, procedural execution, empirical discovery). The strongest model solves fewer than one-third of tasks under strict scoring; failures arise at context anchoring, visual evidence extraction, context reasoning, and response construction (abstract). Closest *multimodal* "apply a given rule system" benchmark, but tasks are heterogeneous (screenshots, manuals, videos) and rubric-scored, not a set-selection task with exact checkers. — [arXiv 2605.12703](https://arxiv.org/abs/2605.12703)
- **Chen et al. 2026, RuleMaze** (2608.20237). MLLMs navigate mazes while obeying natural-language rules; rules are LLM-generated, converted to logical form, and compiled to Python validators ("Language-Logic-Function Hybridization"). Proposes Disentangled Multimodal Planning (perception / execution / rule verification separated) to improve compliance on unseen rules (abstract). Same programmatic-rule philosophy as ours, different task (planning, synthetic mazes). — [arXiv 2608.20237](https://arxiv.org/abs/2608.20237)
- **Xu et al. 2025, VisuLogic** (2504.15279). 1,000 human-verified vision-centric reasoning problems; most MLLMs score below 30%, close to the 25% chance level, versus 51.4% for humans (abstract). Rule *induction* from images, not rule *application* from a prompt definition. — [arXiv 2504.15279](https://arxiv.org/abs/2504.15279)
- **Wu et al. 2023, Bongard-OpenWorld** (2310.10207). Few-shot induction of free-form visual concepts from positive/negative image sets on real images; a significant challenge for few-shot reasoning algorithms and LLM/VLM pipelines (abstract). Again induction, not definition following. — [arXiv 2310.10207](https://arxiv.org/abs/2310.10207)

**C. Text-LLM rule-following and composition benchmarks**
- **Mu et al. 2023, RuLES** (2311.04235) [DL `2311.04235_Mu2023_can-llms-follow-simple-rules.pdf`]. 14 simple text scenarios, each with a programmatic function that detects rule breaks; test suites Benign / Basic / Redteam. "Almost all current models struggle to follow scenario rules, even on straightforward test cases" (abstract); GPT-4 is near-perfect on the benign suite. The paper also reports that adversarial GCG suffixes sharply lower direct-request pass rates (Table 1; per-model figures not re-derived here, (unverified)). Methodological template for programmatic checking. — [arXiv 2311.04235](https://arxiv.org/abs/2311.04235)
- **Zhou et al. 2023, IFEval** (2311.07911). Rule-verifiable "verifiable instructions" for text. — [arXiv 2311.07911](https://arxiv.org/abs/2311.07911)
- **Wen et al. 2024, ComplexBench** (2407.03978). 4 constraint types, 19 dimensions, 4 composition types; scoring follows the dependency structure of compositions; "significant deficiencies" on multi-constraint compositions (abstract). — [arXiv 2407.03978](https://arxiv.org/abs/2407.03978)
- **Jiang et al. 2023, FollowBench** (2310.20410). Adds one constraint per level; 13 LLMs evaluated; performance falls as constraints accumulate (abstract). — [arXiv 2310.20410](https://arxiv.org/abs/2310.20410)
- **Zhou et al. 2024, RuleArena** (2412.08972). Airline baggage, NBA transactions, tax rules; LLMs "struggle to identify and apply the appropriate rules, frequently becoming confused by similar but distinct regulations" and perform poorly overall (abstract). — [arXiv 2412.08972](https://arxiv.org/abs/2412.08972)
- **Parmar et al. 2024, LogicBench** (2404.15522). Systematic propositional / first-order / non-monotonic logic QA (abstract-level only; no numbers recorded here). — [arXiv 2404.15522](https://arxiv.org/abs/2404.15522)
- **Zhang et al. 2025, Inverse IFEval** (2509.04292) [DL `2509.04292_Zhang2025_inverse-ifeval.pdf`]. 1,012 Chinese/English questions, 8 "counter-intuitive" instruction types (Question Correction, Intentional Textual Flaws, Counterfactual Answering, Code without Comments, ...). o3-high is best, then Gemini-2.5-Pro; all models are above 65 on Counterfactual Answering but half score below 30 on Question Correction. Frames the failure as "cognitive inertia" from SFT conventions. Text-only; strongest evidence that instructions that cut against trained conventions are followed worse. — [arXiv 2509.04292](https://arxiv.org/abs/2509.04292)
- **Simpson et al. 2026, "Confidently Wrong: Exception Chain Collapse"** (2607.23386) [DL `2607.23386_Simpson2026_exception-chain-collapse.pdf`]. Nested rules of the form "A required UNLESS B applies, UNLESS C overrides B". Benchmark of 225 scenarios in four regulatory domains. In a 20-scenario adversarial extension, 3 of 4 frontier configurations fail the same coverage-gap edge case; the same alias moved from 96.6% to 100% (GPT-5.4, construction insurance) between March and April 2026 with no version bump (abstract). Vendor paper (sells a neuro-symbolic engine), so treat comparisons as favorable to the authors. Direct evidence that exception *depth* is a failure axis. — [arXiv 2607.23386](https://arxiv.org/abs/2607.23386)

### Implications for our design
1. Our benchmark fills a real hole: no multimodal IF benchmark scores *which image content qualifies* under a prompt-supplied definition with an exact checker. Position it against MIA-Bench / MM-IFEval (response constraints) and MMCL-Bench / RuleMaze (rules over visual content, but non-set outputs).
2. Borrow RuLES's design: every item has a programmatic pass/fail function; report per-rule-class pass rates, not just aggregate.
3. Borrow ComplexBench / FollowBench's *incremental composition* axis: levels L0 (single predicate), L1 (AND), L2 (NOT), L3 (exception) so composition cost is isolated. Exception Chain Collapse supports adding an explicit "UNLESS ... UNLESS" depth axis.
4. Inverse IFEval supplies the text-side argument for a "counter-intuitive rule" arm: define the rule so it contradicts the common meaning of its name.

### Gaps
- I found no multimodal benchmark with prompt-supplied *category/relation definitions* and exact set-level scoring. MMCL-Bench is the nearest; its task format (rubric-scored, heterogeneous) differs.
- I did not find numbers for LogicBench or for modern VLMs on MM-IFEval perception-level split in the abstracts; read the tables before claiming a headline contrast.
- The Selective VQA / abstention line was not located and verified in this pass.

## Q2. Semantic priors vs specified definitions: is "name/prior overrides definition" already shown for VLMs?

### Takeaway
**Yes in spirit, not in our setting.** Three direct predecessors exist: (i) in text LMs, flipped/unrelated-label ICL (Wei 2023), Stroop-style redefinition (Wang 2026), and semantic override (Thota 2026); (ii) in VLMs, Alam 2026's "semantic fixation" with **neutral-alias vs semantically loaded alias** prompts, which is essentially our "Type-K vs distractors" manipulation applied to board-game outcomes; and (iii) Wang 2026 (Princeton), where a "no-prior" setting replaces object names with placeholders such as "target object" and demonstrations specify the rule. What no one has done (to my search): rename a *visual-object selection rule* on natural images (per-object set output), with a counter-intuitive-name arm, scored against a crisp programmatic predicate. Visual counterfactual work (VLMs are Biased, Pixels vs Priors, Vision-Default/Prior-Override) shows prior-over-evidence, which is a different axis (perception vs knowledge) than prior-over-*instructed definition*.

### Cited Findings

**A. Text LMs: priors vs context-specified mappings**
- **Wei et al. 2023, "Larger language models do in-context learning differently"** (2303.03846) [DL `2303.03846_Wei2023_larger-lms-icl-differently.pdf`]. Flipped-label ICL and semantically-unrelated-label (SUL) ICL on GPT-3, InstructGPT, Codex, PaLM, Flan-PaLM. Small models do not dip below chance when labels are flipped (they ignore the flip and rely on semantic priors); large models can override priors. Example: text-davinci-002 falls from 90.3% with 0% flipped labels to 22.5% with 100% flipped. Instruction-tuned models are better at SUL-ICL (learn input-label mappings) but more reluctant to follow flipped labels, i.e. instruction tuning strengthens semantic priors. — [arXiv 2303.03846](https://arxiv.org/abs/2303.03846)
- **Wei et al. 2023, "Symbol tuning improves in-context learning"** (2305.08298). Fine-tuning Flan-PaLM (8B-540B) with natural-language labels replaced by arbitrary symbols improves robustness to underspecified prompts and restores some ability to follow flipped labels that instruction tuning had removed (secondary summary of the Google blog; (unverified) against PDF). — [arXiv 2305.08298](https://arxiv.org/abs/2305.08298)
- **Wu et al. 2023, "Reasoning or Reciting?"** (2307.02477) [DL `2307.02477_Wu2023_reasoning-or-reciting.pdf`]. 11 tasks with "counterfactual" variants that change default assumptions (base-9 arithmetic, rotated coordinates, 1-based indexing). Performance is nontrivial but "substantially and consistently degrades" relative to default (abstract). Establishes the default-vs-counterfactual-rule paired-evaluation protocol we want. — [arXiv 2307.02477](https://arxiv.org/abs/2307.02477)
- **Thota et al. 2026, "When Models Ignore Definitions: Semantic Override Hallucinations"** (2602.17520) [DL `2602.17520_Thota2026_semantic-override-definitions.pdf`]. Defines *semantic override*: the LLM reverts to its pretrained default interpretation despite explicit redefinition in the prompt. 30 logic/digital-circuit tasks (redefined gates, operator overloading) on three frontier LLMs; "persistent noncompliance with local specifications" and dropped constraints even in elementary settings (abstract). Small (n=30) and text-only. — [arXiv 2602.17520](https://arxiv.org/abs/2602.17520)
- **Wang et al. 2026, Stroop-style lexical override** (2606.07555) [DL `2606.07555_Wang2026_stroop-lexical-override.pdf`]. A conflict prompt defines "doctor" as "forest" and probes the familiar associate "hospital"; a *neutral control* replaces "doctor" with a semantically weak word in both definition and query. All 11 models show positive interference; activation patching of the defined word, the definition target, and the query word restores nearly all lost margin (normalized recovery 0.92-1.06) in 1B-2B models (abstract). This is the cleanest text-side template for our named-vs-neutral-vs-counterintuitive contrast. — [arXiv 2606.07555](https://arxiv.org/abs/2606.07555)
- **Ma et al. 2025, symbol remapping for abstraction** (2505.23833). Systematic symbol remapping in rule-based tasks; Delta metric quantifies reliance on specific symbols rather than the underlying pattern; persistent abstraction gaps despite CoT (abstract). — [arXiv 2505.23833](https://arxiv.org/abs/2505.23833)

**B. VLMs: directly relevant**
- **Alam 2026, "Beyond Perception Errors: Semantic Fixation in Large VLMs" (VLM-Fix)** (2604.12119) [DL `2604.12119_Alam2026_semantic-fixation-vlm-fix.pdf`]. **Closest prior work to our headline.** Four abstract strategy games (Tic-Tac-Toe, Connect Four, Reversi, Dots and Boxes), identical terminal boards under *standard* vs *inverse (misere)* rules, 14 VLMs (GPT-4.1, GPT-5.2, Sonnet 4.0/4.5, Qwen2.5-VL, Qwen3-VL).
  - Standard 67.1% vs inverse 52.5% (14.6-point gap); 13 of 14 models lower under inverse (exception: Sonnet-4.5); largest gap on Dots and Boxes (73.8% vs 50.0%).
  - **Neutral alias** tags (POM/TOV) raise inverse accuracy to 63.08 and shrink the gap to 2.29 points; **semantic alias** (POM = favorable, TOV = unfavorable) drops inverse accuracy back to 53.51 and reopens the gap.
  - On VLMBias counting, Flip+Alias reaches 20.7% accuracy (58.9% bias) versus 11.6% / 76.7% baseline.
  - SFT/RLVR on one rule hurts opposite-rule transfer; joint training helps; late-layer activation steering partially recovers.
  - Evaluated on a single-answer outcome (winner/loser), no per-object set output, no relational predicates. Preprint, under review. — [arXiv 2604.12119](https://arxiv.org/abs/2604.12119)
- **Wang et al. 2026, "In-Context Learning Can Help VLMs Overcome Training Priors"** (2610.12942) [DL `2610.12942_Wang2026_icl-overcomes-training-priors-vlm.pdf`]. Five vision-centric tasks, prior-conflicting vs prior-aligned examples. Demonstrations give +27.2% (Qwen3-VL 235B) and +15.1% (Qwen3-VL 32B) on prior-conflicting examples with near-unchanged prior-aligned accuracy; gains +9.7% to +30.3% across five VLM families. A "no-prior" setting replaces object names with placeholders such as "target object". Attention shifts from the prior cue ("peacock") to the grounding cue ("leg"). NeurIPS 2026. Shows few-shot demonstrations can unlock override, a remedy arm for us. — [arXiv 2610.12942](https://arxiv.org/abs/2610.12942)
- **Vo et al. 2025, "Vision Language Models are Biased"** (2505.23941) [DL `2505.23941_Vo2025_vlms-are-biased.pdf`]. Counting on counterfactual images of familiar subjects (e.g. a 4th stripe on an Adidas-like logo) across 7 domains: 17.05% average accuracy. Removing the background nearly doubles accuracy (+21.09 points). Counting accuracy rises with thinking tokens to about 40% then declines with overthinking (abstract). — [arXiv 2505.23941](https://arxiv.org/abs/2505.23941)
- **Golovanevsky et al. 2025, "Pixels Versus Priors"** (2505.17127). Controls knowledge priors in VLMs through visual counterfactuals (abstract-level). — [arXiv 2505.17127](https://arxiv.org/abs/2505.17127)
- **Lietzow et al. 2026, "Vision-Default, Prior-Override"** (2606.28273). Activation patching across three VLM families: visual grounding emerges by default, prior grounding depends on 2.5-4.8% of attention heads in the second half of the network; ablating them flips knowledge-grounded answers to visual in 68-96% of cases under prior-knowledge prompts (abstract). — [arXiv 2606.28273](https://arxiv.org/abs/2606.28273)
- **Rahmanzadehgervi et al. 2024, "VLMs are Blind"** (2407.06581) [DL `2407.06581_Rahmanzadehgervi2024_vlms-are-blind.pdf`]. BlindTest, 7 simple geometric tasks; four VLMs average 58.07% (Sonnet-3.5 best at 77.84%); linear probes show the vision encoder holds sufficient information and the LM fails to decode it. — [arXiv 2407.06581](https://arxiv.org/abs/2407.06581)
- **Shahgir et al. 2026, "VLMs Need Words"** (2604.02486) [DL `2604.02486_Shahgir2026_vlms-need-words.pdf`]. VLMs do much better when relevant entities are *nameable* than unnameable, even though internal features carry the information; teaching completely arbitrary names improves performance (abstract). The converse direction of our claim: names help when they exist and the model reasons "through language". — [arXiv 2604.02486](https://arxiv.org/abs/2604.02486)
- **Xu et al. 2024, multimodal ICL demonstration analysis** (2407.00902) [DL `2407.00902_Xu2024_multimodal-icl-demonstrations.pdf`]. Flipping demonstration annotations can override pretrained semantic priors in multimodal ICL (from search summary; (unverified) against PDF). — [arXiv 2407.00902](https://arxiv.org/abs/2407.00902)

**C. Name/label leakage in CLIP-family classification**
- **Han et al. 2023, "What's in a Name? Beyond Class Indices"** (2304.02364) and **Parisot et al. 2023, "Learning to Name Classes"** (2304.01830): class-name choice affects zero-shot accuracy; ambiguous or technical names hurt (summary). **Ma et al. 2024** (2412.11917): LLM-description gains partly come from repeating the class name, so classname-free descriptors are needed as a control. These are CLIP-era name-sensitivity results; none gives the model a *definition* to follow. — [2304.02364](https://arxiv.org/abs/2304.02364), [2304.01830](https://arxiv.org/abs/2304.01830), [2412.11917](https://arxiv.org/abs/2412.11917)

### Implications for our design
1. **Cite and differentiate from Alam 2026 up front.** Their alias/semantic-alias contrast is our Type-K / "distractors" contrast. Our added value: per-object set outputs, relational predicates (size ratio, contact, left/right), natural images (COCONut-PanCap), compositional logic, and *intrusion analysis* (which non-matching objects does the model add, and do they match the *name's* prior?).
2. Use at least three name arms, following Stroop paper and Alam: (a) priors-aligned name ("distractors" for a rule that really selects distractors), (b) neutral symbol (Type-K), (c) counter-intuitive/misaligned name (rule named "distractors" but defined to select the main subject's contacts). A fourth, definition-only (no name), mirrors SADL's "remove the rules".
3. Report an explicit *prior-intrusion rate*: fraction of false positives that satisfy a plausible alternative definition tied to the name. This operationalizes "errors intrude from the model's own unconstrained notion" and is not in Alam 2026.
4. Include a few-shot arm, since Wang 2026 shows demonstrations can unlock override and Wei 2023 shows instruction-tuned models resist flipped mappings.
5. Control for object-name leakage (cf. Ma 2024 on class-name repetition): rule predicates should not share tokens with the name.

### Gaps
- No VLM study renames a per-object selection rule on natural images; I searched "renamed concept / neutral alias / placeholder label" across VLM/ICL/rule-following angles and found only Alam 2026 and Wang 2026 as near neighbours. Absence of evidence in a web search is not proof; run one more arXiv listing search (cs.CV / cs.CL, Oct 2026) before submission.
- The mechanism question (name prior vs definition) is open for set outputs; Stroop-style patching exists only for 1B-2B text models.

## Q3. Negation, exceptions, and logical composition in VLMs

### Takeaway
Negation is a documented weak point for contrastive VLMs (affirmation bias) and for referring-expression models (the "not" is ignored). For generative MLLMs, direct set-level evidence is thin: the best data are the REC negation analysis (Tumu 2025: on the 42 two-"not" and 16 one-"not" expressions that every model got wrong, at least one model failed because it ignored the negation in 92.86% and 93.75% of cases respectively) and the OpenRef N3R rejection metric. Compositionality benchmarks (CREPE, SugarCrepe, VL-CheckList) are caption-matching, not rule-based selection, and SugarCrepe shows many were hackable by blind models. gRefCOCO/GREC/OmniLabel give multi-target and no-target settings, a good evaluation grammar for set outputs. I found no benchmark that tests complement outputs ("all objects *except* ...") or exception handling ("X unless Y") in VLMs; text LLMs show exception-chain failures (Simpson 2026) and the NeQA inverse/U-shaped scaling for negation. The SADL observation (over-applied exclusion) has no direct prior in VLMs that I could find.

### Cited Findings

**A. Negation**
- **Alhamoud et al. 2025, NegBench "VLMs Do Not Understand Negation"** (2501.09425) [DL `2501.09425_Alhamoud2025_vlms-do-not-understand-negation.pdf`]. 18 task variations, 79k examples across image, video, and medical datasets; retrieval with negated criteria and multiple-choice with options differing by a negated element. Modern VLMs "often perform at chance level" and show affirmation bias; fine-tuning CLIP on synthetic negated captions gives +10% recall on negated queries and +28% accuracy on negated multiple-choice (abstract). CVPR 2025. — [arXiv 2501.09425](https://arxiv.org/abs/2501.09425)
- **Singh et al. 2024, CC-Neg** (2403.20312) and **Park et al. 2025, "Know No Better"** (2501.10913): negation-aware CLIP training data (CC-Neg has 228,246 image-caption pairs with negated captions per search summary, (unverified) against abs). — [2403.20312](https://arxiv.org/abs/2403.20312), [2501.10913](https://arxiv.org/abs/2501.10913)
- **Zhang et al. 2023, NeQA "Beyond Positive Scaling"** (2305.17311). Programmatically negating OpenBookQA questions yields inverse scaling -> U-shape -> positive scaling depending on prompting and model family; the negation-understanding sub-task shows sigmoid (emergent) scaling (from the paper's abstract/summary). — [arXiv 2305.17311](https://arxiv.org/abs/2305.17311)
- **Tumu et al. 2025, "Referring Expressions as a Lens into Spatial Language Grounding in VLMs"** (2511.06146) [DL `2511.06146_Tumu2025_referring-expressions-spatial-negation.pdf`]. REC with LLaVA, Grounding DINO, DeepSeek-VL2, Qwen2.5-VL, MGA-Net. On expressions with one "not" (16 all-model-wrong cases) and two "not"s (42 cases), the share of these failure instances in which a given model failed because it ignored the negation (Table 9): DeepSeek 87.5% / 88.1%, GDINO 62.5% / 78.57%, LLaVA 25% / 35.71%, MGA-Net 50% / 52.38%, Qwen2.5-VL 62.5% / 50% (one-not / two-not); the "at least one model" aggregate is 93.75% / 92.86%. Selection bias note: only instances where *all* models were wrong. IJCNLP-AACL 2025. — [arXiv 2511.06146](https://arxiv.org/abs/2511.06146)
- **Wu et al. 2026, OpenRef** (2605.25706). REC with multi-target and none-target samples; F1 for grounding and N3R (Negative Relative Rejection Reliability) for rejecting negative expressions (abstract). — [arXiv 2605.25706](https://arxiv.org/abs/2605.25706)

**B. Compositionality**
- **Ma et al. 2022, CREPE** (2212.07796). >370K image-text pairs for systematicity; productivity tested up to nine complexities with atomic, swapping, and negation hard negatives; 7 architectures trained with 4 algorithms struggle (abstract). — [arXiv 2212.07796](https://arxiv.org/abs/2212.07796)
- **Hsieh et al. 2023, SugarCrepe** (2306.14610) [DL-other `2306.14610_Hsieh2023_sugarcrepe-hackable-benchmarks.pdf`]. Prior compositionality benchmarks are hackable: "blind models with no access to the image outperform state-of-the-art VLMs"; improvements from compositionality-inducing methods were "hugely overestimated" (abstract). Design lesson: controls against text-only shortcuts. — [arXiv 2306.14610](https://arxiv.org/abs/2306.14610)
- **Zhao et al. 2022, VL-CheckList** (2207.00221). CheckList-style probing of objects / attributes / relations on seven VLP models (abstract). — [arXiv 2207.00221](https://arxiv.org/abs/2207.00221)

**C. Set-valued grounding (multi-target / no-target)**
- **Liu et al. 2023, GRES / gRefCOCO** (2306.00968) [DL-other `2306.00968_Liu2023_gres-grefcoco.pdf`] and **He et al. 2023, GREC** (2308.16182) [DL-other `2308.16182_He2023_grec-grefcoco.pdf`]. Expressions with an arbitrary number of targets, including none; "complex relationship modeling" is the main challenge (abstract). — [2306.00968](https://arxiv.org/abs/2306.00968), [2308.16182](https://arxiv.org/abs/2308.16182)
- **Schulter et al. 2023, OmniLabel** (2304.11463) [DL-other `2304.11463_Schulter2023_omnilabel.pdf`]. >28K object descriptions on >25K images that can refer to one, multiple, or no objects; free-form negatives (abstract). — [arXiv 2304.11463](https://arxiv.org/abs/2304.11463)
- **Zong et al. 2025, Ground-V** (2505.13788). ~500K instruction-segmentation pairs for hallucinated references, multi-object, reasoning, multi-granularity, part-level; improves LISA by 4.4% and PSALM by 7.9% gIoU on average, N-Acc 83.3% on gRefCOCO (search summary of abstract). CVPR 2025. — [arXiv 2505.13788](https://arxiv.org/abs/2505.13788)
- **Gao et al. 2025, RefBench-PRO** (2512.06276). Six REC sub-dimensions (attribute, position, interaction, commonsense, relation, reject); thinking mode improves Relation and Commonsense subsets by roughly 1-8 points for Qwen3-VL-8B, Ovis2.5-9B, GLM4.1V-9B (from search summary, (unverified) against PDF). — [arXiv 2512.06276](https://arxiv.org/abs/2512.06276)
- **Villa et al. 2023, MERLIM** (2312.02219). Asks LVLMs to list visible objects and scores precision/recall/F1 against COCO labels (summary). Enumeration with set metrics. — [arXiv 2312.02219](https://arxiv.org/abs/2312.02219)

### Implications for our design
1. Evaluate with set-level precision, recall, F1 plus *exact-match rate*, and break false positives into (i) rule-violating but plausible by name prior, (ii) off-subject, (iii) hallucinated. Use gRefCOCO's no-target convention: some items should have an empty gold set.
2. Add a negation/complement axis with matched pairs: rule R vs NOT R on the same image and subject, so over-application of exclusions (SADL's finding) shows up as asymmetry between R and NOT R recall. Tumu 2025 gives a prior that "not" is dropped; SADL says exclusion may instead be over-applied. These predict opposite error signs, so report signed error.
3. Add an exception axis (R AND NOT S, then "unless T") with depth 0-2, following Simpson 2026, keeping all predicates crisp.
4. Guard against blind-model shortcuts (SugarCrepe): include a text-only/scene-shuffled baseline and a "same image, different subject" control to show the answer depends on the subject and image.

### Gaps
- No VLM benchmark with exception structure or complement outputs found; Tumu 2025 covers only "not" inside spatial referring expressions and only on all-model-failed items.
- The mechanism behind SADL's over-applied exclusion is untested in any VLM study I found.
- CLIP-style negation results (NegBench) may not transfer to generative MLLMs; the NegBench abstract does not give per-model MLLM numbers I could verify.

## Q4. Prompting/decomposition remedies and reasoning models: do they help visual rule application?

### Takeaway
Evidence points both ways. (a) In text, explicit CoT can *hurt* instruction following (15 models; IFEval and ComplexBench), through reduced attention to constraint tokens, and selective reasoning recovers it. (b) In multimodal models, more reasoning can amplify hallucination via reduced visual attention (RH-Bench), and harmful overthinking occurs (stopping at the first correct prefix improves accuracy by up to 21%). (c) For grounding with relational content, thinking helps on relation/commonsense/reject subsets (RefBench-PRO, Rex-Thinker's per-candidate verification). (d) Strict JSON/format constraints can lower reasoning accuracy, particularly in small models; draft-then-constrain restores much of it. (e) Program-based decomposition (ViperGPT, VisProg) matches our "crisp predicate" structure but needs perception modules. For our benchmark the right prediction is not "thinking fixes it" but "thinking helps composition (AND/NOT/exception) and may hurt or not fix prior intrusion"; this needs a controlled test.

### Cited Findings

**A. Decomposition, consistency, and programmatic approaches**
- **Selvaraju et al. 2020, SQuINT / VQA-Introspect** (2001.06927). In 28.14% of cases where VQA models answer a reasoning question correctly they fail the corresponding perception sub-question (search summary; (unverified) against PDF); SQuINT improves consistency by about 5%. Template for atomic-vs-composed consistency. — [arXiv 2001.06927](https://arxiv.org/abs/2001.06927)
- **Hudson & Manning 2019, GQA** (1902.09506). Compositional questions with consistency, validity, plausibility, and grounding metrics (title-level; no numbers recorded). — [arXiv 1902.09506](https://arxiv.org/abs/1902.09506)
- **Surís et al. 2023, ViperGPT** (2303.08128) and **Gupta & Kembhavi 2022, VisProg** (2211.11559). Compose vision modules via generated Python; no training. Results not re-extracted here. — [2303.08128](https://arxiv.org/abs/2303.08128), [2211.11559](https://arxiv.org/abs/2211.11559)
- **Jiang et al. 2025, Rex-Thinker** (2506.04034) [DL `2506.04034_Jiang2025_rex-thinker.pdf`]. Object referring that "detects all objects matching a description" as explicit CoT: enumerate candidates of the referred category, then verify each step by step, then abstain if none match. Trained with HumanRef-CoT (GPT-4o-generated plan/action/summary traces), SFT then RL (abstract). This is a candidate-then-verify decomposition exactly aligned with set selection; use it as a remedy arm and as a baseline family. — [arXiv 2506.04034](https://arxiv.org/abs/2506.04034)
- **Chen et al. 2026, RuleMaze DMP** (2608.20237): separating perception, execution, and rule verification improves rule compliance on unseen rules vs end-to-end textual planning (abstract). — [arXiv 2608.20237](https://arxiv.org/abs/2608.20237)

**B. Does thinking help?**
- **Li et al. 2025, "When Thinking Fails"** (2505.11423) [DL `2505.11423_Li2025_when-thinking-fails-instruction-following.pdf`]. 15 models on IFEval and ComplexBench: CoT consistently reduces instruction-following accuracy (neglecting simple constraints, adding unnecessary content), helps formatting/lexical precision; "constraint attention" metric shows CoT diverts attention away from instruction tokens. Classifier-selective reasoning recovers much of the loss (abstract). Text-only. — [arXiv 2505.11423](https://arxiv.org/abs/2505.11423)
- **Liu et al. 2025, "More Thinking, Less Seeing?"** (2505.21523) [DL `2505.21523_Liu2025_more-thinking-less-seeing.pdf`]. Eight multimodal reasoning models vs their Qwen2.5-VL backbones on five hallucination benchmarks: reasoning models show higher hallucination at both 3B and 7B; the reasoning model allocates less attention to visual tokens and more to instruction tokens; RH-AUC and RH-Bench introduced (search summary of abstract). NeurIPS 2025. — [arXiv 2505.21523](https://arxiv.org/abs/2505.21523)
- **Caldarella et al. 2026, "Thinking Past the Answer"** (2606.02835). Multimodal benchmarks: stopping at the first correct reasoning prefix improves accuracy over standard reasoning by up to 21% (abstract). — [arXiv 2606.02835](https://arxiv.org/abs/2606.02835)
- **Vo et al. 2025** (2505.23941, above): counting on counterfactual images peaks at ~40% with thinking tokens, then declines.
- **Zhang et al. 2025, MM-CoT** (2512.08228): even the strongest multimodal models struggle to pick a CoT chain that is both visually consistent and logically coherent (abstract). — [arXiv 2512.08228](https://arxiv.org/abs/2512.08228)
- **Bai et al. 2025, Qwen3-VL Technical Report** (2511.21631): releases Instruct and Thinking variants (title-level; I did not extract instruct-vs-thinking benchmark deltas). — [arXiv 2511.21631](https://arxiv.org/abs/2511.21631)
- **Simpson 2026** (2607.23386, above): GPT-5.4 at low reasoning effort passes the adversarial exception case that three other frontier configurations fail (abstract), so more reasoning effort is not monotonically better for exception chains.

**C. Structured output / schema constraints**
- **Tam et al. 2024, "Let Me Speak Freely?"** (2408.02442) [DL `2408.02442_Tam2024_let-me-speak-freely.pdf`]. Format restrictions (JSON/XML/YAML, constrained decoding) cause a significant decline in reasoning ability, stricter formats degrade more; classification tasks are mixed (abstract, plus search summary; the specific GSM8K JSON-mode figures quoted in secondary sources are (unverified) because I did not locate them in the PDF text). EMNLP 2024 Industry. — [arXiv 2408.02442](https://arxiv.org/abs/2408.02442)
- **Ray et al. 2026, "The Constraint Tax"** (2605.26128). Sub-3B models, 15,000 generations: hard answer-only schema decoding raises schema validity from 61.5% to 100.0% but lowers answer accuracy from 19.7% to 11.0% and raises wrong-valid-schema outputs from 49.5% to 88.9% (abstract). — [arXiv 2605.26128](https://arxiv.org/abs/2605.26128)
- **Reddy et al. 2026, Draft-Conditioned Constrained Decoding** (2603.03305). Unconstrained draft first, then constrained decoding; up to +24 points strict structured accuracy (e.g. 15.2% -> 39.0% on GSM8K with a 1B model) (abstract). — [arXiv 2603.03305](https://arxiv.org/abs/2603.03305)
- **He et al. 2024, "Does Prompt Formatting Have Any Impact on LLM Performance?"** (2411.10541): GPT-3.5-turbo performance varies by up to 40% in code translation depending on prompt template; GPT-4 more robust (abstract). — [arXiv 2411.10541](https://arxiv.org/abs/2411.10541)

### Implications for our design
1. Output format is a confound: fix one schema (a list of object IDs, e.g. a JSON array, with an empty list allowed) and report a free-text-then-extract ablation (cf. Tam 2024, DCCD) on a subset, so format cost is not mistaken for rule failure.
2. Include an *evaluation-time compute* axis with matched thinking/non-thinking pairs (e.g. Qwen3-VL Instruct vs Thinking) and report by composition level and name arm separately. Predictions from the literature: thinking helps at AND/NOT/exception depth, may increase reliance on language priors (Liu 2025), and may not remove name-prior intrusion (I did not see a matched thinking-vs-non-thinking comparison in Alam 2026).
3. Add a decomposed arm following Rex-Thinker / DMP: enumerate candidate objects, ask atomic predicate questions per object, compose programmatically. Compare against the single-shot composed prompt; the gap quantifies "composition cost" and, if the atomic accuracy is high, shows the failure is in applying rules rather than perceiving.
4. Report consistency: for each composed item, the atomic sub-answers (from SQuINT) and whether the composed output is logically consistent with them.

### Gaps
- I found no study of thinking vs non-thinking VLMs on *prompt-defined visual rule application*; the nearest are RefBench-PRO (REC) and the instruction-following results in text.
- Structured-output effects on VLM set tasks are unmeasured; the evidence is text-only and mostly small models.
- ViperGPT/VisProg numbers and the Selective VQA literature were not extracted in this pass.

## Closest prior work / novelty threats

| Paper | What overlaps | What it lacks vs our design |
|---|---|---|
| Alam 2026, VLM-Fix (2604.12119) | VLMs follow the prior meaning of a rule over its prompt-specified definition; neutral alias vs semantically loaded alias contrast; 14 VLMs; SFT and steering | Abstract board-game outcomes with single-answer targets; no per-object set outputs; no relational predicates (size, contact, left/right); no AND/NOT/exception composition; no natural images; no counter-intuitive-name arm; no intrusion analysis |
| Wang 2026, ICL overcomes training priors (2610.12942) | Placeholder "no-prior" naming, demonstration-based override on prior-conflicting vs aligned items in VLMs | Perceptual counting/identification of one attribute; goal is to show VLMs can see, not to test following a prompt definition; no logic composition |
| Wang 2026, Stroop-style lexical override (2606.07555); Thota 2026 (2602.17520) | Neutral-control vs conflict redefinition design; semantic override of prompt-local definitions | Text-only; small models / small benchmark; no visual grounding or set outputs |
| MMCL-Bench (2605.12703) and RuleMaze (2608.20237) | Rule application over visual content; programmatic validators (RuleMaze); frontier models fail | Heterogeneous or synthetic tasks; rubric or planning outputs; no name-vs-definition manipulation; no object-set selection on natural images |
| NegBench (2501.09425), Tumu 2025 (2511.06146), OpenRef (2605.25706) | Negation failures in VLM/REC; negative/no-target rejection | Negation within descriptions, not a prompt-supplied rule; no exception logic; no name-prior analysis |
| gRefCOCO / GREC / OmniLabel (2306.00968, 2308.16182, 2304.11463) | Multi-target and no-target set outputs with natural-image grounding | Predicates are natural-language descriptions, not prompt-defined rules; no renaming; no composed exceptions |
| Exception Chain Collapse (2607.23386); RuleArena (2412.08972); Inverse IFEval (2509.04292) | Nested UNLESS rules; counter-intuitive instruction following; programmatic or rubric checks | Text-only; no image grounding; vendor paper in one case |

## Bibliography

All IDs verified via arxiv.org/abs on 2026-10-27 (first author and year shown). "[DL]" = PDF in `lit-review/papers/` (downloaded this session unless noted).

- 2001.06927 Selvaraju 2020, SQuINTing at VQA Models: Introspecting VQA Models with Sub-Questions
- 1902.09506 Hudson 2019, GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering
- 2207.00221 Zhao 2022, VL-CheckList
- 2211.11559 Gupta 2022, Visual Programming (VisProg)
- 2212.07796 Ma 2022, CREPE
- 2303.03846 Wei 2023, Larger language models do in-context learning differently [DL]
- 2303.08128 Surís 2023, ViperGPT
- 2304.01830 Parisot 2023, Learning to Name Classes for Vision and Language Models
- 2304.02364 Han 2023, What's in a Name? Beyond Class Indices for Image Recognition
- 2304.11463 Schulter 2023, OmniLabel [DL-other]
- 2305.08298 Wei 2023, Symbol tuning improves in-context learning in language models
- 2305.17311 Zhang 2023, Beyond Positive Scaling: How Negation Impacts Scaling Trends of Language Models
- 2306.00968 Liu 2023, GRES: Generalized Referring Expression Segmentation [DL-other]
- 2306.14610 Hsieh 2023, SugarCrepe [DL-other]
- 2307.02477 Wu 2023, Reasoning or Reciting? [DL]
- 2308.06595 Bitton 2023, VisIT-Bench
- 2308.16182 He 2023, GREC: Generalized Referring Expression Comprehension [DL-other]
- 2310.10207 Wu 2023, Bongard-OpenWorld
- 2310.20410 Jiang 2023, FollowBench
- 2311.04235 Mu 2023, Can LLMs Follow Simple Rules? (RuLES) [DL]
- 2311.07911 Zhou 2023, Instruction-Following Evaluation for LLMs (IFEval)
- 2312.02219 Villa 2023, MERLIM
- 2403.20312 Singh 2024, Learn "No" to Say "Yes" Better (CC-Neg)
- 2404.15522 Parmar 2024, LogicBench
- 2407.00902 Xu 2024, From Introspection to Best Practices: Principled Analysis of Demonstrations in Multimodal ICL [DL]
- 2407.01509 Qian 2024, MIA-Bench [DL]
- 2407.03978 Wen 2024, ComplexBench
- 2407.06581 Rahmanzadehgervi 2024, Vision language models are blind [DL]
- 2408.02442 Tam 2024, Let Me Speak Freely? [DL]
- 2411.10541 He 2024, Does Prompt Formatting Have Any Impact on LLM Performance?
- 2412.08972 Zhou 2024, RuleArena
- 2412.11917 Ma 2024, Does VLM Classification Benefit from LLM Description Semantics?
- 2501.09425 Alhamoud 2025, Vision-Language Models Do Not Understand Negation (NegBench) [DL]
- 2501.10913 Park 2025, Know "No" Better
- 2504.07957 Ding 2025, MM-IFEngine [DL]
- 2504.15279 Xu 2025, VisuLogic
- 2505.11423 Li 2025, When Thinking Fails [DL]
- 2505.13788 Zong 2025, Ground-V
- 2505.17127 Golovanevsky 2025, Pixels Versus Priors
- 2505.21523 Liu 2025, More Thinking, Less Seeing? [DL]
- 2505.23833 Ma 2025, Benchmarking Abstract and Reasoning Abilities Through A Theoretical Perspective
- 2505.23941 Vo 2025, Vision Language Models are Biased [DL]
- 2506.04034 Jiang 2025, Rex-Thinker [DL]
- 2509.04292 Zhang 2025, Inverse IFEval [DL]
- 2511.06146 Tumu 2025, Referring Expressions as a Lens into Spatial Language Grounding in VLMs [DL]
- 2511.21631 Bai 2025, Qwen3-VL Technical Report
- 2512.06276 Gao 2025, RefBench-PRO
- 2512.08228 Zhang 2025, MM-CoT
- 2601.03198 He 2026, Empowering Reliable Visual-Centric Instruction Following in MLLMs (VC-IFEval)
- 2602.17520 Thota 2026, When Models Ignore Definitions [DL]
- 2603.03305 Reddy 2026, The Hidden Cost of Structured Generation in LLMs
- 2604.02486 Shahgir 2026, VLMs Need Words [DL]
- 2604.12119 Alam 2026, Beyond Perception Errors: Semantic Fixation in Large VLMs [DL]
- 2605.12703 Chen 2026, MMCL-Bench [DL]
- 2605.25706 Wu 2026, Towards Open-World Referring Expression Comprehension (OpenRef)
- 2605.26128 Ray 2026, The Constraint Tax
- 2606.02835 Caldarella 2026, Thinking Past the Answer
- 2606.07555 Wang 2026, Persistent Priors, Preserved Targets: A Stroop-Style Paradigm for Lexical Override [DL]
- 2606.28273 Lietzow 2026, Vision-Default, Prior-Override
- 2607.23386 Simpson 2026, Confidently Wrong: Exception Chain Collapse [DL]
- 2608.20237 Chen 2026, Rule-Compliant Visual Spatial Planning for MLLMs (RuleMaze)
- 2610.12942 Wang 2026, In-Context Learning Can Help VLMs Overcome Training Priors [DL]
