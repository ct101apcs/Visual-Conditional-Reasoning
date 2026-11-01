# Paper scaffold v1.0: relation–set consistency in rule-based grounding

Status: scaffold **v1.1**, 2026-11-01. v1.1 rewrites the abstract, §1 and §2 in conventional paper form: §1 runs context → gaps → approach → findings, and §2 states how ANCHOR differs from each line of work. The "Why not distractors" sidebar is folded into §1. v1.0 rebuilt the scaffold from scratch around what the data show. The previous scaffold (v0.2, "pairwise accuracy overstates object selection") is archived at [archive/paper_scaffold_v0.2_selection-gap.md](archive/paper_scaffold_v0.2_selection-gap.md). The design document [design_proposals.md](design_proposals.md) still describes the testbed correctly. Its headline (§0) and H1 (§6) are superseded by this file.

**Where the numbers come from**
- [data/eval/results/consistency.md](data/eval/results/consistency.md), produced by [tools/eval/consistency.py](tools/eval/consistency.py): Tables 2 and 3, Fig. 2, S2–S5.
- [data/eval/results/results.md](data/eval/results/results.md), produced by [tools/eval/analyze.py](tools/eval/analyze.py): Table 4 and the remaining supplementary tables.
- [tools/eval/MIGRATION.md](tools/eval/MIGRATION.md) maps the old scripts and table layouts onto this scaffold, and explains how to add the SAM 3 Agent row and the proprietary models.

**Models so far:**
- four open instruct VLMs on the full Test set (7,000 items): Qwen3-VL-4B, Qwen3-VL-8B, InternVL3-8B and Gemma-3-4B;
- Qwen3-VL-8B-Thinking on Lite only (1,497 Test items, result row 1).

**Pending:**
- proprietary models;
- the SAM 3 Agent baseline (running; 4,400 of 7,000 items done on 2026-11-01);
- the thinking model's rows 2–5;
- the class-list and segmenter baselines;
- the human audits.

Every claim that depends on the model set is entered in the **claim register** below and re-checked when the proprietary rows arrive.

**Format target:** IEEE MultiMedia. That means 6,500 words in total, with each figure or table counting as 200 words, at most 20 references, and an abstract of at most 150 words.

**Placeholders:** `[●]` marks a number not measured yet, and `[validate]` marks a sentence whose model range must be re-checked after the proprietary runs. Every other number below is measured.

---

## Writing rules

1. **Every abstract term gets a concrete example on first use**, taken from the running example where possible.
2. **The running example** is fixed. It was chosen in the design phase on 2026-10-28, before any model was run, so it is not cherry-picked.
   - Image `000000534270`, a seaside scene. Subject #15 is the seated woman.
   - Rule: *bordering, except larger* (T12).
   - S\* = {#11 umbrella hat, #14 umbrella, #16 dog}.
   - Measured outputs of Qwen3-VL-4B, M-contour:
     - set answer {#8, #12, #16}, F1 0.33;
     - the rule run on its own relation answers gives {#11, #12, #14, #16, #18}, F1 0.75.
3. **Say what is measured.** Write "covers more image pixels", not "is more prominent". Write "the outlines meet in the image", not "is near".
4. **Every number has a CI, and every comparison names its test.** For example: "Δφ 0.29 (95% CI 0.18 to 0.39; paired image-clustered bootstrap)".
5. **Vocabulary.**
   - The task is **rule-based set grounding**.
   - The two answers are the **set answer** and the **relation answers**.
   - The **executed set** is the rule program run on the relation answers.
   - The property measured is **relation–set consistency**.
   - Do not write "object selection ability", "distractor" (except about prior work and SADL), or "faithful".
6. **No mechanism claims.** We measure whether two answers agree. We do not claim to know how the model computes either of them.
7. **Conventional argument, plain sentences.**
   - The abstract and §1 run context → gap → what we do → what we find.
   - The introduction argues the *gaps*, each with citations: what VLMs can do, what is untested or unreliably measured, and why that matters.
   - Related work (§2) summarises each line of work and says *how ANCHOR differs in design*. It does not repeat the gaps.
   - IEEE MultiMedia asks for an educated general audience, so keep sentences short. Equations go in sidebar boxes. Use at most one level of subsection.
8. **Citations in the draft are tags** (`[CV-Bench]`, `[Rädsch 2025]`) that match the reference list below. Author names are written out only where the lit-review report confirms them. Tags are resolved to numbers at layout.

## Terms (for co-authors; the paper defines each on first use)

| Term | Meaning | Running example (Qwen3-VL-4B, T12) |
|---|---|---|
| Set answer | One call: the rule and the marked image in, the list of mark numbers out | `[8, 12, 16]` |
| Relation answers | One call per subject: every marked object × seven properties, as yes/no JSON. The call is shared by every rule on that subject | #8: borders yes, larger yes; #11: borders yes, larger no; … |
| Executed set | The rule program run on the relation answers | {#11, #12, #14, #16, #18} |
| Agreement | The share of marked objects on which two answers make the same in/out decision | 4 of 8 (#9, #12, #13, #16) |
| Independence level | The agreement expected if the two answers erred independently, given their accuracies | 0.54 for this model over Lite |
| Error coupling φ | The correlation between the two answers' correctness, per object: 1 means they make the same errors, 0 means their errors are independent | 0.09 for this model over Lite |
| Reference pairs | The same relation call drawn twice (*retest*), and the same rule worded two ways (*reworded*, the EXCEPT/AND-NOT twins) | T12 "bordering, except larger" vs. T11 "bordering and not larger" |
| Under- / over-applied exception | Keeping an object the NOT/EXCEPT clause removes / dropping one the rule keeps | keeping #8 / dropping #16 |
| Decomposition gain ΔG | Executed-set F1 minus set-answer F1 on L2/L3 rules, minus the same gap on L1 rules | −0.18 for this model (Test) |

---

## Claim register (frozen 2026-11-01; re-test when the proprietary rows arrive)

The analyses behind C1, C2 and C5 were not pre-registered. They were added after the pre-registered H1 failed (see the sidebar in §4). From today they are fixed: each test and pass criterion below is applied unchanged to every new model. Holm correction runs over C1–C5 within each new model.

| ID | Claim, as the paper words it | Test | Now (4 open instruct models + 1 thinking model) | Holds after the proprietary runs if | Wording if it fails |
|---|---|---|---|---|---|
| **C1** | A VLM's set answer does not follow from its own relation answers | Paired Δφ (reworded − relations) > 0, CI excluding 0; agreement within 0.15 of the independence level | **5/5.** Δφ 0.29–0.48 for instruct, 0.13 (0.05, 0.21) for thinking; agreement 0.53–0.69 against independence 0.50–0.56 for instruct, 0.77 vs 0.63 for thinking | Every proprietary model passes the Δφ test | If frontier models pass the agreement test but not the Δφ test, then "consistency is the capability that separates frontier from open models" becomes the headline, and C1 is scoped to open models |
| **C2** | The mismatch is there even for one relation, so it is not a failure to combine relations | Paired Δφ (L1 − L2/L3): CI includes 0, or is below 0 | **4/5.** Paired Δφ (L1 − L2/L3): Qwen-4B 0.08 (−0.01, 0.16), Qwen-8B −0.00 (−0.10, 0.10), InternVL3 −0.03 (−0.11, 0.05), thinking 0.05 (−0.02, 0.12). **Gemma fails: 0.11 (0.03, 0.19).** Even so, every model's L1 φ is low (0.08–0.43) | The same holds for every model | Already partly true: "composition lowers coupling further in [Gemma]". If the proprietary models fail too, write "the mismatch starts at one relation, and composition adds to it" |
| **C3** | Without thinking, exceptions are under-applied | under-applied > over-applied, non-overlapping CIs, in AND-NOT and EXCEPT cells | **4/4 instruct.** Under-applied 52–69%, over-applied 24–39% (Test). L3 nested is mixed: Qwen3-VL-4B over-applies (52% vs. 48%) | Every non-thinking mode passes | Scope it to open models, and report the direction per model |
| **C4** | Models adopt the definition stated in the prompt | Table S6c: inclusion of outline-meeting, contact-not-stated objects, "borders" − "touching", CI > 0 (M-contour) | **4/4.** 27–37 pp for Qwen and InternVL; 12 pp (5.8, 18.0) for Gemma | Every model passes | Report the per-model uptake, and drop "adopt" from the title |
| **C5** | Thinking couples the two answers where exceptions decide, and cuts under-application | Thinking − non-thinking, same model: Δφ in the AND-NOT and EXCEPT cells > 0; under-applied rate falls (paired) | **1 pair** (Qwen3-VL-8B, Lite). Δφ AND-NOT +0.42 (0.27, 0.57), EXCEPT +0.26 (0.07, 0.43); under-applied 53% → 17% and 57% → 25% | It holds for ≥ 2 thinking/non-thinking pairs, including one proprietary model with thinking on and off | Demote it to one sentence: "in one model pair" |
| C6 (secondary) | Without thinking, asking for the relations and running the rule gives *worse* sets than asking for the set | ΔG < 0, CI excluding 0 | **4/4 instruct** (−0.03 to −0.22); thinking −0.03 (−0.08, 0.02) | Report as measured; the sign is expected to follow relation accuracy | If it turns positive for strong models, report the crossover. Practitioners then decompose only with strong models |
| C7 (secondary) | The testbed is not trivial | Every VLM beats the class-list-only and random-at-prior rows on L2/L3 | L2: yes (0.37–0.49 vs. 0.27 at prior). **L3: no**: Qwen3-VL-8B scores 0.32 against 0.37 at prior. Class-list row not run | Proprietary models clear L3 by a margin | Write: "L3 rules sit at chance for open models" (a finding, not a flaw) |

---

# The paper

## Title

**Ask Twice, Get Two Answers: Relation–Set Consistency in Rule-Based Grounding with Vision–Language Models**

Alternatives:
- *The Set Does Not Follow: Vision–Language Models' Object Sets Disagree With Their Own Relation Judgements*
- *Bordering, Except the Big Ones: VLMs Adopt Our Definitions but Drop the Exceptions* (use this if C1 weakens for proprietary models and C3/C4 carry the paper)

## Abstract (≤150 words)

> Vision–language models (VLMs) answer spatial questions about pairs of objects and ground referring expressions. They are now asked to return sets defined by a rule, such as "everything bordering her, except the bigger things." Whether a model's relation judgements predict the set it returns is untested. Benchmarks score one pair at a time, and consistency studies lack ground truth for each sub-answer. We introduce ANCHOR, 7,000 items on human-verified panoptic masks, where a program computes every answer set. We ask each model for the set, and separately about each object's relations. For [four open] VLMs the two answers agree on 53–69% of objects, barely above the 50–56% expected from independent errors. Asking the same relation question twice agrees on 85–92%. The mismatch already appears for single-relation rules. Models adopt the definitions we state, but keep 52–69% of the objects an exception should remove. With thinking, this falls to 17–25%.

*(`[validate]`: model count and every range. Word count is in the budget check.)*

---

## 1. Introduction (≈850 words)

Paragraph plan: (1) what VLMs can do, and the use case; (2–4) three gaps, each argued from the literature; (5) what we do; (6) what we find; (7) contributions and scope. The bold bracket labels are for co-authors and are removed at layout. The introduction argues the *gaps*; §2 says how our design *differs* from each line of work.

> **[Context]** Vision–language models (VLMs) have made rapid progress in relating language to image content. They ground referring expressions, including expressions that pick out several objects or none [gRefCOCO/GREx]. They answer region-level spatial questions about size, contact and position [CV-Bench; SpatialRGPT]. These abilities make VLMs the natural front end for editing assistants and visual agents. Requests to such systems name a *subject* and a *rule*, and expect a *set* of objects back. Take "remove everything that borders her, except the bigger things" (Fig. 1). The answer is the umbrella hat (#11), the umbrella (#14) and the dog (#16). The man (#8) also borders her, but he covers more of the picture, so the exception removes him.
>
> **[Gap 1: evaluation stops at the pair]** Yet how well VLMs return such sets is mostly inferred from pairwise tests. Spatial benchmarks derive questions from masks or depth maps and ask about one object or one pair at a time, with binary or multiple-choice answers [Rädsch 2025; CV-Bench; SpatialRGPT]. Generalized referring benchmarks do return sets, but from free-form expressions written by annotators. No reference object is given, and the ground truth depends on how annotators read each phrase [gRefCOCO/GREx]. The step from relations to sets is assumed rather than measured. To our knowledge, no benchmark asks the same model both questions with computed ground truth for each.
>
> **[Gap 2: consistency is measured without per-answer ground truth]** A second line of work tests whether models answer consistently across forms of a question. VQA models answer reasoning questions correctly while failing the perception sub-questions those answers depend on [Selvaraju 2020]. Language models reject, as validators, answers they produced as generators; GPT-4 is consistent in this sense only 76% of the time [Li 2024]. Large VLMs change their answers when the answer space of a prompt changes [ConBench]. These studies count how often two answers *match*. Without ground truth for each answer, a match cannot be told apart from chance agreement. Two answers that are each right 60% of the time agree on about half of all cases even if they are unrelated. Whether two answers share their errors has therefore not been measured.
>
> **[Gap 3: exceptions over object sets are untested]** Rule following is also less reliable than it looks. VLMs drift from a stated rule toward its familiar meaning [VLM-Fix]. Contrastive VLMs show an affirmation bias [NegBench]. In referring expressions, a missed "not" is the usual cause of failure [Tumu 2025]. Text-only language models collapse on nested exceptions [Simpson 2026]. We found no VLM benchmark that tests exceptions over sets of objects in natural images, or reports which way such errors go. Our earlier attempt observed over-applied exclusions, but under categories annotated by judgement ("is this object distracting?"), on which annotators agreed only moderately (κ = 0.53) [SADL]. Rule following needs rules whose answers can be computed.
>
> **[What we do]** We introduce ANCHOR, a testbed for *rule-based set grounding*. Given an image with numbered objects, a subject and a rule with its operational definition, a model must return every object for which the rule holds. The rules combine six measurable relations with AND, NOT, EXCEPT and a closure operator. The ground truth is computed by running each rule as a program on human-verified panoptic masks [COCONut-PanCap], and borderline objects are not scored. We ask every model the question two ways. The **set answer** is the list of objects it returns. The **relation answers** are its yes/no judgements of each relation for each object, which we run through the same program. Two reference pairs, the same relation question asked twice and the same rule reworded, show how much disagreement re-asking alone produces. Because every answer has ground truth, we measure both agreement and **error coupling**: whether the two answers make the same mistakes.
>
> **[What we find]** Across 7,000 items and [four open] VLMs `[validate]`:
> - **The set answer barely follows from the model's own relation answers.** They agree on 53–69% of objects, only 3–13 points above the 50–56% that independent errors would give. Asking the same relation question twice agrees on 85–92%, and rewording the rule on 69–83%. In Fig. 1, Qwen3-VL-4B judges correctly that the man borders the woman and is larger, yet keeps him in its set.
> - **The mismatch exists before relations are combined.** For four of five models it is as large for single-relation rules as for composed ones.
> - **Models adopt our definitions but under-apply exceptions.** They follow the stated meaning of "borders" (+27–37 points on objects that only overlap the subject), yet keep 52–69% of the objects a NOT or EXCEPT clause removes.
> - **Thinking helps where exceptions decide.** With thinking, Qwen3-VL-8B keeps only 17–25% of such objects, and its two answers become coupled in exactly those cases.
>
> **[Contributions]** We make three contributions:
> 1. ANCHOR: 16 executable rule templates with computed set-valued ground truth on natural images, released with the generator.
> 2. A per-object consistency measurement of set answers against a model's own relation answers, calibrated by chance agreement and two reference pairs.
> 3. An evaluation of [five] VLMs `[validate]` that yields the four findings above.
>
> We make no claims about human attention or editing quality. We measure only whether a model's answers to a stated rule agree with the program, and with each other.

**Figure 1 (≈200 words of budget). One model, two ways to ask.**
- *Left:* the running-example image with the magenta subject contour and the number marks. Draft render: [figures/fig1_draft_running_example.png](figures/fig1_draft_running_example.png).
- *Middle (relation answers):* a compact table of Qwen3-VL-4B's measured relation answers for the eight objects (borders / larger). Wrong cells are shaded: #12 and #18 "borders" is wrong. The executed set is {#11, #12, #14, #16, #18}, F1 0.75.
- *Right (set answer):* {#8, #12, #16}, F1 0.33. Arrows mark the three disagreements that matter: #8 kept although its own answers exclude it, and #11 and #14 dropped although its own answers include them. Agreement is 4 of 8 objects.
- *Caption line:* "Measured outputs (Qwen3-VL-4B, M-contour, greedy). The example was fixed before any model was run. Qwen3-VL-8B gets this set nearly right ({#11, #14, #16, #18}, F1 0.86), while its own relation answers would add #12, so the mismatch runs both ways."

---

## 2. Related work (≈550 words, 0 floats)

Each paragraph summarises one line of work, then states how ANCHOR differs in design. The gaps themselves were argued in §1 and are not repeated here.

> **Spatial and relational evaluation.** Early probes matched images to captions with spatial words. Later benchmarks generate questions from annotations. CV-Bench [Tong 2024] and SpatialRGPT-Bench [SpatialRGPT] ask region-level questions about position, depth and size. Rädsch et al. [2025] derive 25 tasks per image from COCO, COCONut and KITTI masks, with thresholds and no language model in the ground truth. Their answers are binary or four-way, about single objects or pairs. ANCHOR adopts their style of predicate: relations computed from masks, numeric margins, and an explicit unscored band. It differs in three ways. The answer is a set. The relations are anchored to a given subject and composed by rules. And the pairwise judgement is not the score but one of two answers from the same model, compared against the other.
>
> **Referring and set grounding.** Generalized referring expression comprehension extends the one-expression, one-object task to several targets or none [gRefCOCO; GREx]. Reasoning segmentation moves to implicit queries. SAM 3 [Carion 2025] returns every instance of a concept from text or exemplar prompts. Set-of-Mark prompting [Set-of-Mark] lets a VLM answer with the numbers of pre-segmented regions. ANCHOR differs in what defines the target. Its queries are not free-form expressions but a given subject plus an executable rule, so the ground truth does not depend on how a phrase is read. Answers are mark numbers over ground-truth segments, so no segmenter's errors enter the score. SAM 3 enters as a baseline. Its concept search covers "same kind as the subject", but not size, contact, position or exceptions relative to that subject.
>
> **Consistency across ways of asking.** VQA-Introspect pairs reasoning questions with perception sub-questions and finds that models answer the former without the latter [Selvaraju 2020]. GQA adds a consistency metric over entailed questions [Hudson 2019]. Generator–validator consistency checks whether a language model accepts its own answers, and improves it by fine-tuning on consistent pairs [Li 2024]. ConBench varies the answer space of a question and measures how often large VLMs keep their answer [ConBench]. These works score agreement between answers. ANCHOR adds three things:
> - ground truth for every sub-answer, which separates shared errors from independent ones;
> - an exact executor, which turns relation answers into a set without a judge model;
> - two reference pairs (re-asking and rewording), which calibrate how much disagreement any second question produces.
>
> **Rule following, negation and exceptions.** VLM-Fix changes game rules and shows that 14 VLMs drift back toward the familiar rule; neutral aliases reduce the drift and loaded aliases restore it [Alam 2026]. NegBench documents an affirmation bias in contrastive VLMs [NegBench]. Tumu et al. [2025] trace referring failures to ignored negation. Simpson [2026] finds that text-only models collapse as exceptions nest. Our earlier SADL benchmark reported over-applied exclusions under annotated categories. ANCHOR tests negation and exception in a different setting: over sets of objects in natural images, with answers computed rather than annotated. Errors are reported with their direction. Under- and over-application are each computed only over objects where that error is possible, which lets the two documented directions be compared on one footing. The definition-uptake test asks VLM-Fix's question of a geometric definition rather than a game rule. The thinking contrast holds the model fixed and changes only the decoding mode.

**Sidebar dropped (was "Why not just label distractors?").** Gap 3 in §1 now carries the SADL argument in one sentence. If a reviewer asks for more, restore it as a supplementary note that gives our earlier attempt in two sentences, the κ of 0.53, and the goal- and time-dependence of attention [Zhang et al.].


---

## 3. The ANCHOR testbed (≈900 words; Table 1)

### 3.1 The task in one sentence, then the running example
"Given an image, a subject and a rule, return every other marked object for which the rule is true." Walk through Fig. 1 with a second rule, L2 *left of the subject and not a person*. Its set is the boat, two umbrellas and the handbag, {#9, #12, #13, #18}. The standing man is left of her but is a person, so he is out.

### 3.2 Relations, each with its wording and margin
Each relation gets one sentence and one example from Fig. 1. This is the prose form of Table 1.
- **larger**: more image pixels, by a ratio above 1.5. "The man (#8) covers 1.64× her pixels, so yes. The boat (#9) covers 0.22×, so no, even though boats are bigger than people."
- **borders**: the outlines meet in the image, with a gap of at most 1.5 px. A gap above 2% of the image diagonal counts as "no", and anything in between is not scored. "The umbrella hat (#11) borders her; the small umbrella on the wall (#13) does not." Say plainly that this is not physical contact: in a pilot check, a quarter of bordering objects met the subject only because one stood in front of the other, and every prompt says so.
- **left / above**: the object's centre lies beyond the subject's extreme point, with a ±2% band.
- **same kind**: the same COCO category.
- **reaches the image edge**: some pixel of the object lies on the image edge.

### 3.3 Rules
- **Level** is the number of relations a rule uses. The running example "bordering, except larger" is L2. The L3 rules are a nested exception and a closure.
- **EXCEPT** means "the exception wins". **CLOSURE** is "the same kind as her, plus anything bordering one of those".
- **Twins** are two wordings with the same answer set: T12 "bordering, except larger" and T11 "bordering and not larger". They are the "reworded" reference pair of §4.

### 3.4 Pointing at the subject
- **M-contour**, a magenta outline, is the primary condition.
- **M-text** is the caption phrase; **M-mark** gives the subject a number like every other object.
- The pointing effects are small (0.02–0.07 F1) and go to Table S10, in one sentence.

### 3.5 From masks to items
- **Source:** COCONut-PanCap.
- **Subject eligibility:** a thing segment covering 2–60% of the image, not a crowd, with a unique caption phrase that names its own category.
- **Candidates:** 4–20 marked objects.
- **Sampling** is by answer-set size, so about 15% of items have the empty set as the answer. "Otherwise answering 'none' would already score up to 53% exact match."
- **Totals:** 7,000 Test items on 3,120 subjects in 2,596 images. Lite is a 1,497-item subset drawn as whole subjects.

### 3.6 Is the ground truth right?
- **GT audit:** [●] items, error [●]% (CI).
- **Solvability audit:** [●]%.
- **Margin sweep:** the conclusions are unchanged (Table S8).

These are label-quality checks, not claims about human performance.

**Table 1 (≈200 words).** Columns: relation | wording shown to the model | margin | % positive | % not scored | audit error (CI). One extra row lists the operators with an example rule each.

---

## 4. Three ways to ask the same question (≈650 words; sidebar with formulas)

### 4.1 The set answer and the relation answers
- **The set answer** is one call: the rule plus the marked image in, a list of numbers out. Fig. 1 example: `[8, 12, 16]`.
- **The relation answers** come from one call per subject. The model answers seven yes/no properties for every marked object, as JSON. Fig. 1 example: `{"#8": {"borders": true, "larger": true, …}, …}`. The call is shared by every rule on that subject, so any rule can be executed offline. The executor itself is verified: run on the ground-truth relations, it scores F1 = 1.0 on all 7,000 items.
- **Both calls** list the mark numbers, use greedy decoding, and parse at ≥ 99.9% (Table S7).

### 4.2 Agreement, and what agreement by chance looks like
For each scored object we have two in/out decisions:
- **Agreement** is the share of objects where the decisions match. In Fig. 1 it is 4 of 8.
- But two answers that are each right 60% of the time agree on about half the objects by chance. So we also report the **independence level**: the agreement expected if the two answers erred independently, given their accuracies.
- **Error coupling φ** is the correlation between the two answers' correctness. φ = 1 means they make the same errors; φ = 0 means their errors are unrelated.
- Because the ground truth is fixed, agreement and φ come from one 2×2 table (sidebar).
- We also compute φ separately on objects that must be kept and on objects that must be dropped. A shared yes- or no-bias therefore cannot create the coupling.

### 4.3 Two reference pairs
How much disagreement does simply asking twice produce? Two reference pairs answer that:
- **Retest:** the relation call drawn twice with sampling (T 0.7). Each draw is run through the rule. This gives the ceiling for "the same question, asked again". It is conservative, because sampled draws disagree more than greedy ones.
- **Reworded:** the set answers to the two twin wordings on the same subject. This gives the level for "a different prompt, the same answer set".

The headline contrast is paired: φ(reworded) − φ(set vs. relations), on the same objects and the same bootstrap draws.

### 4.4 Which way exceptions fail
- **Under-applied:** P(kept | the NOT or EXCEPT clause removes it). Keeping #8 is an example.
- **Over-applied:** P(dropped | the rule keeps it). Dropping #16 is an example.
- Each rate is computed only over objects where that error is possible.

### 4.5 Statistics
- **CIs** come from an image-clustered bootstrap: 1,000 resamples of 2,596 images, where every item and object of a drawn image comes with it.
- **Model contrasts** are paired: the same objects and the same draws.
- **Holm correction** runs over the claim register's tests C1–C5 within each model.

**Sidebar: definitions.**
- The 2×2 table (a: both right, b: only the set answer right, c: only the executed set right, d: both wrong).
- Agreement = (a + d)/N. Independence level = pₛpₑ + (1 − pₛ)(1 − pₑ). φ = (ad − bc)/√((a+b)(c+d)(a+c)(b+d)).
- Set F1 with the empty-set convention. ΔG.

**Sidebar (≈150 words): "What we expected, and what we changed."**
> We pre-registered the opposite story. Running a model's own relation answers through the rule should beat its set answer, which would mean that pairwise accuracy overstates what models can do with sets. For every open model it did the reverse (§5.5). Looking for the reason, we found that the two answers barely track each other. That analysis was not pre-registered. We froze its tests before running further models, and we report the original hypothesis and its result in full (Table 4).

---

## 5. What we found (≈1,100 words; Tables 2–4, Fig. 2)

Each subsection opens with the claim in **bold**, then gives the number, then one concrete item.

### 5.1 The set does not follow from the model's own relation answers (C1)

**Table 2 (≈200 words). Three ways to ask the same question.** Lite, 1,497 Test items, L2/L3 rules, M-contour. Each cell gives agreement · error coupling φ, with 95% CIs.

| Model | Same relation call, two draws | Reworded rule (twins) | **Set vs. own relations** | Independence level | Set vs. own relations, L1 rules | Δφ reworded − set/relations |
|---|---|---|---|---|---|---|
| Qwen3-VL-4B | 0.92 · 0.81 (0.77, 0.85) | 0.81 · 0.57 (0.47, 0.67) | **0.58 · 0.09 (0.05, 0.14)** | 0.54 | 0.62 · 0.17 (0.10, 0.25) | 0.48 (0.37, 0.59) |
| Qwen3-VL-8B | 0.91 · 0.78 (0.75, 0.82) | 0.83 · 0.59 (0.49, 0.69) | **0.69 · 0.30 (0.25, 0.35)** | 0.56 | 0.72 · 0.30 (0.20, 0.39) | 0.29 (0.18, 0.39) |
| InternVL3-8B | 0.85 · 0.66 (0.62, 0.70) | 0.79 · 0.54 (0.46, 0.62) | **0.59 · 0.11 (0.08, 0.15)** | 0.54 | 0.59 · 0.08 (0.01, 0.15) | 0.42 (0.33, 0.51) |
| Gemma-3-4B | 0.92 · 0.83 (0.80, 0.87) | 0.69 · 0.37 (0.28, 0.45) | **0.53 · 0.06 (0.01, 0.10)** | 0.50 | 0.58 · 0.17 (0.10, 0.23) | 0.31 (0.22, 0.40) |
| Qwen3-VL-8B-Thinking | not run (row 2) | 0.83 · 0.50 (0.43, 0.58) | **0.77 · 0.38 (0.34, 0.41)** | 0.63 | 0.83 · 0.43 (0.36, 0.50) | 0.13 (0.05, 0.21) |
| *Proprietary models* | [●] | [●] | [●] | [●] | [●] | [●] |

The agreement CIs are in [consistency.md](data/eval/results/consistency.md). On the full 7,000-item Test set (Table S2), the instruct models give the same values to within 0.02.

**Draft text.**
> **For every model, the set answer is the least consistent of three ways to ask.** Asking for the relations twice agrees on 85–92% of objects (φ 0.66–0.83). Rewording the rule agrees on 69–83% (φ 0.37–0.59). The set answer and the rule run on the model's own relation answers agree on only 53–69% of objects. That is 3–13 points above what two independent answers with the same accuracies would reach (φ 0.06–0.30). The paired drop from the reworded pair is 0.29–0.48 in φ for the instruct models, and 0.13 (CI 0.05 to 0.21) for the thinking model, with every CI excluding 0 `[validate]`. Figure 1 is typical rather than extreme: Qwen3-VL-4B's 4-of-8 agreement there sits close to its average of 0.58.

### 5.2 It is not about combining relations (C2)

> **The mismatch is already there when the rule has one relation.** On L1 rules such as "bordering the subject", the set answer and the executed set apply the same single relation, and there is nothing to combine. Their coupling is still only φ 0.08–0.30 for the instruct models. For four of five models it is indistinguishable from the coupling on L2/L3 rules (paired Δφ −0.03 to 0.08, every CI includes 0). Only Gemma loses a further 0.11 (CI 0.03 to 0.19) when relations are combined. It is not an artefact of the long relation call either:
> - the call is as accurate as a rule-specific call (Δ −1.4 to +2.8 pp, every CI includes 0);
> - the executor scores F1 1.0 on the true relations;
> - the coupling stays low within objects that must be kept (φ 0.09–0.25) and within objects that must be dropped (0.12–0.17), so a shared yes-bias does not explain it (Tables S2, S7).
>
> Listing a set and judging each object one by one are, for these models, two different answers to the same question.

*Scaffold note:* the paired Δφ (L1 − L2/L3) values come from [consistency.md](data/eval/results/consistency.md), Table 2, last column. Re-check the "four of five" wording when the proprietary rows arrive.

### 5.3 Models adopt the definition, but drop the exception (C3, C4)

**Table 3 (≈200 words). What models do with a stated rule.** Direct set answers, M-contour, % (95% CI). Instruct models on Test; the thinking model on Lite.

| Row | Qwen3-VL-4B | Qwen3-VL-8B | InternVL3-8B | Gemma-3-4B | Qwen3-VL-8B-Thinking |
|---|---|---|---|---|---|
| *Definition uptake:* "borders" − "physically touching", objects whose outlines meet but whose contact the caption does not state (pp) | 37 (30, 44) | 31 (24, 37) | 27 (21, 35) | 12 (6, 18) | not run (row 3) |
| AND-NOT: under-applied | 62 (59, 65) | 54 (51, 57) | 55 (52, 58) | 52 (49, 54) | **17 (13, 22)** |
| AND-NOT: over-applied | 31 (29, 34) | 38 (36, 40) | 39 (37, 41) | 37 (35, 39) | 43 (39, 47) |
| EXCEPT: under-applied | 69 (66, 72) | 56 (53, 60) | 60 (57, 64) | 66 (63, 70) | **25 (20, 31)** |
| EXCEPT: over-applied | 24 (22, 26) | 31 (28, 33) | 32 (30, 34) | 26 (23, 28) | 35 (30, 40) |
| L3 nested exception: under / over | 48 / 52 | 61 / 42 | 41 / 41 | 58 / 41 | 38 / 53 |
| "except" vs. "and not" wording (twins): Δ under-applied (pp) | +11 (8, 15) | +9 (6, 12) | +2 (−1, 6) | +8 (4, 12) | not run |
| *Proprietary models* | [●] | | | | |

**Draft text.**
> **Models take up the definitions we state.** Our "borders" includes objects that merely overlap the subject in the picture. Told this, models include 27–37 points more of such objects than when asked about physical touching (12 points for Gemma, whose relation answers are near chance). So the testbed measures the stated rule, not the model's own idea of touching.
>
> **But they leave exceptions unapplied.** Where a NOT or EXCEPT clause must remove an object, as with the man in Fig. 1, the instruct models keep it 52–69% of the time. They wrongly drop only 24–39% of the objects the rule keeps. The word "except" makes this worse than "and not" for three of four models (+8 to +11 points on the same answer sets). Restating the negation positively ("a different kind" instead of "not the same kind") changes nothing (Table S6b). Nested L3 exceptions fail in both directions at similar rates.

### 5.4 Thinking couples the answers where exceptions decide (C5)

**Fig. 2 (≈200 words). Error coupling by the cell that decides each object.**
- *Layout:* a dot plot. x is φ (set vs. own relations); there is one row per operator cell (AND, AND-NOT, EXCEPT, EXCEPT base clause, L3, CLOSURE).
- *Marks:* grey dots for the four instruct models; a filled marker for Qwen3-VL-8B-Thinking; an open marker of the same colour for Qwen3-VL-8B, its non-thinking twin. CIs as whiskers. The data come from Table S3 (Lite).

**Draft text.**
> **Thinking cuts under-application by more than half, and it couples the two answers exactly where exceptions decide.** On the same Lite objects, Qwen3-VL-8B with thinking keeps 17% of the objects an AND-NOT clause removes, against 53% without it. Under EXCEPT the figures are 25% against 57%. In those cells its set answer and its relation answers now share their errors: Δφ +0.42 (0.27 to 0.57) for AND-NOT and +0.26 (0.07 to 0.43) for EXCEPT. CLOSURE shows the same jump (+0.42), while the cell where the base clause alone decides does not move (+0.01). Overall the coupling rises only modestly (+0.08, CI 0.03 to 0.14; Table S4). Scale works differently. Qwen3-VL 8B over 4B couples the answers more overall (+0.21) but not in the exception cells. The cost of the fix: thinking now over-applies AND-NOT slightly more often than it under-applies it (43% vs. 17%).

Caveats, to state in the text:
- this is one thinking model, on Lite only;
- it used sampled decoding (T 1.0) where the instruct models used greedy decoding;
- 0.5% of its set answers were truncated.

`[validate]` with the proprietary thinking on/off pairs.

### 5.5 What this costs in set accuracy (C6, C7)

**Table 4 (≈200 words). Set accuracy, and what decomposing the question buys.** Test, M-contour, set F1; ΔG with its 95% CI. This is the former Table 2, reorganised. It reports the pre-registered H1.

| Model | Relation accuracy (replica / call, balanced) | Set answer F1 L1 / L2 / L3 | Executed set F1 L2/L3 | ΔG | M-text set F1 L2/L3 (for the SAM 3 Agent comparison) |
|---|---|---|---|---|---|
| Answer `[]` | — | 0.36 / 0.15 / 0.15 | — | — | — |
| Random at prior | — | 0.24 / 0.27 / 0.37 | — | — | — |
| Class list only, no image | not run | | | | |
| SAM 3 Agent (driver: Qwen3-VL-8B; raw image + M-text rule) | — | [●] (running) | — | — | [●] |
| Segmenter + rule program | not run | | | | |
| Qwen3-VL-4B | 0.67 / 0.62 | 0.43 / 0.48 / 0.40 | 0.38 | −0.18 (−0.20, −0.15) | [●] |
| Qwen3-VL-8B | 0.67 / 0.64 | 0.57 / 0.49 / 0.32 | 0.42 | −0.03 (−0.06, −0.00) | [●] |
| InternVL3-8B | 0.69 / 0.59 | 0.40 / 0.44 / 0.45 | 0.30 | −0.20 (−0.23, −0.17) | [●] |
| Gemma-3-4B | 0.50 / 0.51 | 0.31 / 0.37 / 0.48 | 0.26 | −0.22 (−0.25, −0.19) | [●] |
| Qwen3-VL-8B-Thinking (Lite) | 0.77 / 0.75 | 0.63 / 0.55 / 0.52 | 0.56 | −0.03 (−0.08, 0.02) | not run |
| *Proprietary models* | [●] | | | | |

**Draft text.**
> **Neither route to the set is reliable, and for open models decomposing the question is worse.** The best instruct models select at F1 0.44–0.49 on L2 rules, against 0.27 for guessing at the prior. On L3 rules they reach 0.32–0.48, around the 0.37 that guessing reaches. Running each model's relation answers through the rule loses a further 0.03–0.22 beyond the L1 floor. This is the reverse of our pre-registered hypothesis, and it survives every margin and reference variant we tried (Table S8). With thinking the two routes tie (−0.03, CI −0.08 to 0.02). §5.1 explains why decomposing cannot help a model whose two answers are this weakly coupled. Its relation answers are only 59–64% accurate (balanced), and the set answer does not draw on them. Shortcuts hurt both routes. Where the boxes of two objects overlap but their outlines do not meet, as with the handbag (#18) in Fig. 1, set-answer accuracy falls by 21–37 points (Table S9).

SAM 3 Agent sentence (fill when the run finishes): "The SAM 3 Agent, driven by the same Qwen3-VL-8B, reaches [●] on 'same kind' but [●] on 'bordering, except larger'. Its tools find objects by concept and cannot test a relation to the subject." Compare it against the **M-text** set-answer column, because the agent gets the raw image and the rule in words.

---

## 6. Lessons (≈400 words)

Each bullet comes with one example:
- **For benchmark builders:** a model's pairwise relation accuracy describes a different answer from the set it returns. Report the set answer, and report the agreement between the two answers against an independence level. Agreement alone flatters weak models: two coin flips agree half the time.
- **For practitioners:**
  - *Do not swap one set call for relation calls plus code* with open instruct models. On our rules this loses 0.03–0.22 F1. With a thinking model, the two routes tie.
  - Disagreement between the two routes is cheap to compute. Whether it flags wrong sets is `[to test]`: compute P(set correct | the routes agree) against P(set correct | they disagree), per object.
  - Expect kept exceptions. When a request contains "except", check the excluded objects first. In Fig. 1 that is the man.
- **For model builders:** exceptions are where open models fail, and where thinking helps most. Consistency training of the generator–validator kind [Li 2024] is a natural next step: make the set answer agree with the model's own relation answers.

**Limitations (one honest paragraph).**
- The relations are 2D by design, though models do take up the stated definition (C4).
- COCO images are probably in the pretraining data.
- The captions behind M-text were drafted by GPT-4V.
- The data carry a research-only license.
- Crowds and ambiguous subjects are excluded.
- The analysis behind C1, C2 and C5 is exploratory. Its tests were frozen before the proprietary runs.
- The thinking result rests on [one] model pair.
- We make no claims about human attention.

## 7. Conclusion (≈150 words)

Return to the opening request: "everything bordering her, except the bigger stuff."
- Asked for the set, the model keeps the man.
- Asked about him, the model knows he borders her and covers more of the picture.
- Those two answers barely track each other in [four] of [five] models `[validate]`. The exceptions are where they part most, and where thinking reunites them.

Close with one sentence on the release: the generator, items, prompts, every model output, and the audit data.

---

## Floats (6 × 200 = 1,200 words)

| Float | Content | Source |
|---|---|---|
| Fig. 1 | Running example: the relation answers vs. the set answer (measured, Qwen3-VL-4B) | `figures/` + the runs (MIGRATION.md §5) |
| Table 1 | Relations and operators: wording, margin, % positive, % not scored, audit error | `data/anchor_gt/summary.json` + audit |
| Table 2 | Three ways to ask: agreement · φ | consistency.md, Table 2 |
| Table 3 | Definition uptake and exception direction | consistency.md S-C3 + results.md S3 |
| Fig. 2 | φ by operator cell, thinking vs. non-thinking | consistency.md S-C2 (Lite) + S-C4 |
| Table 4 | Set accuracy, executed set, ΔG, baselines incl. the SAM 3 Agent | results.md Table 2 + 2-Lite + SAM 3 run |

## Supplementary tables (do not count toward the limit)

Old → new numbering is in [MIGRATION.md](tools/eval/MIGRATION.md) §2.

| Table | Question it answers | Supports | Source | Status |
|---|---|---|---|---|
| S1 | What is in the Test set? | testbed | results.md S1 | done |
| S2 | Does Table 2 hold on all 7,000 Test items, and within kept/dropped objects? | C1, C2 | consistency.md S-C1 | done (instruct models) |
| S3 | Error coupling by operator cell | C5, Fig. 2 | consistency.md S-C2 | done |
| S4 | Paired contrasts: thinking and scale | C5 | consistency.md S-C4 + results.md S8 | done (1 thinking pair) |
| S5 | Exception direction, Lite and Test | C3 | consistency.md S-C3 | done |
| S6 | Wording and definition ablations | C3, C4 | results.md S3 | done; the thinking model needs row 3 |
| S7 | Is the mismatch a relation-call artefact? | C1, C2 | results.md S7 | done; the thinking model needs row 2 |
| S8 | Robustness to margins, ignore band and M-text | C1, C6 | results.md S2, plus φ under each variant | ΔG done; **φ under the variants still to add** |
| S9 | Shortcut (counter-heuristic) objects | §5.5 | results.md S4 | done |
| S10 | How you point at the subject | §3.4 | results.md S9 | done |
| S11 | What makes an object hard (GLM) | §5.5 | results.md S5 | done; demoted from the main text |
| S12 | Relation and operator difficulties (LLTM) | C3 | results.md S6 | done; demoted |

## Budget check

| Part | Words |
|---|---|
| Abstract | 150 |
| §1 Introduction | 880 |
| §2 Related work | 550 |
| §3 Testbed | 900 |
| §4 Measurement + sidebar | 650 + 150 |
| §5 Findings | 1,070 |
| §6 Lessons | 400 |
| §7 Conclusion | 150 |
| **Text subtotal** | **≈ 4,900** |
| Floats (6) | 1,200 |
| References (20) + bios | ≈ 400 |
| **Total** | **≈ 6,500 / 6,500** |

§5 must lose 30 words against its draft: shorten the §5.5 shortcut sentence first. If the venue becomes IEEE TMM, promote S2, S3 and S8 into the main text and restore the pointing results (S10) as a subsection.

## References (≤ 20)

Each entry gives the reason it stays. IDs for entries 1–9 and 13–18 are in [the lit-review report](lit-review/reports/subject-conditioned-object-reasoning.md). Entries 10–12 were checked on 2026-11-01 and added to the closest-work list.

1. SADL (our earlier attempt; motivation)
2. COCONut-PanCap (data)
3. COCO (images)
4. Rädsch et al. 2025 (pairwise mask-derived relations)
5. CV-Bench
6. SpatialRGPT
7. gRefCOCO / GREx (set grounding)
8. SAM 3 (baseline)
9. Set-of-Mark (number marks)
10. **Selvaraju et al. 2020, SQuINT**, arXiv 2001.06927, CVPR 2020 (perception–reasoning inconsistency in VQA)
11. **Li et al. 2024, generator–validator consistency**, arXiv 2310.01846, ICLR 2024 (GV-consistency; consistency fine-tuning)
12. **Zhang et al. 2024, ConBench**, arXiv 2405.14156, NeurIPS 2024 (LVLM consistency across answer spaces)
13. VLM-Fix (Alam 2026) (rule drift; C4)
14. NegBench (negation)
15. Tumu 2025 ("not" ignored)
16. Exception Chain Collapse (nested exceptions)
17. Zhang et al. (temporal attention; supplementary note only. Drop it if the reference slot is needed)
18. GQA (programmatic ground truth and the consistency metric)
19. Miller 2024 (clustered errors; statistics)
20. Qwen3-VL technical report (the model family behind three of five rows)

Dropped since v0.2: MUSE/MMR, VPBench, VP-Bench and MMStar (blind baselines). Re-add MMStar if the class-list row runs.
