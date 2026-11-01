# Relation–set consistency (tools/eval/consistency.py)

Each cell: agreement (CI) · error coupling φ (CI). Agreement = share of scored candidates on which the two answers make the same in/out decision. φ = correlation of the two answers' correctness (1 = same errors, 0 = independent errors). M-contour, L2/L3 items unless noted. 95% CIs: image-clustered bootstrap, B = 1000.

## Table 2. Three ways to ask the same question (Lite, 1,497 Test items)

| Model | Same probe, two draws | Reworded rule (twins) | Set vs. own relations | … expected if independent | Set vs. own relations, L1 rules | Δφ retest − relations | Δφ reworded − relations | Δφ L1 − L2/L3 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B | 0.92 (0.90, 0.93) · φ 0.81 (0.77, 0.85) | 0.81 (0.76, 0.86) · φ 0.57 (0.47, 0.67) | 0.58 (0.56, 0.60) · φ 0.09 (0.05, 0.14) | 0.54 (0.53, 0.54) | 0.62 (0.58, 0.66) · φ 0.17 (0.10, 0.25) | 0.72 (0.67, 0.77) | 0.48 (0.37, 0.59) | 0.08 (-0.01, 0.16) |
| Qwen3-VL-8B | 0.91 (0.89, 0.92) · φ 0.78 (0.75, 0.82) | 0.83 (0.78, 0.87) · φ 0.59 (0.49, 0.69) | 0.69 (0.67, 0.72) · φ 0.30 (0.25, 0.35) | 0.56 (0.55, 0.57) | 0.72 (0.68, 0.76) · φ 0.30 (0.20, 0.39) | 0.48 (0.43, 0.54) | 0.29 (0.18, 0.39) | -0.00 (-0.10, 0.10) |
| InternVL3-8B | 0.85 (0.83, 0.87) · φ 0.66 (0.62, 0.70) | 0.79 (0.75, 0.82) · φ 0.54 (0.46, 0.62) | 0.59 (0.57, 0.61) · φ 0.11 (0.08, 0.15) | 0.54 (0.53, 0.54) | 0.59 (0.55, 0.63) · φ 0.08 (0.01, 0.15) | 0.55 (0.49, 0.61) | 0.42 (0.33, 0.51) | -0.03 (-0.11, 0.05) |
| Gemma-3-4B | 0.92 (0.90, 0.93) · φ 0.83 (0.80, 0.87) | 0.69 (0.64, 0.73) · φ 0.37 (0.28, 0.45) | 0.53 (0.50, 0.55) · φ 0.06 (0.01, 0.10) | 0.50 (0.50, 0.50) | 0.58 (0.54, 0.61) · φ 0.17 (0.10, 0.23) | 0.77 (0.72, 0.84) | 0.31 (0.22, 0.40) | 0.11 (0.03, 0.19) |
| Qwen3-VL-8B-Thinking | not run | 0.83 (0.80, 0.86) · φ 0.50 (0.43, 0.58) | 0.77 (0.75, 0.79) · φ 0.38 (0.34, 0.41) | 0.63 (0.62, 0.64) | 0.83 (0.80, 0.85) · φ 0.43 (0.36, 0.50) | — | 0.13 (0.05, 0.21) | 0.05 (-0.02, 0.12) |

Retest draws are sampled (T 0.7) where direct and probe answers are greedy, so the retest column is a conservative ceiling. The thinking model has no retest row yet (row 2 not run).

## Table S-C1. The same pairs on the full Test set (7,000 items; instruct models)

| Model | n cand. | Reworded rule (twins) | Set vs. own relations | … expected if independent | Set vs. own relations, L1 | φ within S* / within non-S* | Accuracy direct / executed |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B | 45839 | 0.80 (0.78, 0.82) · φ 0.56 (0.51, 0.60) | 0.58 (0.57, 0.59) · φ 0.10 (0.07, 0.12) | 0.53 (0.53, 0.54) | 0.59 (0.57, 0.61) · φ 0.11 (0.08, 0.15) | 0.09 (0.06, 0.12) / 0.13 (0.10, 0.15) | 0.61 (0.60, 0.62) / 0.65 (0.64, 0.66) |
| Qwen3-VL-8B | 45839 | 0.84 (0.82, 0.86) · φ 0.62 (0.57, 0.66) | 0.68 (0.67, 0.69) · φ 0.28 (0.26, 0.30) | 0.56 (0.55, 0.56) | 0.73 (0.71, 0.74) · φ 0.28 (0.24, 0.32) | 0.25 (0.22, 0.28) / 0.17 (0.14, 0.20) | 0.67 (0.66, 0.68) / 0.67 (0.66, 0.68) |
| InternVL3-8B | 45839 | 0.81 (0.79, 0.82) · φ 0.58 (0.54, 0.62) | 0.59 (0.58, 0.60) · φ 0.12 (0.10, 0.14) | 0.54 (0.54, 0.54) | 0.55 (0.54, 0.57) · φ 0.02 (-0.02, 0.05) | 0.14 (0.12, 0.17) / 0.14 (0.11, 0.16) | 0.62 (0.61, 0.62) / 0.67 (0.66, 0.68) |
| Gemma-3-4B | 45839 | 0.69 (0.67, 0.71) · φ 0.38 (0.34, 0.42) | 0.52 (0.51, 0.53) · φ 0.05 (0.03, 0.07) | 0.50 (0.50, 0.50) | 0.56 (0.54, 0.58) · φ 0.14 (0.11, 0.17) | 0.13 (0.10, 0.15) / 0.12 (0.09, 0.14) | 0.48 (0.48, 0.49) / 0.54 (0.53, 0.55) |

φ within S* / non-S*: the coupling computed separately on candidates that must be kept and on those that must be dropped, so a shared yes/no bias cannot create it.

## Table S-C2. Error coupling of set vs. own relations, by operator cell (φ, CI · n)


**Test, instruct models**

| Operator cell | Qwen3-VL-4B | Qwen3-VL-8B | InternVL3-8B | Gemma-3-4B |
|---|---|---|---|---|
| AND | 0.01 (-0.06, 0.07) · 2884 | 0.02 (-0.05, 0.08) · 2884 | 0.09 (0.03, 0.15) · 2884 | 0.01 (-0.05, 0.07) · 2884 |
| AND-NOT | 0.07 (0.02, 0.13) · 2378 | 0.09 (0.04, 0.14) · 2378 | 0.27 (0.22, 0.31) · 2378 | 0.14 (0.10, 0.18) · 2378 |
| EXCEPT | 0.16 (0.10, 0.21) · 1882 | 0.13 (0.07, 0.19) · 1882 | 0.25 (0.19, 0.31) · 1882 | 0.20 (0.14, 0.26) · 1882 |
| EXCEPT, base clause | 0.12 (0.07, 0.16) · 3963 | 0.18 (0.14, 0.23) · 3963 | 0.13 (0.09, 0.17) · 3963 | 0.19 (0.14, 0.23) · 3963 |
| L3 nested | 0.06 (-0.03, 0.16) · 594 | 0.18 (0.10, 0.27) · 594 | 0.09 (0.00, 0.18) · 594 | 0.01 (-0.08, 0.09) · 594 |
| CLOSURE | -0.00 (-0.08, 0.09) · 667 | 0.12 (0.02, 0.23) · 667 | 0.13 (0.04, 0.21) · 667 | 0.09 (0.01, 0.16) · 667 |

**Lite, all models**

| Operator cell | Qwen3-VL-4B | Qwen3-VL-8B | InternVL3-8B | Gemma-3-4B | Qwen3-VL-8B-Thinking |
|---|---|---|---|---|---|
| AND | 0.06 (-0.06, 0.19) · 601 | -0.00 (-0.12, 0.13) · 601 | 0.01 (-0.11, 0.12) · 601 | 0.01 (-0.09, 0.13) · 601 | 0.25 (0.14, 0.37) · 601 |
| AND-NOT | 0.11 (-0.02, 0.23) · 531 | 0.11 (-0.02, 0.22) · 531 | 0.35 (0.24, 0.44) · 531 | 0.15 (0.06, 0.24) · 531 | 0.53 (0.41, 0.64) · 531 |
| EXCEPT | 0.22 (0.11, 0.33) · 403 | 0.17 (0.04, 0.31) · 403 | 0.27 (0.14, 0.38) · 403 | 0.26 (0.15, 0.37) · 403 | 0.43 (0.31, 0.56) · 403 |
| EXCEPT, base clause | 0.09 (-0.01, 0.18) · 890 | 0.25 (0.15, 0.35) · 890 | 0.09 (0.01, 0.18) · 890 | 0.25 (0.15, 0.34) · 890 | 0.26 (0.17, 0.36) · 890 |
| L3 nested | -0.12 (-0.35, 0.14) · 116 | 0.28 (0.10, 0.45) · 116 | 0.19 (-0.01, 0.39) · 116 | 0.22 (0.03, 0.39) · 116 | 0.24 (0.03, 0.46) · 116 |
| CLOSURE | -0.12 (-0.19, -0.06) · 142 | 0.09 (-0.12, 0.31) · 142 | -0.00 (-0.17, 0.16) · 142 | 0.09 (-0.12, 0.23) · 142 | 0.51 (0.36, 0.65) · 142 |

## Table S-C3. Which way exceptions fail (direct answers, M-contour, %)

Under-applied: P(kept | the NOT/EXCEPT clause excludes it). Over-applied: P(dropped | in S*), over the same templates (definitions as Table 3 of results.md).


**Lite, all models**

| Operator | Rate | Qwen3-VL-4B | Qwen3-VL-8B | InternVL3-8B | Gemma-3-4B | Qwen3-VL-8B-Thinking |
|---|---|---|---|---|---|---|
| AND-NOT | under-applied | 65 (58, 71) | 53 (47, 58) | 55 (48, 60) | 49 (43, 54) | 17 (13, 22) |
| AND-NOT | over-applied | 30 (26, 34) | 39 (34, 43) | 40 (36, 44) | 37 (33, 41) | 43 (39, 47) |
| EXCEPT | under-applied | 69 (62, 75) | 57 (50, 64) | 63 (57, 69) | 69 (64, 75) | 25 (20, 31) |
| EXCEPT | over-applied | 19 (15, 23) | 30 (26, 35) | 29 (24, 33) | 27 (21, 32) | 35 (30, 40) |
| L3 nested | under-applied | 45 (32, 58) | 61 (49, 72) | 46 (34, 57) | 48 (39, 58) | 38 (27, 48) |
| L3 nested | over-applied | 55 (45, 64) | 39 (31, 49) | 40 (33, 47) | 37 (30, 45) | 53 (45, 61) |

**Test, instruct models**

| Operator | Rate | Qwen3-VL-4B | Qwen3-VL-8B | InternVL3-8B | Gemma-3-4B |
|---|---|---|---|---|---|
| AND-NOT | under-applied | 62 (59, 65) | 54 (51, 57) | 55 (52, 58) | 52 (49, 54) |
| AND-NOT | over-applied | 31 (29, 34) | 38 (36, 40) | 39 (37, 41) | 37 (35, 39) |
| EXCEPT | under-applied | 69 (66, 72) | 56 (53, 60) | 60 (57, 64) | 66 (63, 70) |
| EXCEPT | over-applied | 24 (22, 26) | 31 (28, 33) | 32 (30, 34) | 26 (23, 28) |
| L3 nested | under-applied | 48 (43, 53) | 61 (56, 66) | 41 (36, 46) | 58 (54, 62) |
| L3 nested | over-applied | 52 (48, 57) | 42 (38, 47) | 41 (37, 44) | 41 (37, 44) |

## Table S-C4. Paired model contrasts on Lite (Δ agreement · Δφ, CI)

Same Lite candidates and the same bootstrap draws for both models.

| Pair | thinking − no thinking (Qwen3-VL-8B) | 8B − 4B (Qwen3-VL, no thinking) |
|---|---|---|
| Set vs. own relations, L2/L3 | 0.08 (0.05, 0.10) · 0.08 (0.03, 0.14) | 0.11 (0.08, 0.14) · 0.21 (0.14, 0.27) |
| Set vs. own relations, L1 | 0.11 (0.06, 0.15) · 0.13 (0.02, 0.24) | 0.10 (0.04, 0.15) · 0.13 (0.01, 0.23) |
| Reworded rule (twins) | -0.00 (-0.06, 0.05) · -0.09 (-0.21, 0.03) | 0.02 (-0.04, 0.08) · 0.02 (-0.12, 0.14) |
| cell: AND | -0.05 (-0.13, 0.03) · 0.25 (0.08, 0.42) | 0.16 (0.06, 0.25) · -0.06 (-0.23, 0.11) |
| cell: AND-NOT | 0.33 (0.27, 0.39) · 0.42 (0.27, 0.57) | 0.03 (-0.05, 0.11) · 0.00 (-0.18, 0.17) |
| cell: EXCEPT | 0.20 (0.11, 0.30) · 0.26 (0.07, 0.43) | 0.01 (-0.07, 0.10) · -0.05 (-0.18, 0.11) |
| cell: EXCEPT, base clause | 0.00 (-0.06, 0.06) · 0.01 (-0.11, 0.14) | 0.17 (0.10, 0.24) · 0.17 (0.05, 0.28) |
| cell: L3 nested | -0.03 (-0.19, 0.13) · -0.04 (-0.34, 0.27) | 0.23 (0.03, 0.41) · 0.40 (0.03, 0.72) |
| cell: CLOSURE | -0.01 (-0.14, 0.14) · 0.42 (0.15, 0.70) | 0.08 (-0.07, 0.22) · 0.21 (-0.00, 0.44) |
