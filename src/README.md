# Predicting Answer Correctness from Top-10 Probability Curves

This project predicts whether a generated Turkish GSM8K answer is correct using
only the token-level confidence curve produced during generation. For every
generated token, the stored signal is the sum of the probabilities of the top-10
most likely next tokens. Lower values are interpreted as moments where the model
is less certain.

[GitHub link](https://github.com/ozguraslank/ytu-nlp-project)

## Dataset

The source data comes from local 250-question train/test samples of
`ytu-ce-cosmos/gsm8k_tr`.

For each question, `unsloth/Qwen3-4B` generates 10 answers. We keep only mixed
questions: questions where at least one generated answer is correct and at least
one is incorrect. This prevents the classifier from learning only question
difficulty.

Final data:

| Split | Questions | Answers | Correct | Incorrect |
|---|---:|---:|---:|---:|
| Train | 68 | 680 | 376 | 304 |
| Test | 64 | 640 | 336 | 304 |
| Total | 132 | 1320 | 712 | 608 |

The generated answer records are stored in `answers.jsonl`.

## Pipeline

### Step 1: Data generation

```bash
python step1.py
```

Generates 10 answers per sampled question with Qwen3-4B, stores top-10
probability sums per token, labels each answer by comparing the parsed final
number with the gold answer, and writes mixed questions to `answers.jsonl`.

### Step 2: Visualization

```bash
python step2.py --question_id 19 --window 15
```

Outputs:

- `img/top10_ma_q<id>.png`: one mixed question, green = correct, red = incorrect
- `img/top10_ma_grid.png`: nine mixed questions side by side

The y-axis is zoomed because top-10 probability sums are usually very close to
1.0. Balanced mixed questions, such as 5 correct and 5 incorrect answers, show
the clearest visual separation. Highly unbalanced questions, such as 9 correct
and 1 incorrect, are much noisier.

### Step 3: ML classification

```bash
python step3.py
```

Features are extracted from the moving-average curve: summary statistics,
percentiles, trend, local drops, roughness, answer length, uncertainty area, and
within-question relative features. The split is by question to avoid leakage
between sibling answers.

Final test results:

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| XGBoost | 0.5578 | 0.5928 | 0.5623 |
| CatBoost | 0.5750 | 0.6233 | 0.6093 |
| GradientBoosting | 0.5734 | 0.6106 | 0.6151 |
| RandomForest | 0.5781 | 0.6143 | 0.6073 |
| LGBM | 0.5578 | 0.6053 | 0.5875 |

In terms of no-threshold-dependency, the best model is GradientBoosting with ROC-AUC 0.6151
