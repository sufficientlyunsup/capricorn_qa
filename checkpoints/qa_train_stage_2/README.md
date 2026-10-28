# Bernoulli QA training stage 2

This directory contains the stage 2 adaptation run initialized from the stage 1
NLI checkpoint. The Qwen backbone remains frozen. The shared fusion layer is
trained with binary cross-entropy over Bernoulli trials from BoolQ, MultiRC,
CommonsenseQA, ARC-Challenge, MNLI, ANLI, and SNLI.

Each NLI row expands into three trials: entailment, neutral, and contradiction.
Each multiple-choice QA row expands into one trial per candidate. The training
command caps each dataset at 10,000 native rows before expansion, then shuffles
all resulting trials together. No class weighting or dataset balancing is used.

## Training command

```bash
uv run python -m capricorn.bin.train_stage2 \
  --checkpoint checkpoints/nli_train_stage_1/capricorn-concat.pt \
  --max-keep-rows 10000 \
  --epochs 10 \
  --patience 2 \
  --batch-size 16 \
  --max-context-tokens 4096 \
  --device cuda \
  --output checkpoints/qa_train_stage_2/capricorn-qa.pt
```

Stage 2 starts with fresh optimizer state and no gradients from stage 1. Use an
epoch `.resume.pt` bundle to resume an interrupted stage 2 run.

The first epoch completed with these losses:

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 1 | 0.860394 | **0.349781** |

The high training loss includes the initial mismatch between stage 1’s softmax
scores and stage 2’s independent sigmoid probabilities. Later epoch 1 batches
settled around 0.4–0.45.

## Bernoulli validation baseline and stage 2 result

The original stage 1 checkpoint was also evaluated through the Bernoulli QA
interface before stage 2 training. It has no useful independent sigmoid
calibration, so its scores collapse toward the negative majority class and its
binary cross-entropy is very large.

| Dataset | Stage 1 accuracy | Stage 1 BCE | Stage 2 epoch 1 accuracy | Stage 2 epoch 1 BCE |
|---|---:|---:|---:|---:|
| BoolQ | 37.83% | 51.1272 | 75.81% | 0.5008 |
| MultiRC | 57.20% | 35.1893 | 81.68% | 0.4043 |
| CommonsenseQA | 80.00% | 17.2907 | 82.92% | 0.3752 |
| ARC | 74.96% | 21.5664 | 80.15% | 0.4566 |
| MNLI | 66.67% | 26.8047 | 86.97% | 0.3121 |
| SNLI | 66.67% | 26.1039 | 91.53% | 0.2202 |
| ANLI | 66.67% | 26.7794 | 65.76% | 0.7558 |

The binary accuracy for candidate-expanded datasets has a high negative-class
baseline because each multiple-choice row contributes one positive and several
negative trials. BCE is therefore an important part of this comparison. The
stage 2 values are validation results; the stage 1 values are the same
validation interface applied to the unadapted checkpoint.

## Stage 1 benchmark retention

The epoch 1 stage 2 checkpoint was evaluated with the stage 1 evaluator. NLI
results changed as follows:

| Benchmark | Stage 1 accuracy | Stage 2 epoch 1 | Change |
|---|---:|---:|---:|
| MNLI matched | 84.71% | 82.49% | -2.22 pp |
| MNLI mismatched | 85.30% | 84.10% | -1.20 pp |
| ANLI R1 | 60.50% | 53.70% | -6.80 pp |
| ANLI R2 | 46.30% | 43.30% | -3.00 pp |
| ANLI R3 | 45.00% | 42.67% | -2.33 pp |
| SNLI | 89.15% | 88.69% | -0.46 pp |

The larger ANLI decline is an observed result of this stage 2 run. Its cause
has not been isolated; possible factors include the capped replay sample, the
changed objective, and interference from the additional QA data.

## Transfer evaluation

The same epoch 1 checkpoint was evaluated on RACE and HellaSwag, neither of
which was included in stage 2 training:

| Benchmark | Stage 1 accuracy | Stage 2 epoch 1 | Change |
|---|---:|---:|---:|
| RACE statement subset | 49.50% | 69.22% | +19.72 pp |
| HellaSwag validation | 48.81% | 56.67% | +7.86 pp |

RACE uses the filtered 497-row statement subset, not the full RACE test set.
HellaSwag uses all 10,042 validation rows. Both are evaluated by scoring the
four candidates and selecting the highest-scoring candidate.

## Evaluation commands

Evaluate the retained NLI performance:

```bash
uv run python -m capricorn.bin.evaluate_stage1 \
  --checkpoint checkpoints/qa_train_stage_2/capricorn-qa-epoch-001.pt \
  --evaluation-type entailment \
  --batch-size 16 \
  --max-context-tokens 4096 \
  --device cuda
```

Evaluate RACE:

```bash
uv run python -m capricorn.bin.evaluate_stage1 \
  --checkpoint checkpoints/qa_train_stage_2/capricorn-qa-epoch-001.pt \
  --evaluation-type multiple_choice \
  --batch-size 16 \
  --max-context-tokens 4096 \
  --device cuda
```

Evaluate HellaSwag:

```bash
uv run python -m capricorn.bin.evaluate_stage1 \
  --checkpoint checkpoints/qa_train_stage_2/capricorn-qa-epoch-001.pt \
  --evaluation-type hellaswag \
  --batch-size 16 \
  --max-context-tokens 4096 \
  --device cuda
```

Evaluate the Bernoulli validation splits:

```bash
uv run python -m capricorn.bin.evaluate_stage2 \
  --checkpoint checkpoints/qa_train_stage_2/capricorn-qa-epoch-001.pt \
  --batch-size 16 \
  --max-context-tokens 4096 \
  --device cuda
```
