# NLI training stage 1

This directory contains the completed flexible-schema NLI training run. The backbone was frozen and the fusion layer was trained with two self-attention layers over concatenated context/question inputs.

## Training command

```bash
uv run python -m capricorn.bin.train_stage1 \
  --sa-layers 2 \
  --epochs 10 \
  --patience 2 \
  --batch-size 16 \
  --max-context-tokens 4096 \
  --device cuda \
  --output checkpoints/nli_train_stage_1/capricorn-concat.pt
```

The run stopped early after epoch 5 and restored epoch 3, which had the best validation loss.

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 1 | 0.432633 | 0.494041 |
| 2 | 0.395134 | 0.464903 |
| 3 | 0.381298 | **0.459744** |
| 4 | 0.372384 | 0.461031 |
| 5 | 0.364059 | 0.471657 |

The training set contains 1,104,934 valid rows from MNLI, ANLI R1–R3, and SNLI. The validation set contains 22,857 valid rows. SNLI rows labelled `-1` were excluded.

`capricorn-concat.pt` is the restored best model from epoch 3. The epoch checkpoints and restart bundles are retained for comparison and resumption.

## Evaluation results

The final model was evaluated with the local evaluator. MNLI uses its validation
splits; ANLI and SNLI use their test splits. RACE is the 497-row filtered
statement subset, and HellaSwag uses its validation split.

| Benchmark | Accuracy | Cross-entropy | Examples | Random guess |
|---|---:|---:|---:|---:|
| MNLI matched validation | 84.71% | 0.3878 | 9,815 | 33.33% |
| MNLI mismatched validation | 85.30% | 0.3773 | 9,832 | 33.33% |
| ANLI R1 test | 60.50% | 1.0010 | 1,000 | 33.33% |
| ANLI R2 test | 46.30% | 1.2938 | 1,000 | 33.33% |
| ANLI R3 test | 45.00% | 1.3539 | 1,200 | 33.33% |
| SNLI test | 89.15% | 0.2845 | 9,824 | 33.33% |
| RACE statement subset | 49.50% | 1.1448 | 497 | 25.00% |
| HellaSwag validation | 48.81% | 1.2576 | 10,042 | 25.00% |

RACE uses the NLI-compatible template:

```text
Does the passage entail that the answer to "{question}" is "{option}"?
```

## Evaluation commands

Evaluate all entailment benchmarks:

```bash
uv run python -m capricorn.bin.evaluate_stage1 \
  --checkpoint checkpoints/nli_train_stage_1/capricorn-concat.pt \
  --evaluation-type entailment \
  --batch-size 16 \
  --max-context-tokens 4096 \
  --device cuda
```

Evaluate the filtered RACE statement subset:

```bash
uv run python -m capricorn.bin.evaluate_stage1 \
  --checkpoint checkpoints/nli_train_stage_1/capricorn-concat.pt \
  --evaluation-type multiple_choice \
  --batch-size 16 \
  --max-context-tokens 4096 \
  --device cuda
```

Evaluate HellaSwag validation:

```bash
uv run python -m capricorn.bin.evaluate_stage1 \
  --checkpoint checkpoints/nli_train_stage_1/capricorn-concat.pt \
  --evaluation-type hellaswag \
  --batch-size 16 \
  --max-context-tokens 4096 \
  --device cuda
```
