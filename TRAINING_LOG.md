# Training log

CAPRICORN was trained in two stages on a workstation with an NVIDIA RTX 3090.
The backbone was kept frozen in both runs. Training therefore updates the
fusion and prediction layers while preserving the pretrained Qwen language
representations.

To recreate the training environment, first download the full Qwen backbone
and configured datasets:

```bash
uv run python -m capricorn.bin.initialise --train
```

Stage 1 trained the flexible three-way entailment model on the combined NLI
corpus: MNLI, ANLI R1–R3, and SNLI. Each input uses the same configurable
context/question representation, while the output head predicts the three NLI
labels. One epoch took approximately eight hours on the 3090. The run was
allowed up to ten epochs with early stopping and completed five epochs before
restoring the epoch 3 checkpoint with the best validation loss.

## Stage 1: NLI training

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

The NLI training set contained 1,104,934 valid rows. Rows with the invalid SNLI
label `-1` were excluded. The resulting checkpoint was used as the starting
point for stage 2.

## Stage 2: Bernoulli QA adaptation

Stage 2 adapted the NLI model to independent Bernoulli predictions over
multiple QA datasets: BoolQ, MultiRC, CommonsenseQA, ARC-Challenge, MNLI, ANLI,
and SNLI. Each multiple-choice row expands into one binary trial per option.
Each NLI row expands into three binary trials, one for each entailment label.
The resulting trials are shuffled together and trained with binary
cross-entropy.

At most 10,000 native rows were retained from each dataset before expansion.
The stage 2 optimizer was initialized from scratch; no optimizer state or
gradients were carried over from stage 1. One epoch took just over one hour on
the 3090 (about 67 minutes in the recorded run).

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

The first stage 2 epoch completed in 4,030 seconds. Its training loss was
`0.860394` and its validation loss was `0.349781`. The high initial training
loss reflects the change from stage 1's three-way softmax objective to
independent sigmoid probabilities; later batches in the first epoch settled
around `0.4`–`0.45`.

The published `capricorn-qa.pt` checkpoint is the stage 2 model used by the
web service. The training datasets and local checkpoints are excluded from the
public source checkout; the trained checkpoint is distributed separately from
Hugging Face.
