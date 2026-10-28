# CAPRICORN

Configurable Architecture for Probabilistic Inference and Contextual Reasoning over Natural language

## Project overview

CAPRICORN aims to develop a lightweight, general-purpose probabilistic inference architecture capable of multiclass classification and repeated Bernoulli inference over textually defined contexts and candidate outcomes.

Through independently encoded, reusable representations and late-stage attention, the architecture seeks to minimise redundant computation while supporting dynamically specified classification tasks without task-specific fine-tuning.

The project targets a memory footprint of approximately 2 GB RAM, enabling efficient, accessible deployment on consumer hardware and resource-constrained systems.

## Getting started

This project uses Python 3.12+ and [uv](https://docs.astral.sh/uv/) for dependency and environment management.

```bash
uv sync
uv run main.py
```

Add dependencies with `uv add <package>` and development dependencies with `uv add --dev <package>`.

## Download the trained model

From the project root, run:

```bash
uv run python -m capricorn.bin.initialise
```

This downloads the published CAPRICORN checkpoint from
`sufficientlyunsup/capricorn-qa` into
`checkpoints/qa_train_stage_2/capricorn-qa.pt`, together with the small Qwen
configuration and tokenizer files needed to run inference. The full Qwen
backbone weights are already included in the CAPRICORN checkpoint. If the
files already exist, the downloads are skipped without accessing the network.

To download again, or recover from an incomplete download, run:

```bash
uv run python -m capricorn.bin.initialise --force
```

## Download training assets

To download the Qwen backbone and all configured NLI and QA datasets, run:

```bash
uv run python -m capricorn.bin.initialise --train
```

This downloads Qwen/Qwen3-1.7B-Base into `models/qwen3-1.7b` and the datasets
into `data/` as native Hugging Face repository files. MNLI and MultiRC are
restricted to their relevant subtrees of larger repositories. Add `--force` to
redownload these assets.

View all options with:

```bash
uv run python -m capricorn.bin.initialise -h
```

## Inspect the base model

After downloading the model, load it on CUDA in float16 and print its transformer layer count and total parameter count:

```bash
uv run python -m capricorn.bin.test_base
```

The report includes each transformer's full block name and class, parameter totals by component (attention, MLP, and normalisation), and each parameter tensor's name, shape, dtype, count, and percentage of its block. It also lists parameters outside the transformer blocks. Shapes follow PyTorch dimension order; linear weights are `(out_features, in_features)`.

The total parameter count includes the language modelling head, with shared weights counted once. Use `-h` for help.

## Stage 1: NLI

The NLI entry points are now `capricorn.bin.train_stage1` and
`capricorn.bin.evaluate_stage1`. Their training and evaluation options are unchanged.

## Stage 2: Bernoulli QA

Continue from the stage 1 checkpoint with BoolQ, MultiRC, CommonsenseQA, and
ARC-Challenge, plus MNLI, ANLI, and SNLI replay. NLI rows expand into three
Bernoulli trials (entailment, neutral, contradiction) using the fixed stage 1
question wording; unlabelled rows are skipped. ANLI’s row budget is shared
across rounds 1–3. Validation uses MNLI matched, ANLI dev rounds, and SNLI
validation. The backbone stays frozen; the fusion layer and scalar head are
trained with binary cross-entropy. Each dataset contributes its first at most
10,000 native training rows, expanded into individual Bernoulli trials before
shuffling and batching. Consequently, multiple-choice sources contribute more
trials per source row. No class weighting or balancing is applied.

```bash
uv run python -m capricorn.bin.train_stage2 \
    --checkpoint checkpoints/nli_train_stage_1/capricorn-concat.pt \
    --max-keep-rows 10000 --epochs 10 --patience 2 \
    --batch-size 16 --max-context-tokens 4096 --device cuda \
    --output checkpoints/qa_train_stage_2/capricorn-qa.pt
```

Early stopping uses the full labelled validation splits. Epoch weights and
`.resume.pt` bundles are saved alongside the final best checkpoint. Resume with
`--resume path/to/epoch.resume.pt --epochs 10`.

```bash
uv run python -m capricorn.bin.evaluate_stage2 \
    --checkpoint checkpoints/qa_train_stage_2/capricorn-qa.pt \
    --batch-size 16 --max-context-tokens 4096 --device cuda
```

Evaluation reports per-dataset binary accuracy at probability 0.5, binary
cross-entropy, majority-class baseline, and trial count. These are validation
results on the same splits used for early stopping, not independent test results
or multiple-choice question accuracy. Use `--max-keep-rows` to shorten evaluation.

## Using the CAPRICORN model as a web service

The web service loads the model and an `OutputSchema` JSON file once at startup.
Each request supplies one article or passage. CAPRICORN combines that context
with every question in the schema and returns one probability per question.
Example schemas are in `example_schemas/`.

Start the service with the article-category schema:

```bash
uv run python -m capricorn.bin.deploy \
  --checkpoint checkpoints/qa_train_stage_2/capricorn-qa.pt \
  --schema example_schemas/article_category.json \
  --device cuda
```

The server listens on `http://127.0.0.1:5000` and stays running in the
terminal. Leave it running while sending requests from another terminal. Use
`Ctrl+C` to stop it. To try the football-news schema instead, change the schema
argument to `example_schemas/football_news.json`.

Send a JSON object containing a `context` string to `/predict`:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H 'Content-Type: application/json' \
  -d "{\"context\":\"Manchester United confirmed that their captain will miss Sunday's Premier League match because of a hamstring injury. The manager said the team is assessing several players before the fixture.\"}"
```

The response contains a `vector` whose positions match the questions in the
schema. For `article_category.json`, the four positions are politics, sport,
business, and entertainment:

```json
{
  "vector": [0.0505, 0.6084, 0.0783, 0.2628]
}
```

`type: "choice"` schemas return probabilities that sum to one. Use them when
exactly one category should be selected. `type: "bernoulli"` schemas return
independent probabilities, so several questions can be true at once. For
example, the football-news schema can assign high probabilities to both
“football” and “team news”.

Requests longer than `--max-context-tokens` are truncated and include a warning
alongside the prediction. The deployment limit must be between 1 and 4096
tokens. A response for a truncated request has this additional shape:

```json
{
  "vector": [0.62, 0.21, 0.04, 0.13],
  "warning": "The context and question were truncated to max_context_tokens.",
  "max_context_tokens": 4096,
  "longest_sequence_tokens": 4382
}
```

The following manual smoke tests use the same three article requests with both
example schemas. These are qualitative checks, not benchmark measurements.

`football_news.json`:

| Article | Football | Team news | Man Utd news |
|---|---:|---:|---:|
| Manchester United confirmed that their captain will miss Sunday's Premier League match because of a hamstring injury. The manager said the team is assessing several players before the fixture. | 0.7028 | 0.6316 | 0.5881 |
| The Chancellor announced a new tax package in Parliament, promising changes to income tax thresholds and public spending. Opposition leaders criticised the proposal during a heated debate. | 0.0155 | 0.4247 | 0.0340 |
| The company reported record quarterly revenue after strong sales in its cloud computing division. Executives said they expect continued growth despite rising production costs. | 0.0239 | 0.4042 | 0.0470 |

`article_category.json`:

| Article | Politics | Sport | Business | Entertainment |
|---|---:|---:|---:|---:|
| Manchester United confirmed that their captain will miss Sunday's Premier League match because of a hamstring injury. The manager said the team is assessing several players before the fixture. | 0.0505 | 0.6084 | 0.0783 | 0.2628 |
| The Chancellor announced a new tax package in Parliament, promising changes to income tax thresholds and public spending. Opposition leaders criticised the proposal during a heated debate. | 0.9125 | 0.0020 | 0.0722 | 0.0133 |
| The company reported record quarterly revenue after strong sales in its cloud computing division. Executives said they expect continued growth despite rising production costs. | 0.0077 | 0.0059 | 0.9640 | 0.0224 |
