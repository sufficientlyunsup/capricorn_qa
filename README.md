# CAPRICORN

Configurable Architecture for Probabilistic Inference and Contextual Reasoning over Natural language

## Project overview

CAPRICORN aims to develop a lightweight, general-purpose probabilistic inference architecture capable of multiclass classification and repeated Bernoulli inference over textually defined contexts and candidate outcomes.

Through independently encoded, reusable representations and late-stage attention, the architecture seeks to minimise redundant computation while supporting dynamically specified classification tasks without task-specific fine-tuning.

The model uses the first 20 layers of Qwen3-1.7B-Base, followed by an output head consisting of two bidirectional transformer blocks and a scalar prediction layer. In my local inference tests with the default context-length setting, it used just over 3 GB of VRAM.

**Warning:** I developed this project over the course of a couple of days as a
personal project. I have tried to make the implementation and results as
reproducible and correct as possible, but please approach them with healthy
scepticism.


## Quickstart

Install the dependencies from the project root:

```bash
uv sync
```

Download and initialise the published CAPRICORN model:

```bash
uv run python -m capricorn.bin.initialise
```

This downloads the published CAPRICORN checkpoint from
`sufficientlyunsup/capricorn-qa` into
`checkpoints/qa_train_stage_2/capricorn-qa.pt`, together with the small Qwen
configuration and tokenizer files needed to run inference. The full Qwen
backbone weights are already included in the CAPRICORN checkpoint. If the
files already exist, the downloads are skipped without accessing the network.

Start the web service with the example article-category schema:

```bash
uv run python -m capricorn.bin.deploy \
  --checkpoint checkpoints/qa_train_stage_2/capricorn-qa.pt \
  --schema example_schemas/article_category.json \
  --device cuda
```

Leave the server running, then send a request from another terminal:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{"context":"The Chancellor announced a new tax package in Parliament, promising changes to income tax thresholds and public spending. Opposition leaders criticised the proposal during a heated debate."}'
```


The response contains a `vector` whose positions match the questions in the
schema. For `article_category.json`, the four positions are politics, sport,
business, and entertainment:

```json
{
    "vector": [
        0.9116727113723755,
        0.001994718099012971,
        0.0728483572602272,
        0.013484260998666286
    ]
}
```

`type: "choice"` schemas return probabilities that sum to one. Use them when
exactly one category should be selected. `type: "bernoulli"` schemas return
independent probabilities, so several questions can be true at once. For
example, the football-news schema can assign high probabilities to both
“football” and “team news”.

### Checking equivariance

To check that output probabilities follow the schema rather than fixed output
positions, start the server with `example_schemas/article_permuted.json`. It
contains the same article-category questions in a different order, so the
response should contain the same probabilities in that permuted order. For
example, a politics probability that appears first with
`article_category.json` should appear in its third position with the permuted
schema.

### Other gotchas
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

I have run the following mini-tests with the following articles and both the example schema.
These are qualitative checks, not benchmark measurements.

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
