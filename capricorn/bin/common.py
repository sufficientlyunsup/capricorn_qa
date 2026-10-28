"""Shared model and dataset settings for CAPRICORN command-line scripts."""

from pathlib import Path


HUGGINGFACE_REPO_ID = "Qwen/Qwen3-1.7B-Base"
MODEL_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
# Relative to the working directory; run scripts from the project root.
MODEL_DIR = Path("models/qwen3-1.7b")
DEFAULT_QWEN3_LAYERS = 20

# Dataset repositories are downloaded as their native Hugging Face files.
# The MNLI and MultiRC repositories contain other datasets, so restrict those
# downloads to the relevant subtrees.
DATASETS = (
    ("MNLI", "nyu-mll/glue", "data/mnli", ("mnli/**",)),
    ("ANLI", "facebook/anli", "data/anli", None),
    ("SNLI", "stanfordnlp/snli", "data/snli", None),
    ("RACE", "ehovy/race", "data/race", None),
    ("HellaSwag", "Rowan/hellaswag", "data/hellaswag", None),
    # QA datasets used for candidate-level Bernoulli training.
    ("BoolQ", "google/boolq", "data/boolq", None),
    ("MultiRC", "aps/super_glue", "data/multirc", ("multirc/**",)),
    ("CommonsenseQA", "tau/commonsense_qa", "data/commonsense_qa", None),
    ("ARC", "allenai/ai2_arc", "data/arc", None),
)
