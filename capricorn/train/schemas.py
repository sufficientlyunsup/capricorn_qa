"""Output schemas for the benchmark task families."""

from capricorn.lib.model import OutputSchema


NLI_SCHEMA = OutputSchema(
    name="nli",
    type="choice",
    questions=(
        "Does the passage entail the statement?",
        "Is the statement neutral with respect to the passage?",
        "Does the passage contradict the statement?",
    ),
)

RACE_SCHEMA = OutputSchema(
    name="race",
    type="choice",
    questions=("option A", "option B", "option C", "option D"),
)

HELLASWAG_SCHEMA = OutputSchema(
    name="hellaswag",
    type="choice",
    questions=("ending A", "ending B", "ending C", "ending D"),
)


QA_SCHEMA = OutputSchema(name="qa", type="bernoulli", questions=("Is the answer yes?",))
