"""Serve a schema-configured CAPRICORN checkpoint over HTTP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from flask import Flask, jsonify, request
from transformers import AutoTokenizer

from capricorn.bin.common import MODEL_DIR
from capricorn.lib.model import CapricornFactory, OutputSchema
from capricorn.train.dataset import _tokenize_sequences

MAX_CONTEXT_TOKENS = 4096


def load_schema(path: Path) -> OutputSchema:
    """Load and validate an output schema from a JSON file."""
    with path.open(encoding="utf-8") as stream:
        return OutputSchema.model_validate(json.load(stream))


def create_app(
    *,
    checkpoint: Path,
    schema_path: Path,
    model_path: Path = MODEL_DIR,
    device: str | None = None,
    max_context_tokens: int = 4096,
) -> Flask:
    """Build a Flask application with one loaded model and fixed schema."""
    if not 1 <= max_context_tokens <= MAX_CONTEXT_TOKENS:
        raise ValueError(
            f"max_context_tokens must be between 1 and {MAX_CONTEXT_TOKENS}"
        )

    schema = load_schema(schema_path)
    factory = CapricornFactory(
        schema,
        model_path=model_path,
        nlayers=20,
        n_attention_layers=2,
        device=device,
    )
    model = factory.build_from_saved_file(checkpoint)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs a pad token or EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    app = Flask(__name__)

    @app.get("/health")
    def health() -> tuple[Any, int]:
        return jsonify({"status": "ok", "schema": schema.name})

    @app.post("/predict")
    def predict() -> tuple[Any, int] | Any:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "request body must be a JSON object"}), 400
        context = payload.get("context")
        if not isinstance(context, str):
            return jsonify({"error": "context must be a string"}), 400

        sequences = [
            f"<context>{context}</context><question>{question}</question>"
            for question in schema.questions
        ]
        untruncated = tokenizer(sequences, padding=False, truncation=False)
        lengths = [len(token_ids) for token_ids in untruncated["input_ids"]]
        longest = max(lengths, default=0)
        encoded = _tokenize_sequences(tokenizer, sequences, max_context_tokens)
        inputs = {
            name: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
            for name, value in encoded.items()
        }
        with torch.inference_mode():
            vector = model(inputs).squeeze(0).detach().cpu().tolist()
        response = {"vector": vector}
        if longest > max_context_tokens:
            response.update({
                "warning": (
                    "The context and question were truncated to max_context_tokens."
                ),
                "max_context_tokens": max_context_tokens,
                "longest_sequence_tokens": longest,
            })
        return jsonify(response)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=MODEL_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    app = create_app(
        checkpoint=args.checkpoint,
        schema_path=args.schema,
        model_path=args.model_path,
        device=args.device,
        max_context_tokens=args.max_context_tokens,
    )
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
