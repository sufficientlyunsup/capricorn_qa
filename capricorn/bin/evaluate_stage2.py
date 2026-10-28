"""Evaluate Bernoulli predictions on the labelled QA and NLI validation splits."""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from capricorn.bin.common import MODEL_DIR
from capricorn.bin.evaluate_stage1 import latest_checkpoint
from capricorn.train.schemas import QA_SCHEMA
from capricorn.train.renormalise import load_splits
from capricorn.train.dataset import make_collate_qa
from capricorn.lib.model import CapricornFactory


def evaluate_split(model, dataset, tokenizer, batch_size, max_context_tokens):
    loader = DataLoader(dataset, batch_size=batch_size,
                        collate_fn=make_collate_qa(tokenizer, max_context_tokens))
    total = correct = positives = 0
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, response in loader:
            logits = model.logits_model(inputs).float()
            targets = response.to(logits.device)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite evaluation logits")
            total_loss += torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="sum").item()
            correct += ((logits >= 0) == targets.bool()).sum().item()
            positives += targets.sum().item()
            total += targets.numel()
    if not total:
        raise ValueError("Evaluation dataset is empty")
    return correct / total, total_loss / total, total, max(positives, total - positives) / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/qa_train_stage_2/capricorn-qa.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-path", type=Path, default=MODEL_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--nlayers", type=int, default=20)
    parser.add_argument("--sa-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-keep-rows", type=int, default=None,
                        help="Limit native validation rows per dataset; default uses all.")
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    args = parser.parse_args()
    if min(args.nlayers, args.sa_layers, args.batch_size, args.max_context_tokens) < 1:
        parser.error("Layer counts, batch size, and token limit must be positive")
    if args.max_keep_rows is not None and args.max_keep_rows < 1:
        parser.error("--max-keep-rows must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    datasets = load_splits(args.data_dir, "validation", args.max_keep_rows)
    checkpoint = latest_checkpoint(args.checkpoint)
    logging.info("loading checkpoint %s", checkpoint)
    model = CapricornFactory(QA_SCHEMA, model_path=args.model_path, nlayers=args.nlayers,
                             n_attention_layers=args.sa_layers, device=args.device
                             ).build_from_saved_file(checkpoint)
    model.logits_model.fusion_layer.float()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs a pad token or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    for name, dataset in datasets.items():
        accuracy, loss, total, majority = evaluate_split(
            model, dataset, tokenizer, args.batch_size, args.max_context_tokens)
        print(f"{name} validation: binary_accuracy={accuracy:.4f} loss={loss:.4f} "
              f"majority_baseline={majority:.4f} trials={total}")


if __name__ == "__main__":
    main()
