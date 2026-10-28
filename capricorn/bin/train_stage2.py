"""Continue a stage 1 checkpoint on QA and NLI datasets using Bernoulli loss."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from time import monotonic

import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from capricorn.bin.common import MODEL_DIR
from capricorn.lib.model import CapricornFactory
from capricorn.train.schemas import QA_SCHEMA
from capricorn.train.renormalise import blended_split
from capricorn.train.dataset import make_collate_qa

logger = logging.getLogger(__name__)

def save_atomic(value, path: Path) -> None:
    """Publish a completed checkpoint without exposing a partial file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train(args: argparse.Namespace) -> None:
    resume = None
    if args.resume is not None:
        resume = torch.load(args.resume, map_location="cpu", weights_only=True)
        for name, value in resume["config"].items():
            setattr(args, name, value)
        args.sa_layers = resume["config"].get("sa_layers", 2)
        args.output = args.resume.parent / resume["output_name"]
        if args.epochs <= resume["epoch"]:
            raise ValueError("--epochs must exceed the completed epoch in the resume bundle")
    torch.manual_seed(args.seed)

    dataset = blended_split(args.data_dir, "train", args.max_keep_rows)
    validation_dataset = blended_split(args.data_dir, "validation", args.validation_head)
    logger.info("training trials=%d validation trials=%d", len(dataset), len(validation_dataset))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs a pad token or EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_qa(
            tokenizer, args.max_context_tokens, template_seed=args.seed
        ),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate_qa(tokenizer, args.max_context_tokens),
    )

    factory = CapricornFactory(
        QA_SCHEMA,
        model_path=args.model_path,
        nlayers=args.nlayers,
        n_attention_layers=args.sa_layers,
        device=args.device,
    )
    model = factory.build_trainable_capricorn()
    if resume is None:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.logits_model.fusion_layer.float()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("model has no trainable parameters")

    optimizer = AdamW(trainable, lr=args.learning_rate)
    loss_fn = BCEWithLogitsLoss()
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stable_epochs = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    if resume is not None:
        model.load_state_dict(torch.load(
            args.resume.parent / resume["model_file"], map_location="cpu", weights_only=True,
        ))
        optimizer.load_state_dict(resume["optimizer"])
        best_validation_loss = resume["best_validation_loss"]
        best_epoch = resume["best_epoch"]
        stable_epochs = resume["stable_epochs"]
        if best_epoch:
            best_state = torch.load(
                args.resume.parent / resume["best_model_file"],
                map_location="cpu", weights_only=True,
            )
        torch.set_rng_state(resume["torch_rng"])
        if torch.cuda.is_available() and resume["cuda_rng"]:
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
        random.setstate(resume["python_rng"])
        loader.collate_fn.rng.setstate(resume["template_rng"])
        start_epoch = resume["epoch"] + 1
        logger.info("resuming at epoch %d with restored optimizer and random states", start_epoch)

    for epoch in range(start_epoch, args.epochs + 1):
        if stable_epochs >= args.patience:
            logger.info("saved run already reached early stopping")
            break
        model.train()
        total_loss = 0.0
        total_examples = 0
        started = monotonic()

        for step, (inputs, response) in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            logits = model.logits_model(inputs)
            targets = response.to(device=logits.device)
            loss = loss_fn(logits.float(), targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            batch_size = response.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            if step == 1 or step % args.log_every == 0 or step == len(loader):
                print(
                    f"\rINFO epoch={epoch} train batch={step}/{len(loader)} "
                    f"({100 * step / len(loader):.1f}%) "
                    f"loss={total_loss / total_examples:.6f} "
                    f"elapsed={monotonic() - started:.1f}s",
                    end="\n" if step == len(loader) else "",
                    file=sys.stderr, flush=True,
                )

        epoch_loss = total_loss / total_examples
        model.eval()
        validation_total = 0.0
        validation_examples = 0
        started = monotonic()
        with torch.no_grad():
            for step, (inputs, response) in enumerate(validation_loader, start=1):
                logits = model.logits_model(inputs)
                targets = response.to(device=logits.device)
                loss = loss_fn(logits.float(), targets)
                batch_size = response.shape[0]
                validation_total += loss.item() * batch_size
                validation_examples += batch_size
                if step == 1 or step % args.log_every == 0 or step == len(validation_loader):
                    print(
                        f"\rINFO epoch={epoch} validation batch={step}/{len(validation_loader)} "
                        f"({100 * step / len(validation_loader):.1f}%) "
                        f"loss={validation_total / validation_examples:.6f} "
                        f"elapsed={monotonic() - started:.1f}s",
                        end="\n" if step == len(validation_loader) else "",
                        file=sys.stderr, flush=True,
                    )
        validation_loss = validation_total / validation_examples
        logger.info(
            "epoch=%d train_loss=%.6f validation_loss=%.6f",
            epoch,
            epoch_loss,
            validation_loss,
        )

        epoch_path = output_path.with_name(
            f"{output_path.stem}-epoch-{epoch:03d}{output_path.suffix}"
        )
        save_atomic(model.state_dict(), epoch_path)
        logger.info("saved epoch %d checkpoint to %s", epoch, epoch_path)

        if validation_loss < best_validation_loss - args.min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            stable_epochs = 0
        else:
            stable_epochs += 1

        restart_path = epoch_path.with_suffix(".resume.pt")
        save_atomic({
            "epoch": epoch,
            "model_file": epoch_path.name,
            "output_name": output_path.name,
            "optimizer": optimizer.state_dict(),
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "best_model_file": f"{output_path.stem}-epoch-{best_epoch:03d}{output_path.suffix}",
            "stable_epochs": stable_epochs,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python_rng": random.getstate(),
            "template_rng": loader.collate_fn.rng.getstate(),
            "config": {
                name: str(value) if isinstance(value, Path) else value
                for name in (
                    "model_path", "nlayers", "sa_layers", "max_keep_rows", "data_dir", "validation_head", "batch_size",
                    "learning_rate", "max_context_tokens", "patience", "min_delta", "seed",
                )
                for value in [getattr(args, name)]
            },
        }, restart_path)
        logger.info("saved restart bundle to %s", restart_path)
        if stable_epochs >= args.patience:
            logger.info("early stopping after epoch %d", epoch)
            break

    if best_state is None:
        raise RuntimeError("validation did not produce a checkpoint")
    model.load_state_dict(best_state)
    logger.info(
        "restored best validation checkpoint from epoch %d (loss=%.6f)",
        best_epoch,
        best_validation_loss,
    )

    torch.save(model.state_dict(), output_path)
    logger.info("saved model state to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=MODEL_DIR)
    parser.add_argument("--resume", type=Path,
                        help="Resume from an epoch .resume.pt bundle; restores saved training settings.")
    parser.add_argument("--device", default=None, help="Device, for example cuda or cpu.")
    parser.add_argument("--nlayers", type=int, default=20)
    parser.add_argument("--sa-layers", type=int, default=2, help="Number of added self-attention blocks.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/nli_train_stage_1/capricorn-concat.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-keep-rows", type=int, default=10000,
                        help="Native training rows per dataset, at most 10000, before expansion.")
    parser.add_argument(
        "--validation-head",
        type=int,
        default=None,
        help="Rows per validation source for a small run.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20, help="Total epoch ceiling, including resumed epochs.")
    parser.add_argument(
        "--log-every", type=int, default=10,
        help="Log progress every N batches, plus the first and last (default: 10).",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/qa_train_stage_2/capricorn-qa.pt"))
    args = parser.parse_args()
    if args.max_context_tokens < 1 or args.patience < 1 or args.learning_rate <= 0:
        parser.error("Token limit, patience, and learning rate must be positive")
    if args.log_every < 1:
        parser.error("--log-every must be positive")

    if (not 1 <= args.max_keep_rows <= 10000) or (
        args.validation_head is not None and args.validation_head < 1
    ):
        parser.error("--max-keep-rows must be 1..10000; --validation-head must be positive")
    if args.sa_layers < 1:
        parser.error("sa-layers must be positive")
    if args.batch_size < 1 or args.epochs < 1 or args.nlayers < 1:
        parser.error("batch size, epochs, and nlayers must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train(args)


if __name__ == "__main__":
    main()
