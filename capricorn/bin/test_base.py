"""Inspect layer names, parameter shapes, and parameter allocation on CUDA."""

import argparse
import logging

from capricorn.bin.common import DEFAULT_QWEN3_LAYERS, MODEL_DIR
from capricorn.lib.model import CapricornModel, FusionLayer, OutputSchema
from capricorn.lib.model.construct import make_output_layer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    model_path = MODEL_DIR.resolve()
    if not model_path.is_dir():
        parser.error("Model not found. Run: uv run python -m capricorn.bin.initialise")

    # Defer heavy imports so CLI help and argument validation remain lightweight.
    import torch
    from transformers import AutoModelForCausalLM

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; check your NVIDIA driver and PyTorch installation.")
    logging.info("Loading %s on %s (float16)", model_path, torch.cuda.get_device_name(0))
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="cuda",
        local_files_only=True,
    )
    model.requires_grad_(False)
    model.eval()

    layers = model.model.layers
    print(f"Transformer layers: {len(layers)}")
    for i, layer in enumerate(layers):
        params = sum(p.numel() for p in layer.parameters())
        print(f"\nmodel.layers.{i} ({type(layer).__name__}): {params:,} parameters")
        for name, module in layer.named_children():
            count = sum(p.numel() for p in module.parameters())
            share = 100 * count / params if params else 0
            print(f"  {name} ({type(module).__name__}): {count:,} ({share:.2f}% of block)")
        print("  Parameter tensors (shape uses PyTorch dimension order):")
        for name, parameter in layer.named_parameters():
            count = parameter.numel()
            share = 100 * count / params if params else 0
            print(
                f"    model.layers.{i}.{name}: shape={tuple(parameter.shape)} | "
                f"{count:,} parameters | {share:.2f}% of block | {parameter.dtype}"
            )

    print("\nParameters outside transformer blocks (e.g. embeddings, final norm, LM head):")
    block_parameter_ids = {id(p) for layer in layers for p in layer.parameters()}
    for name, parameter in model.named_parameters():
        if id(parameter) not in block_parameter_ids:
            print(f"  {name}: shape={tuple(parameter.shape)} | {parameter.numel():,} parameters")
    print("Shared parameters are listed under their first name and counted once.")

    print(f"Parameters (including LM head; shared weights counted once): {sum(p.numel() for p in model.parameters()):,}")

    # Assemble a three-way choice CAPRICORN model so its parameter allocation
    # can be compared with the Qwen backbone. This is an inspection-only
    # construction; the tokenizer/input adapter is not run by this command.
    backbone = model.model
    selected_layers = list(backbone.layers)[:DEFAULT_QWEN3_LAYERS]
    backbone.layers = torch.nn.ModuleList(selected_layers)
    backbone.config.num_hidden_layers = DEFAULT_QWEN3_LAYERS
    logging.info(
        "Using the first %d transformer layers for the CAPRICORN representation",
        DEFAULT_QWEN3_LAYERS,
    )
    schema = OutputSchema.model_validate_json(
        '{"name":"three_way_choice","type":"choice",'
        '"questions":["entailment","neutral","contradiction"]}'
    )
    fusion = FusionLayer(
        hidden_size=backbone.config.hidden_size,
        num_heads=backbone.config.num_attention_heads,
        intermediate_size=backbone.config.intermediate_size,
        n_attention_layers=2,
    ).to(dtype=torch.float16)
    capricorn = CapricornModel(
        backbone=backbone,
        fusion_layer=fusion,
        output_layer=make_output_layer(schema),
    )
    print("\nCAPRICORN three-way choice model parameters:")
    for name, module in capricorn.named_children():
        count = sum(parameter.numel() for parameter in module.parameters())
        print(f"  {name}: {count:,} parameters")
    total_capricorn = sum(parameter.numel() for parameter in capricorn.parameters())
    print(f"  total: {total_capricorn:,} parameters")

    fusion = capricorn.logits_model.fusion_layer
    fusion_total = sum(parameter.numel() for parameter in fusion.parameters())
    print(f"\nCAPRICORN fusion parameters: {fusion_total:,}")
    for group_name, group in fusion.named_children():
        group_count = sum(parameter.numel() for parameter in group.parameters())
        group_share = 100 * group_count / fusion_total if fusion_total else 0
        print(
            f"\n  {group_name} ({type(group).__name__}): "
            f"{group_count:,} ({group_share:.2f}% of fusion)"
        )
        children = list(group.named_children())
        if children:
            for child_name, child in children:
                child_count = sum(parameter.numel() for parameter in child.parameters())
                child_share = 100 * child_count / fusion_total if fusion_total else 0
                print(
                    f"    {group_name}.{child_name} ({type(child).__name__}): "
                    f"{child_count:,} ({child_share:.2f}% of fusion)"
                )
                for parameter_name, parameter in child.named_parameters():
                    count = parameter.numel()
                    share = 100 * count / fusion_total if fusion_total else 0
                    print(
                        f"      {group_name}.{child_name}.{parameter_name}: "
                        f"shape={tuple(parameter.shape)} | {count:,} parameters | "
                        f"{share:.2f}% of fusion | {parameter.dtype}"
                    )
        else:
            for parameter_name, parameter in group.named_parameters():
                count = parameter.numel()
                share = 100 * count / fusion_total if fusion_total else 0
                print(
                    f"    {group_name}.{parameter_name}: shape={tuple(parameter.shape)} | "
                    f"{count:,} parameters | {share:.2f}% of fusion | {parameter.dtype}"
                )


if __name__ == "__main__":
    main()
