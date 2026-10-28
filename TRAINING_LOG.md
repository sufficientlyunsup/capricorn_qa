# Training log

## Full dataset run

Command recorded; results pending.

```bash
uv run python -m capricorn.bin.train_stage1 \
  --sa-layers 2 \
  --epochs 10 \
  --patience 2 \
  --batch-size 16 \
  --max-context-tokens 4096 \
  --device cuda \
  --output checkpoints/capricorn-concat.pt
```
