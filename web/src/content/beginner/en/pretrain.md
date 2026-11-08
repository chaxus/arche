## In one sentence
A tiny AdamW loop teaches next-token prediction on toy batches.

## Before you start
- You can run scripts; Part 1 stack idea helps. Time: ~15 minutes.

## The idea (plain language)
**Pretraining** here means: show random token ids, ask the model to predict the next id, measure cross-entropy, take an optimizer step.
We use toy sizes so you feel the API—not internet-scale training.

## In the code
- `arche/train/pretrain.py`
- `chapters/02-training/ep18_pretrain.py`

## Do this now
```bash
python chapters/02-training/ep18_pretrain.py
```
Watch loss print a short history.

## If something breaks
- Loss not decreasing on random data → try the test’s repeated-batch idea; demos may wiggle.

## Don't worry about yet
Data pipelines, distributed training, or learning-rate science.

## Next
**SFT · DPO** (`?c=sft-dpo`).
