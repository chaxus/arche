## In one sentence
SFT supervises only the answer span; DPO nudges preferred replies over rejected ones.

## Before you start
- Pretrain chapter. Time: ~18 minutes.

## The idea (plain language)
**SFT** masks the prompt tokens so loss focuses on the response.
**DPO** compares chosen vs rejected sequences against a frozen reference model (Rafailov et al.)—one educational step is enough to see the API.

## In the code
- `arche/train/sft.py`, `arche/train/dpo.py`
- `chapters/02-training/ep19_sft_dpo.py`

## Do this now
```bash
python chapters/02-training/ep19_sft_dpo.py
```

## If something breaks
- Prompt lengths longer than sequences → keep demo tensors.

## Don't worry about yet
Full RLHF stacks or preference data collection.

## Next
**Frontier checkpoints** (`?c=frontier-ckpt`).
