## Idea

After pretrain, models still need to **follow chats** and **prefer better answers**.

- **SFT** (supervised fine-tune): train on prompt→response examples, but **mask the prompt** so loss only hits the reply tokens.  
- **DPO** (Direct Preference Optimization): given chosen vs rejected replies, push the model toward chosen **relative to a frozen reference**—no separate reward model in the basic form. We run **one teaching step**.

## Run

```bash
python chapters/02-training/ep19_sft_dpo.py
```

Expect SFT masking behavior and a single DPO update printout.

## Next

**Frontier checkpoints** (`?c=frontier-ckpt`): save/load MoE & MLA shaped weights.
