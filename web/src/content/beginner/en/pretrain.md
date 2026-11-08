## Idea

**Pretraining** = learn to predict the next token on lots of text. Loss is usually cross-entropy between logits and the true next id; an optimizer (here **AdamW**) updates weights.

Our loop is intentionally **tiny** (toy batch, few steps): you feel the API and watch loss move—not train a real 1B on the internet. On this page, drag learning rate / steps and run a real tiny loop in the browser.

## Run

```bash
python chapters/02-training/ep18_pretrain.py
```

Expect a few steps of decreasing-ish toy loss. The page demo simulates the same idea visually.

## Next

**SFT · DPO** (`?c=sft-dpo`): follow instructions and preferences.
