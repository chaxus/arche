## Idea

A dense MLP runs the **same** feed-forward for every token. **Mixture of Experts (MoE)** keeps many expert MLPs (here: SwiGLU experts) and a **router** that, per token, picks only the **top-k** experts to run.

You can grow total parameters a lot while each token pays for only a few experts—**sparse compute** (Mixtral-style idea). This chapter is a teaching MoE block, not a full production Mixtral stack.

## Run

```bash
python chapters/01-inference/ep14_moe.py
```

Expect router choices / shapes for a tiny MoE forward.

## Next

**MLA** (`?c=mla`): compress what you cache for decode.
