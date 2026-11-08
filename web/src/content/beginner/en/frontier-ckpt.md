## Idea

**What:** Speak **checkpoint names and shapes** for MoE/MLA-style open weights—save/load with a key map, including Mixtral-like `w1/w2/w3` expert tensors.

**Why:** Frontier structure on paper is useless if you can’t round-trip tensors. Bad maps → silent random experts.

**How:** Name map ↔ open names; [`save_checkpoint` / `load_checkpoint`](https://github.com/chaxus/arche/blob/main/arche/checkpoint.py); fail loudly on missing keys.

### 1. What — round-trip

Teaching ckpts exercise: write state + meta, reload into a matching module graph, assert alignment. Expert keys must all be present—partial MoE loads look “fine” until one expert is noise.

### 2. Why — naming debt

| Risk | Effect |
| --- | --- |
| HF vs local name drift | Missing keys / wrong slots |
| Expert index off-by-one | One dead expert forever |
| Strict=False abuse | Hides map bugs |

### 3. How — from zero

| Step | Do |
| --- | --- |
| 1 | List open hub key patterns (Mixtral / DeepSeek docs) |
| 2 | Build explicit map into your modules |
| 3 | Round-trip on a tiny synthetic ckpt first |
| 4 | Only then touch multi-GB downloads |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Round-trip `allclose`; no missing experts |
| **Hard** | Unexpected keys fail or are explicitly ignored |
| **Soft** | Hooks ready for real hubs |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Consistent naming | Cuts converter debt |
| Bad maps | Silent quality holes |
| Converters as standard | Same story as dense HF maps |

### 6. This course

```bash
python chapters/02-training/ep20_frontier_ckpt.py
```

## Run

```bash
python chapters/02-training/ep20_frontier_ckpt.py
```

## Next

**Sharded MoE / MLA** (`/sharded-frontier/`).
