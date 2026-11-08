## Idea

Frontier MoE/MLA only helps if you can speak their **checkpoint names and shapes**—save and load with a key map, including Mixtral-like `w1/w2/w3` expert tensors. Structure on paper is useless if tensors won’t round-trip; a bad map leaves silent random experts.

Map open names into your modules, use [`save_checkpoint` / `load_checkpoint`](https://github.com/chaxus/arche/blob/main/arche/checkpoint.py), and fail loudly on missing keys.

### Round-trip

Teaching ckpts write state + meta, reload into a matching module graph, and assert alignment. Every expert key must be present—partial MoE loads can “run” while one expert is noise.

### Naming debt

| Risk | Effect |
| --- | --- |
| HF vs local name drift | Missing keys / wrong slots |
| Expert index off-by-one | One dead expert forever |
| Strict=False abuse | Hides map bugs |

### From zero

| Step | Do |
| --- | --- |
| 1 | List open hub key patterns (Mixtral / DeepSeek docs) |
| 2 | Build an explicit map into your modules |
| 3 | Round-trip a tiny synthetic ckpt first |
| 4 | Only then touch multi-GB downloads |

### Check yourself

- Round-trip `allclose`; no missing experts.
- Unexpected keys fail or are explicitly ignored.
- Hooks are ready for real hubs.

### Levers

| Lever | Impact |
| --- | --- |
| Consistent naming | Cuts converter debt |
| Bad maps | Silent quality holes |
| Converters as standard | Same story as dense HF maps |

### This course

```bash
python chapters/02-training/ep20_frontier_ckpt.py
```

## Run

```bash
python chapters/02-training/ep20_frontier_ckpt.py
```

## Next

**Sharded MoE / MLA** (`/sharded-frontier/`).
