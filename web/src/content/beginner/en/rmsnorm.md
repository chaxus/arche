## Idea

**RMSNorm** scales by root-mean-square (no mean subtract). Used **pre-norm**: before attention and before MLP.

$$\mathrm{RMSNorm}(x)=x/\sqrt{\mathrm{mean}(x^2)+\varepsilon}\odot\gamma$$

## Run

```bash
python chapters/01-inference/ep04_rmsnorm.py
```

## Next

**RoPE** (`?c=rope`).
