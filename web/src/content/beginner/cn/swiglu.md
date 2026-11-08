## 要点

**是什么（What）：** **SwiGLU**——门控前馈：`down(silu(gate(x)) * up(x))`，三张**无 bias** 线性层。

**为什么（Why）：** 注意力在位置*之间*混合；FFN 在*每个*位置上变换。SwiGLU（而非纯 GELU MLP）是 Llama/Qwen 稠密默认；MoE 的 expert 常常就是同一块。

**怎么做（How）：** 选定 `intermediate_dim`，三张矩阵，接在 block 里第二次 pre-norm 之后。

### 1. What — 代码

[`arche/mlp.py`](https://github.com/chaxus/arche/blob/main/arche/mlp.py)：

```python
class SwiGLU(nn.Module):
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

gate / up 把 `D → I`；down 回到 `I → D`。无 bias——对齐 Llama 权重文件。

### 2. Why — 门控 vs 普通 MLP

| 选择 | 原因 |
| --- | --- |
| SiLU(gate)×up | 软门控；LM 上经验强 |
| 三矩阵 | 同 `I` 下比两矩阵 GELU MLP 参数略多 |
| 无 bias | 更省，对齐开源 Llama/Qwen 张量 |
| 按位置 | 易并行；配残差 |

### 3. How — 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | 选 `I`（Llama 3.2 1B 由 config 给出） |
| 2 | `gate_proj` / `up_proj` / `down_proj` 均 bias=False |
| 3 | 放在 `post_attention_layernorm` 后 |
| 4 | 以后 MoE：许多 SwiGLU + router |

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 进出维 `D`；无 bias；gate 上 SiLU |
| **软** | 用整模 loss / 质量评判，不孤立看 FFN |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 更宽 `I` | 容量与算力 |
| GELU → SwiGLU | 需重训 |
| MoE experts | 同一砖 × E，稀疏点火 |

### 6. 本课

```bash
python chapters/01-inference/ep07_swiglu.py
```

## 动手

```bash
python chapters/01-inference/ep07_swiglu.py
```

## 下一章

**Transformer block**（`/cn/block/`）：把零件夹成一层。
