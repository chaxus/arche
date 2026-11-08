## 要点

从零搭块：注意力混合位置；**FFN** 在每个位置做非线性——现代 dense 默认 SwiGLU。

### 1. 有什么作用

$$\\mathrm{FFN}(x)=W_d(\\mathrm{SiLU}(W_g x)\\odot W_u x)$$

三条无 bias 线性：gate / up / down。

### 2. 从零开始：怎么生成

| 步骤 | 做什么 |
| --- | --- |
| **1** | 定 `ffn_dim`（通常 > `D`） |
| **2** | 建三套权重，无 bias |
| **3** | 接在块内第二段（pre-norm 后） |
| **4** | 整网训；MoE 时每个专家往往是一个 SwiGLU |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 进出 `D`；无 bias（Llama 契约）；形状与权重一致 |
| **软** | 相对 GELU MLP 的收益看整模 |

### 4. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 加宽 `ffn_dim` | 容量↑、算力↑ |
| GELU MLP→SwiGLU | 与 Llama/Qwen 对齐；要重训 |
| 改成 MoE 专家 | 参数总量↑、单 token 稀疏算 |

### 5. 业界 · 开源

| 系 | 做法 |
| --- | --- |
| **Llama / Qwen dense** | SwiGLU |
| **Mixtral 等 MoE** | 专家为 SwiGLU 类 |
| **GPT-2 风教学** | GELU MLP |

### 6. 本课

```bash
python chapters/01-inference/ep07_swiglu.py
```

## 动手

```bash
python chapters/01-inference/ep07_swiglu.py
```

## 下一章

**Transformer 块**（`?c=block`）。
