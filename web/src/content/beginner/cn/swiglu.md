## 要点

### 1. 为什么注意力后还要 MLP

注意力混合各位置；每个位置还要**各自**变换特征。

### 2. SwiGLU 做什么

$$\mathrm{FFN}(x)=W_d\big(\mathrm{SiLU}(W_g x)\odot W_u x\big)$$

三条无偏置线性：**gate**、**up**、**down**。

### 3. 其他做法

| FFN | 人话 | 今天 |
| --- | --- | --- |
| GELU MLP（GPT-2 风） | 一层隐层 + GELU | 偏考古教学 |
| **SwiGLU** | 门控 SiLU × up | **Llama / Qwen dense** |
| SwiGLU 专家 MoE | 路由 top-k | Mixtral 风（后面） |

### 4. 业界实际

现代 dense 解码器：SwiGLU（或近亲门控）。

### 5. 怎样算对

- 进出 `embed_dim`；中间更宽  
- 三层无 bias（Llama 风格）  

### 6. 本课怎么做

`arche/mlp.py` 的 `SwiGLU`。

## 动手

```bash
python chapters/01-inference/ep07_swiglu.py
```

## 下一章

**Transformer 块**（`?c=block`）：拼成三明治。
