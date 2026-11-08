## 要点

从零模型要**给人用**：设备、体积、对话格式是上线胶水——不算新层，但缺一不可。

### 1. 有什么作用

1. **设备** CUDA→MPS→CPU  
2. **量化（教学 INT8）** 缩小权重  
3. **对话模板** 角色→模型字符串  

### 2. 从零开始：怎么做

| 步骤 | 做什么 |
| --- | --- |
| **1** | 探测后端，模块 `.to(device)` |
| **2** | 需要落地时选量化配方（本课 INT8 示意；产品常用 GPTQ/AWQ/FP8…） |
| **3** | 使用**该模型自带** chat template，勿自创 |
| **4** | 与生成循环、KV 一起接到服务 |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 无 CUDA 可回退；模板与模型配对 |
| **软** | 量化后质量可接受；多轮角色清晰 |

### 4. 如何改进 · 对整模影响

更强量化 → 更小更快、质量风险↑。错模板 → 胡话。多后端维护成本在工程不在算法。

### 5. 业界 · 开源

HF `apply_chat_template`；vLLM/TGI 内置模板与量化加载；Llama/Qwen 均带官方模板。llama.cpp 系偏 GGML 量化。

### 6. 本课

```bash
python chapters/01-inference/ep16_device_quant_chat.py
```

## 动手

```bash
python chapters/01-inference/ep16_device_quant_chat.py
```

## 下一章

**连续批处理**（`?c=continuous-batching`）。
