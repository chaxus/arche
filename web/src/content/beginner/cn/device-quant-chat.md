## 要点

本章是上线胶水：**设备**放置、教学用 **INT8** 缩小，以及把角色变成 token id 的 **chat 模板**。稠密栈写对了，若跑不起来、塞不进显存、或对话格式错，用户照样失败——这些不是新层，是最后一公里。探测设备、可选量化菜谱、套 **本模型自己的** 模板，再接入 generate 和 KV。

### 仓库里的零件

| 零件 | 角色 |
| --- | --- |
| 设备回退 | 演示用 CPU/CUDA/MPS |
| INT8 玩具 | 教学缩小——不是生产 GPTQ/AWQ/FP8 |
| Chat 模板 | [`apply_chat_template` / `encode_chat`](https://github.com/chaxus/arche/blob/main/arche/chat.py) |

模板错 ⇒ 权重再完美也会「无视」system/user 结构。

### 产品上的压力

| 压力 | 应对 |
| --- | --- |
| 没 GPU | CPU 通路要能跑（实验慢可接受） |
| 显存紧 | 量化 / 更小 dtype |
| 多轮体验 | 官方 chat 模板 + specials |
| 多后端 | 工程成本，不是新数学 |

### 跑通演示

| 步骤 | 做什么 |
| --- | --- |
| 1 | 探测可用设备 |
| 2 | 以可跑 dtype 加载权重 |
| 3 | 用**配对**模板格式化消息 |
| 4 | 照常带 KV 生成 |

### 自己核对

| 必须成立 | 回退能跑；模板对齐分词器/模型 |
| 尽量做到 | 量化质量达到演示条 |
| 尽量做到 | 角色在 prompt 字符串/ids 里清晰 |

### 走出玩具路径

| 杠杆 | 影响 |
| --- | --- |
| 更狠的量化 | 更小/更快/更险 |
| 错模板 | 对话垃圾 |
| 生产量化 | 引擎地盘（vLLM/TGI/llama.cpp） |

### 本课

```bash
python chapters/01-inference/ep16_device_quant_chat.py
```

## 动手

```bash
python chapters/01-inference/ep16_device_quant_chat.py
```

## 下一章

**连续批处理**（`/cn/continuous-batching/`）。
