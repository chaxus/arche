## 要点

围着算子核心的三块实用东西：

1. **设备** — 有 CUDA 用 CUDA，否则 Apple MPS，再否则 CPU（`pick_device`）  
2. **INT8（教学）** — 权重存成 int8 省内存；计算走简单反量化路径，方便学  
3. **对话模板** — 把 system/user/assistant 消息排成模型认识的字符串；没有 HF 模板时回退成可读纯文本（可试本页 demo）

## 动手

```bash
python chapters/01-inference/ep16_device_quant_chat.py
```

预期：选中的设备、量化说明、渲染后的对话串。

## 下一章

**连续批处理**（`?c=continuous-batching`）：多段短生成共享步进。
