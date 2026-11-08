## 要点

从零：一块不是语言模型；要 **嵌入 + N 块 + 最终 norm + LM head** 才有 next-token 分布。

### 1. 有什么作用

`ids → 嵌入 → Block×N → 最终 RMSNorm → Linear(V) → logits`

小 Llama 常 **tie** LM head ↔ 嵌入。

### 2. 从零开始：怎么生成

| 步骤 | 做什么 |
| --- | --- |
| **1** | 用已定 V、D、N、头数建空壳 |
| **2** | 串嵌入→块→最终 norm→head |
| **3** | 决定 tie 与否 |
| **4** | 随机初始化后要么加载权重，要么开预训练 |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | logits `(B,T,V)`；配置与权重可对上；tie 约定一致 |
| **软** | 装权/训完后文字与指标达标 |

### 4. 如何改进 · 对整模影响

加深加宽、改 tie、改词表 V——都是架构级，需重训或严格迁移。堆叠本身不「单独优化」，优化的是各子模块与训练。

### 5. 业界 · 开源

开放 dense（Llama/Qwen）同一骨架；差异在超参与数据。MoE 则把块内 FFN 换成路由专家，骨架仍在。

### 6. 本课

```bash
python chapters/01-inference/ep09_stack.py
```

## 动手

```bash
python chapters/01-inference/ep09_stack.py
```

## 下一章

**加载权重**（`/cn/weights/`）。
