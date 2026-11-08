## 要点

本章用 **键名映射** 把开源 Llama 3.2 检查点 **加载** 进你手写的栈：safetensors / HF 名字通过映射落到 arche 的 `state_dict`。随机权重是「图对、字胡说」；官方小数让同一次前向马上像语言模型。预训练章讲过小数怎么被写出来，这里 **借用** 一次跑完的结果，不必自己上集群也能生成和对齐。接受协议、下载、映射键、配对 **官方分词器**，再核对形状。

### 入口

[`from_pretrained`](https://github.com/chaxus/arche/blob/main/arche/load_weights.py) 建 `Llama(cfg)`，把 HF 张量拷进对应名字。

| 路径 | 作用 |
| --- | --- |
| **加载（本章）** | 例子 checkpoint → 马上能说话 |
| **预训练（更早）** | 这些小数从零怎么被写出 |
| **HF 对齐（稍后）** | 证明映射 + 公式对齐 HF logits |

### 映射要注意什么

| 风险 | 后果 |
| --- | --- |
| 键错 / 转置错 | 静默垃圾或崩溃 |
| 分词器错 | id 指到错误嵌入行 |
| dtype / 设备滑 | 慢或 NaN |
| 只加载一部分 | 「能跑」但层是坏的 |

### 清单

| 步骤 | 做什么 |
| --- | --- |
| 1 | 若门禁：`huggingface-cli login` |
| 2 | 解析模型 id（课默认 Llama 3.2 ~1B） |
| 3 | Config → 空 `Llama` → 拷权重 |
| 4 | **永远**用与 checkpoint 同一套分词器 |

### 自己核对

| 必须成立 | 无 missing/unexpected；形状对齐 |
| 必须成立 | 词表大小 = 嵌入行数 |
| 尽量做到 | 短生成像人话；随后 HF 对齐 |

### 对整模的杠杆

| 杠杆 | 影响 |
| --- | --- |
| 修键映射 | 解锁后续全部演示 |
| 恢复自己的 ckpt | 同一 API，你的小数 |
| 量化加载 | 见后部服务章 |

### 本课

```bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
```

## 动手

```bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
```

## 翻车

协议、登录、镜像——与分词器下载同类问题。

## 下一章

**文本生成**（`/cn/generate/`）：在 logits 上循环采样。
