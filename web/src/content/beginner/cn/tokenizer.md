## 要点

### 1. 分词表有什么作用

分词表（词表 + 切分规则）做两件事：

1. **切** — 文字 → 小段（token）  
2. **编号** — 小段 → 固定整数 **token id**

进模型前编码，出模型后用**同一张表**解码。特殊符（BOS/EOS/…）也是表里的固定 id。

```mermaid
flowchart LR
  text[文字] --> cut[切成 token]
  cut --> ids[token id]
  ids --> model[模型]
  model --> ids2[新 id]
  ids2 --> back[解码成文字]
```

一句话：**同一套规则，文字 ↔ 编号。**

### 2. 为什么大模型过程需要它

模型只会算数字。整句当一个符号 → 词表炸；纯字节 → 序列太长。  
分词找中间态：稳定、可逆、长度可控。没有分词表，后面的**嵌入表**没有合法行号可对。

### 3. 分词表是怎么生成的

主流开放解码器用 **字节级 BPE**（Llama / 多数 GPT 系同思路）。**不是**人手工给每个词编号。

| 步骤 | 做什么 |
| --- | --- |
| **1. 准备语料** | 大量文本（决定哪些片段常见） |
| **2. 底层词表** | 从 UTF-8 **256 个字节**起（或等价字节符号） |
| **3. 反复合并** | 统计相邻两片共现；最常见的合并成新 token；写入 merges 列表 |
| **4. 停下来** | 达到目标词表大小（如 ~128K），再加特殊符 |
| **5. 存盘** | `tokenizer.json` / merges + 特殊符；和模型一起发布 |
| **6. 你使用** | **加载官方表**编码/解码；一般不再自己训（除非从头做模型） |

和嵌入表的分工：分词表由 **BPE 等算法在语料上跑出来**；嵌入表由 **神经网络训练改小数**（下一章）。

本页上方是**预置 merges** 的教学 BPE（看直觉）。下面脚本可**真的在小语料上训出一张表**。

### 4. 代码：怎么生成 · 怎么用 · 怎么量

教学实现在 `arche/teach_bpe.py`（字节 BPE，极简）。核心是「数 pair → 合并」：

```python
# 伪代码 / 与 arche.teach_bpe.train_bpe 同思路
vocab = [bytes([i]) for i in range(256)]   # 底层
merges = []
for _ in range(num_merges):
    # 在语料里数相邻 pair 频率
    a, b = most_frequent_adjacent_pair(corpus)
    merges.append((a, b))
    vocab.append(a + b)                    # 新 token
    corpus = apply_merge(corpus, a, b)     # 语料里合并
```

编码：先拆成字节，再**按 merges 顺序**合并，查表得 id。解码：id → 字节 → UTF-8。

先跑教学训表（**不用** Hugging Face）：

```bash
python chapters/01-inference/ep02_teach_bpe.py
```

预期：打印 `vocab_size`、某句的 ids、decode 还原、`fertility≈`（约：每词多少 token）。

再用**官方 Llama 3.2 表**（与权重配对；要许可）：

```python
from arche.tokenizer import load_tokenizer
tok = load_tokenizer()          # meta-llama/Llama-3.2-1B
ids = tok.encode("The cat sat on the mat.")
text = tok.decode(ids)
```

```bash
huggingface-cli login
python chapters/01-inference/ep02_tokenizer.py
```

### 5. 怎样算「好」

| 标准 | 人话 | 怎么看 |
| --- | --- | --- |
| **可逆** | encode→decode 回到原文（特殊空白规则下可有规范化） | 脚本里 decode 对比 |
| **压缩 / fertility** | 同样文字更少 token 更省；fertility = 平均一词几片，**越大越碎** | 教学脚本打印；官方表可对平行语料比 |
| **语言公平** | 中英等同义句长度别差太多 | 平行句 token 数比 |
| **结构** | 别切坏代码缩进、URL、数字 | 抽样检查 |
| **与检查点配对** | 第 `i` 号必须对嵌入第 `i` 行 | **硬约束**；换家族表就乱 |
| **词表大小** | 更大常更省 token，但嵌入更胖 | 产品权衡 |
| **下游** | 最终仍看训完的模型好不好 | 主裁判之一 |

配对为何重要：训练时 `the`→`65`→嵌入第 65 行；换表变成 `902`，模型仍读 902 行 → 乱。

| 何时表会「不一样」 | 例子 |
| --- | --- |
| 换模型家族 | Llama ≠ Qwen ≠ `cl100k` |
| 同厂换代 | GPT-3.5 ~100K → GPT-4o ~200K |
| 自己从头训 | 新 BPE；旧嵌入对不上 |
| 加特殊符 | 表尾加 id |
| 操作失误 | Llama 权重 + 别家 `tokenizer.json` |

### 6. 业界实际 · 本课怎么做

| 系 | 约词表 | 备注 |
| --- | --- | --- |
| GPT 系 tiktoken | 100K–200K | BPE |
| Llama 3.2 | 128K+ | **本课官方表** |
| DeepSeek / Qwen / Kimi | ~128K–163K | 同路线，表各训 |

教学训表看「怎么生成」；官方表看「怎么和真模型配对」。两张**不要混用**。

## 动手

```bash
# 1) 自己训一张玩具分词表（看生成过程）
python chapters/01-inference/ep02_teach_bpe.py

# 2) 官方 Llama 3.2 表（要许可）
huggingface-cli login
python chapters/01-inference/ep02_tokenizer.py
```

大陆网络可：`export HF_ENDPOINT=https://hf-mirror.com`

## 翻车

官方表：未许可 / 未 login / `401` / 下载慢 → 许可、重新 login、HF 镜像。  
教学 BPE 不需要网络。

## 下一章

**嵌入层**（`?c=embedding`）：分词给出 id 之后，嵌入表如何生成、取行、算好。
