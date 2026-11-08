## 要点

### 1. 嵌入表有什么作用

嵌入表是矩阵 `(词表大小, 维度)`：**第 `i` 行 = token id `i` 的那一排小数。**

| 作用 | 人话 |
| --- | --- |
| **① 接口** | id → 后面层能乘加的小数 |
| **② 可学习身份** | 每个 id 固定一行；训练改这些数 |
| **③ 常绑 LM head** | 小 Llama：进模型取行、出模型打分共用一张表 |

推理时几乎就是**按 id 取行**（本页上方小表可试）。不是词典释义。

```mermaid
flowchart LR
  text[文字] --> tok[分词表 → id]
  tok --> emb[嵌入表取行]
  emb --> rest[后面各层]
```

### 2. 为什么大模型过程需要它

分词只给出整数；`3+7=10` 没有语言意义。内部全是张量运算，必须有「编号 → 可计算小数」的一层。现代 decoder 用查表：简单、可训、与词表一行对一个 id。

### 3. 嵌入表是怎么生成的

**不是**手写小数，也**不是**再跑 BPE。分词表管「谁叫几号」；嵌入表管「几号对应哪串可学习的数」。

| 阶段 | 发生什么 |
| --- | --- |
| **1. 定形状** | 行数 = 分词词表大小；列数 = 模型维度（超参） |
| **2. 初始化** | `nn.Embedding`：随机小数（此时几乎无意义） |
| **3. 预训练** | 预测下一 token；loss 反传，**改被用到的行**（及别的层） |
| **4. 存盘** | 写入检查点，与注意力、MLP 等放一起 |
| **5. 你使用** | 下载权重 → 加载已训好的表；推理默认只取行 |
| **6. 微调（可选）** | 继续改数，或表尾加行（新特殊符）再训 |

和分词对照：

| | 分词表 | 嵌入表 |
| --- | --- | --- |
| 生成算法 | BPE 等在语料上合并 | 随机初始化 + **梯度更新** |
| 产物文件 | `tokenizer.json` 等 | 检查点里的 `embed_tokens.weight` 一类 |
| 推理 | 编码/解码 | 按 id gather |

### 4. 代码：建表 · 取行 · 训练改一行

库代码（`arche/embedding.py`）就是一张表：

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.weight = nn.Embedding(vocab_size, embed_dim)  # 形状 (V, D)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight(token_ids)  # 按 id 取行
```

「生成」在训练里发生——玩具一步（与本章脚本相同思路）：

```python
emb = TokenEmbedding(vocab_size=32, embed_dim=8)  # 随机初始化 = 建表
ctx = torch.tensor([[3, 7, 11]])
target = torch.tensor([5])                        # 假装下一 token
h = emb(ctx)[:, -1, :]                            # 取最后位置那一行参与预测
loss = F.cross_entropy(head(h), target)
loss.backward()
opt.step()                                        # 第 11 行等被用到的行会动
```

跑完整演示（假表，免下载）：

```bash
python chapters/01-inference/ep03_embedding.py
```

预期：表形状、取出的行、一步 SGD 后某行的平均变化量、有限数值检查。

真 Llama 嵌入在官方权重里，行数 ≈ `128256`，维度 `2048`（1B）；必须配官方分词表。

### 5. 怎样算「好」

**硬约束（不对就坏）**

| 标准 | 人话 |
| --- | --- |
| **形状** | `(V, D)`；`T` 个 id → `T` 行 |
| **配对** | 第 `i` 行 ↔ 分词器 id `i` |
| **数值** | 无 NaN/Inf；尺度别炸 |
| **架构约定** | 本课栈用 RoPE，嵌入上不加绝对 PE |

**好不好（和分词不同）**

分词可单独看 fertility。嵌入小数是**整网训出来的**，很少单独打官方「嵌入分」。常见信号：

| 看什么 | 人话 |
| --- | --- |
| **预训练 / 下游指标** | 同架构下 loss、评测——主裁判 |
| **维度预算** | 太小不够表达；太大更难吃数据 |
| **稀有 token** | 很少出现的 id，那一行几乎不更新 |
| **邻域直觉（辅助）** | 近邻是否像相关子词；仅体检 |
| **绑权** | 小模型常绑 LM head |

结论：合格 = 形状对 + 配对 + 能稳定训推；好 = 整模表现好。用现成模型时拿官方嵌入 + 官方分词，不要外接「别家更好的嵌入表」。

### 6. 业界实际 · 本课怎么做

Llama 3.2 / Qwen：查表；位置交给 RoPE；小模型常绑 LM head。  
本课：`TokenEmbedding` 假表看取行与一步更新；真表在检查点里。

## 动手

```bash
python chapters/01-inference/ep03_embedding.py
```

## 下一章

**RMSNorm**（`?c=rmsnorm`）：小数进重层前先稳住尺度。
