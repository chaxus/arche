## 一句话
把 logits 变成 token：采样（或贪心）选出下一个 id，拼上，重复直到结束。

## 开始之前
- 理解权重章（即使下载延后）。约 15 分钟。

## 用人话讲
每步看最后位置的分数。**temperature** 调节尖锐程度；**top-k** 只保留前 k 个。拼上 id；遇到 EOS 或长度上限就停。

## 对应代码
- `arche/generate.py`
- `chapters/01-inference/ep11_generate.py`

## 现在就做
```bash
python chapters/01-inference/ep11_generate.py
```
没有权重时，小 demo 仍能展示循环形状。

## 如果翻车了
- 随机权重输出乱码是正常的——流畅性来自 1B 权重。

## 先别深挖
KV cache（下一部分）与 chat template（更后）。

## 下一章
第 2 部分开始：**KV 缓存**（`?c=kv-cache`）。
