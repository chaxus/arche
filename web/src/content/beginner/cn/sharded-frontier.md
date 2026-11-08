## 一句话
写出 index.json + 分片，再装回 Mixtral 形 MoE 与 DeepSeek 形 MLA，并用 allclose 检查。

## 开始之前
- 前沿检查点章。约 15 分钟。

## 用人话讲
真实 HF 仓库用 index 列出分片。我们用合成 `.pt` 练习同形布局——不必下好几 GB。

## 对应代码
- `arche/shards.py`、`load_mixtral.py`、`load_deepseek_mla.py`
- `chapters/02-training/ep22_sharded_frontier.py`

## 现在就做
```bash
python chapters/02-training/ep22_sharded_frontier.py
```

## 如果翻车了
- 临时目录权限：在普通用户目录运行。

## 先别深挖
各种宿主上的 safetensors 细节。

## 下一章
**Paged KV · CUDA Graph**（`?c=paged-cuda-graph`）。
