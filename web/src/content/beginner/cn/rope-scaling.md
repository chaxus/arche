## 一句话
Llama 3.2 拉伸长波长，让上下文能远超原始窗口。

## 开始之前
- RoPE 与 KV cache。约 15 分钟。

## 用人话讲
按短上下文训练的 RoPE，突然拉到很长会出问题。
Llama 3.2 保留大 `rope_theta`，再用 **factor / low / high** 调度：短波长不动，长波长拉伸，中间平滑。

## 对应代码
- `arche/rope.py` / attention 配置
- `chapters/01-inference/ep13_rope_scaling.py`

## 现在就做
```bash
python chapters/01-inference/ep13_rope_scaling.py
```

## 如果翻车了
- 小 demo 看不到 128k：关注打印的频率变化即可。

## 先别深挖
YaRN/NTK 争论；这里教本仓库用的 Llama 3.2 日程。

## 下一章
**MoE**（`?c=moe`）。
