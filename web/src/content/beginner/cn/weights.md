## 一句话
把官方 Hugging Face Llama 3.2 张量映射进手写的 `arche.Llama`。

## 开始之前
- 堆叠章已完成。约 20 分钟。
- 需要 HF 登录并接受协议（见第 0 部分地图）。

## 用人话讲
模块建好后还是空的。`from_pretrained` 搭好我们的图，下载 HF 权重，再按 key 对照表拷贝。

## 对应代码
- `arche/load_weights.py`
- `chapters/01-inference/ep10_load_weights.py`

## 现在就做
```bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
```

## 如果翻车了
- 门禁错误：接受协议并重新登录；网络问题可试镜像。
- 磁盘/内存不够：关掉其它应用；目标是 1B。

## 先别深挖
量化部署格式、多卡分片。

## 下一章
**文本生成**（`?c=generate`）。
