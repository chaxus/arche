## 一句话
选择设备，可选 INT8 存权重，并用模板渲染对话。

## 开始之前
- 会跑脚本。约 15 分钟。

## 用人话讲
`pick_device` 优先 CUDA → MPS → CPU。
**INT8 权重量化**更省空间（教学级）。
**Chat template** 格式化角色对话；有 HF 模板用官方的，否则回退纯文本。

## 对应代码
- `arche/device.py`、`quantize.py`、`chat.py`
- `chapters/01-inference/ep16_device_quant_chat.py`

## 现在就做
```bash
python chapters/01-inference/ep16_device_quant_chat.py
```
也可点页面上的小 demo。

## 如果翻车了
- 设备错误：强制 `"cpu"`。

## 先别深挖
GPTQ/AWQ 生产量化。

## 下一章
**连续批处理**（`?c=continuous-batching`）。
