## 要点

超大模型很少一个文件搞定。Hugging Face 风格常用 **`index.json`** + 许多 **`.safetensors` / `.pt` 分片**。

我们造一份 **合成** index + 分片（Mixtral 形 MoE、DeepSeek 形 MLA），再装回来做 `allclose`——看清打包/拆包，不必真下几十 GB。

## 动手

```bash
python chapters/02-training/ep22_sharded_frontier.py
```

预期：分片写入/读回，张量对齐。

## 下一章

**Paged KV · CUDA Graph**（`?c=paged-cuda-graph`）：服务端显存与捕获的思路。
