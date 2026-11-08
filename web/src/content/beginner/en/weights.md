## Idea

### 1. What it is for

**Loading weights** copies trained tensors into the modules you built. A correct graph with random params is nonsense; official Llama 3.2 weights make the same forward a language model.

### 2. Why the LLM pipeline needs it

“Knowledge” lives in the checkpoint. Infer / align / fine-tune all start from a **correct load**. Wrong keys, tokenizer, or dtype → silent junk or crashes.

### 3. Where it comes from · how you build it

| Step | What happens |
| --- | --- |
| **1** | Build empty modules from config |
| **2** | Download HF Llama 3.2 (license + login) |
| **3** | Copy tensors via a **key map** into arche names |
| **4** | Use the matching official tokenizer |

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Param count / shapes match | Missing layers or wrong dims |
| **Hard** | Key map covers every tensor | Unmapped = still random |
| **Hard** | Tokenizer pairing | See tokenizer / embedding |
| **Soft** | Logits close to HF on a short prompt | Alignment chapter |

### 5. What labs ship · what we do

Official HF Llama 3.2 checkpoints.  
`arche/load_weights.py` + this episode’s script.

## Run

```bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
```

## Fix

Same as tokenizer: license, login, mirror.

## Next

**Text generation** (`?c=generate`).
