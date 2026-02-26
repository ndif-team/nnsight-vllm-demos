# Chat

Stream tokens from a chat model using NNsight's async vLLM engine.

This demo shows the simplest use of NNsight + vLLM: a multi-turn chat interface with token-by-token streaming. No interventions — just the async engine doing what vLLM does best, but through NNsight's tracing interface.

## How it works

NNsight's `mode="async"` uses vLLM's `AsyncLLM` under the hood. The flow:

1. **Trace setup** — `model.trace(prompt, ...)` compiles the trace and prepares sampling params. No generation happens yet.
2. **Streaming** — `async for output in tracer.backend()` submits the request to `AsyncLLM.generate()` and yields `RequestOutput` objects as tokens are generated.
3. **Delta printing** — Each output contains the full text so far. We print only the new characters since last output.

```python
with model.trace(prompt, temperature=0.7, max_tokens=1024) as tracer:
    pass  # no interventions needed for plain chat

async for output in tracer.backend():
    # output.outputs[0].text grows with each token
    delta = output.outputs[0].text[len(full_text):]
    print(delta, end="", flush=True)
    full_text = output.outputs[0].text
```

The trace context is where you'd normally add interventions (saving activations, modifying hidden states, etc.). Here we leave it empty — the point is just streaming generation.

## Run

```bash
# Default: Llama-3.3-70B-Instruct on 4 GPUs
python chat.py

# Smaller model, single GPU
python chat.py --model meta-llama/Llama-3.1-8B-Instruct --tp 1

# Adjust generation params
python chat.py --temperature 0.3 --max-tokens 512
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `meta-llama/Llama-3.3-70B-Instruct` | HuggingFace model ID |
| `--tp` | `4` | Tensor parallel size (number of GPUs) |
| `--max-tokens` | `1024` | Max tokens per response |
| `--temperature` | `0.7` | Sampling temperature |
| `--gpu-mem` | `0.9` | GPU memory utilization |

Type `exit`, `quit`, or Ctrl+C to end the session.

---

## Chat with SAE Steering (`chat_steer.py`)

Steer model behavior during chat using SAE feature directions from [Llama Scope](https://huggingface.co/fnlp/Llama-Scope). Type `/steer shakespeare 15` and subsequent responses take on a Shakespearean tilt.

### What is SAE steering?

Sparse Autoencoders (SAEs) learn to decompose model activations into interpretable features. Each feature has a **direction** in activation space (a decoder column). By adding a scaled direction to the model's hidden states during generation, we push the model toward that feature's behavior.

The math: the full SAE steering formula `decode(encode(x) + Δ) + error` simplifies to `x + scale * decoder_column[feature]` because the decoder is linear and the error term cancels. This means we only need one decoder column (~8KB) per feature — not the full SAE (~512MB).

### Commands

| Command | Description |
|---------|-------------|
| `/steer <feature> <scale>` | Activate a steering direction |
| `/clear` | Remove all active steerings |
| `/features` | List available features |
| `/active` | Show current steerings |

Features can be curated names (`shakespeare`, `pirate`, `code`, etc.) or numeric `layer:index` format (`28:8401`).

### Run

```bash
# Llama-3.1-8B-Instruct on 1 GPU
python chat_steer.py --model meta-llama/Llama-3.1-8B-Instruct --tp 1
```

### Example interaction

```
You: Tell me about Paris
Assistant: Paris is the capital of France, known for the Eiffel Tower...

You: /steer shakespeare 15
Steering active: shakespeare at scale 15 (L28R)

You: Tell me about Paris
Assistant: Hark! Paris, that fair jewel upon the Seine, doth bewitch...

You: /steer romance 10
Steering active: romance at scale 10 (L22R)

You: Tell me about Paris
Assistant: O Paris! City of lovers, where hearts entwine beneath...

You: /clear
All steerings cleared.
```

### How it works

1. **Feature directions are tiny** — `FeatureManager` uses `safetensors.get_slice()` to extract a single decoder column (~8KB) from the full SAE safetensors file. The file itself is cached by `hf_hub_download`.

2. **Steerings are sorted by forward-pass order** — Multiple steerings on different layers/positions are grouped and sorted by `(layer, position_priority)` to respect NNsight's requirement that module accesses follow execution order. Features on the same `(layer, position)` share a single `.output` access.

3. **Clone before modifying** — vLLM runs in `torch.inference_mode()`, which forbids in-place ops on inference tensors. The code clones the output, modifies the clone, then assigns back (matching vLLM test patterns).

### Adding custom features

1. Browse features on [Neuronpedia Llama Scope](https://www.neuronpedia.org/llama-scope)
2. Add entries to `features.json`:
   ```json
   "my_feature": {
     "layer": 20,
     "position": "R",
     "expansion": "8x",
     "index": 12345,
     "description": "What this feature does"
   }
   ```
3. Or use numeric format directly: `/steer 20:12345 10`
