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
