# nnsight + vLLM Demos

Demos of [NNsight](https://github.com/ndif-team/nnsight)'s vLLM integration — interpretability and intervention on production inference.

## Why

There is growing demand for deploying interpretability at scale: chat interfaces backed by production LLMs where you can observe internal representations, steer behavior via activation manipulation, and serve this to concurrent users — all without retraining.

NNsight gives you programmatic access to every intermediate tensor in a model's forward pass. The vLLM integration runs this on top of vLLM's high-performance inference engine — PagedAttention, continuous batching, tensor parallelism — so you get research-grade introspection with production-grade throughput.

This is not a constrained API with predefined operations. It's arbitrary Python running inside the forward pass. You can call `torch.svd()`, run a learned probe, apply an SAE, or do anything else you'd do in a research notebook — but it executes on vLLM workers with PagedAttention and TP sharding handled transparently.

## Demos

| Demo | Description |
|------|-------------|
| [chat](chat/) | Stream tokens from a chat model using NNsight's async vLLM engine |

## Setup

```bash
pip install nnsight
```

NNsight pins a specific vLLM version. If there's a mismatch, it will tell you the exact version to install.

## How It Works

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("meta-llama/Llama-3.1-70B", tensor_parallel_size=4, dispatch=True)

with model.trace("The Eiffel Tower is in", temperature=0.0) as tracer:
    # Read any layer
    h5 = model.model.layers[5].output[0].save()

    # Modify any layer
    model.model.layers[10].mlp.output[:] = 0

    # Access logits and sampling
    logits = model.logits.output.save()
```

1. **Tracing**: Your intervention code is extracted via AST parsing, compiled, and wrapped in a serializable mediator
2. **Transport**: The mediator rides vLLM's existing request pipeline via `SamplingParams.extra_args`
3. **Execution**: On GPU workers, a mediator thread synchronizes with PyTorch forward hooks — when your code reads `model.layers[5].output`, the thread blocks until layer 5 fires, receives the real tensor, and continues
4. **TP transparency**: Sharded tensors are gathered before your code sees them and re-sharded after. Your code always sees complete tensors regardless of `tensor_parallel_size`
5. **Collection**: Saved values are pickled back through vLLM's request output path to the user process

## Links

- [NNsight](https://github.com/ndif-team/nnsight) — the library
- [nnsight.net](https://nnsight.net) — documentation
- [NDIF](https://ndif.us) — run NNsight on shared GPU infrastructure without your own hardware
