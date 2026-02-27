"""Test generation-level kill switch: += scale until natural activation exceeds threshold, then stop entirely."""

import asyncio
import sys
import os

sys.path.insert(0, ".")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import torch
from nnsight.modeling.vllm import VLLM
from sae import FeatureManager, LlamaScopeSAE, ensure_downloaded
from sae import _worker_sae_cache

FM = FeatureManager()
FEATURE = "shakespeare"
LAYER, POS, EXP, IDX = FM.resolve_feature(FEATURE)
MAX_TOKENS = 80
PROMPT = "Tell me about your favorite vacation destination"


def _get_sae(x):
    key = (LAYER, POS, EXP)
    if key not in _worker_sae_cache:
        _worker_sae_cache[key] = LlamaScopeSAE.from_pretrained(
            LAYER, POS, EXP, device=str(x.device),
        )
    sae = _worker_sae_cache[key]
    if sae.encoder.weight.device != x.device or sae.encoder.weight.dtype != x.dtype:
        sae.to(device=x.device, dtype=x.dtype)
    return sae


def steer_additive(x, scale):
    """Plain additive (no kill switch, for reference)."""
    sae = _get_sae(x)
    encoded = sae.encode(x)
    recon = sae.decode(encoded)
    error = x - recon
    encoded[..., IDX] += scale
    return sae.decode(encoded) + error


def steer_with_killswitch(x, scale, cap_mult, state):
    """Additive with generation-level kill switch.

    If natural activation exceeds cap_mult * scale, set state["stopped"]
    and return x unmodified for this and all future steps.
    """
    if state.get("stopped"):
        return x

    sae = _get_sae(x)
    encoded = sae.encode(x)
    recon = sae.decode(encoded)
    error = x - recon

    # Check natural activation before modifying
    natural_max = encoded[..., IDX].max().item()
    if natural_max > cap_mult * scale:
        state["stopped"] = True
        return x

    encoded[..., IDX] += scale
    return sae.decode(encoded) + error


async def stream_response(tracer):
    full_text = ""
    async for output in tracer.backend():
        if output.outputs:
            full_text = output.outputs[0].text
    return full_text


def repetition_score(text):
    words = text.lower().split()
    if len(words) < 10:
        return 0.0, ""
    tail = words[-20:]
    from collections import Counter
    counts = Counter(tail)
    word, count = counts.most_common(1)[0]
    return count / len(tail), word


async def main():
    ensure_downloaded(LAYER, POS, EXP)

    print("Loading model...", flush=True)
    model = VLLM(
        "meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.4,
        max_model_len=4096,
        dispatch=True,
        mode="async",
    )
    print("Model loaded!\n", flush=True)

    # Baseline
    print(f"{'='*70}")
    print(f"  BASELINE")
    print(f"{'='*70}")
    with model.trace(PROMPT, temperature=0.0, max_tokens=MAX_TOKENS) as tracer:
        pass
    baseline = await stream_response(tracer)
    score, word = repetition_score(baseline)
    print(f"  [{score:.0%} rep] {baseline}\n")

    # Plain additive at scale=15 (for reference — should degenerate)
    print(f"{'='*70}")
    print(f"  ADDITIVE +=15 (no kill switch, for reference)")
    print(f"{'='*70}")
    with model.trace(PROMPT, temperature=0.0, max_tokens=MAX_TOKENS) as tracer:
        for step in tracer.iter[:]:
            layer_out = model.model.layers[LAYER].output
            hidden = layer_out[0].clone()
            model.model.layers[LAYER].output = (
                steer_additive(hidden, 15.0),
                *layer_out[1:],
            )
    text = await stream_response(tracer)
    score, word = repetition_score(text)
    quality = "OK" if score < 0.3 else "REPETITIVE" if score < 0.5 else "DEGENERATE"
    print(f"  [{quality} {score:.0%} rep ({word})] {text}\n")

    # Kill switch: vary scale and cap_mult
    for scale in [10.0, 15.0, 20.0]:
        for cap_mult in [2.0, 3.0, 5.0]:
            print(f"{'='*70}")
            print(f"  KILL SWITCH +=  {scale}, cap={cap_mult}x (stop all steering when natural > {cap_mult*scale})")
            print(f"{'='*70}")
            state = {}
            with model.trace(PROMPT, temperature=0.0, max_tokens=MAX_TOKENS) as tracer:
                for step in tracer.iter[:]:
                    layer_out = model.model.layers[LAYER].output
                    hidden = layer_out[0].clone()
                    model.model.layers[LAYER].output = (
                        steer_with_killswitch(hidden, scale, cap_mult, state),
                        *layer_out[1:],
                    )
            text = await stream_response(tracer)
            stopped = "STOPPED" if state.get("stopped") else "never triggered"
            score, word = repetition_score(text)
            quality = "OK" if score < 0.3 else "REPETITIVE" if score < 0.5 else "DEGENERATE"
            print(f"  [kill={stopped}] [{quality} {score:.0%} rep ({word})] {text}\n")


if __name__ == "__main__":
    asyncio.run(main())
