"""Minimal test for SAE steering with vLLM async engine."""

import asyncio
import sys
import os

sys.path.insert(0, ".")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import torch
from collections import defaultdict
from nnsight.modeling.vllm import VLLM
from sae import FeatureManager, steer_activations, ensure_downloaded

POSITION_PRIORITY = {"A": 0, "M": 1, "R": 2}


def build_sorted_steerings(active_steerings):
    grouped = defaultdict(list)
    for name, info in active_steerings.items():
        key = (info["layer"], info["position"], info["expansion"])
        grouped[key].append((info["index"], info["scale"]))
    return sorted(
        [(layer, pos, exp, mods) for (layer, pos, exp), mods in grouped.items()],
        key=lambda x: (x[0], POSITION_PRIORITY[x[1]]),
    )


def apply_steerings(model, sorted_steerings):
    """Apply steering interventions inside a trace context.

    Calls steer_activations() which lazy-loads the SAE on the worker process.
    """
    for layer, position, exp, mods in sorted_steerings:
        if position == "R":
            layer_out = model.model.layers[layer].output
            hidden = layer_out[0].clone()
            model.model.layers[layer].output = (
                steer_activations(hidden, layer, position, exp, mods),
                *layer_out[1:],
            )
        elif position == "M":
            out = model.model.layers[layer].mlp.output.clone()
            model.model.layers[layer].mlp.output = steer_activations(
                out, layer, position, exp, mods
            )
        elif position == "A":
            out = model.model.layers[layer].self_attn.output.clone()
            model.model.layers[layer].self_attn.output = steer_activations(
                out, layer, position, exp, mods
            )


async def stream_response(tracer):
    """Stream and collect a response."""
    full_text = ""
    async for output in tracer.backend():
        if output.outputs:
            new_text = output.outputs[0].text
            delta = new_text[len(full_text):]
            print(delta, end="", flush=True)
            full_text = new_text
    print()
    return full_text


async def main():
    print("Loading model...", flush=True)
    model = VLLM(
        "meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        dispatch=True,
        mode="async",
    )
    print("Model loaded!\n", flush=True)

    fm = FeatureManager()
    passed = 0
    failed = 0

    # Test 1: Plain trace
    print("--- Test 1: Plain trace ---", flush=True)
    try:
        with model.trace("The capital of France is", temperature=0.0, max_tokens=10) as tracer:
            pass
        text = await stream_response(tracer)
        assert len(text.strip()) > 0, "Empty response"
        print(f"  PASS\n", flush=True)
        passed += 1
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}\n", flush=True)
        import traceback; traceback.print_exc()
        failed += 1

    # Test 2: Single R steering (dragons L22R) via SAE encode-modify-decode
    print("--- Test 2: Single R steering (dragons) ---", flush=True)
    try:
        layer, pos, exp, idx = fm.resolve_feature("dragons")
        ensure_downloaded(layer, pos, exp)
        active = {"dragons": {"layer": layer, "position": pos, "expansion": exp, "index": idx, "scale": 15.0}}
        sorted_steerings = build_sorted_steerings(active)

        with model.trace("Tell me about Paris", temperature=0.0, max_tokens=30) as tracer:
            apply_steerings(model, sorted_steerings)

        text = await stream_response(tracer)
        assert len(text.strip()) > 0, "Empty response"
        print(f"  PASS\n", flush=True)
        passed += 1
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}\n", flush=True)
        import traceback; traceback.print_exc()
        failed += 1

    # Test 3: Second steering request (reproduces the user's bug)
    print("--- Test 3: Second steered request ---", flush=True)
    try:
        with model.trace("What is the meaning of life?", temperature=0.0, max_tokens=30) as tracer:
            apply_steerings(model, sorted_steerings)

        text = await stream_response(tracer)
        assert len(text.strip()) > 0, "Empty response"
        print(f"  PASS\n", flush=True)
        passed += 1
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}\n", flush=True)
        import traceback; traceback.print_exc()
        failed += 1

    # Test 4: MLP steering (drama L16M)
    print("--- Test 4: MLP steering (drama) ---", flush=True)
    try:
        layer, pos, exp, idx = fm.resolve_feature("drama")
        ensure_downloaded(layer, pos, exp)
        active = {"drama": {"layer": layer, "position": pos, "expansion": exp, "index": idx, "scale": 15.0}}
        sorted_steerings = build_sorted_steerings(active)

        with model.trace("Tell me about Paris", temperature=0.0, max_tokens=30) as tracer:
            apply_steerings(model, sorted_steerings)

        text = await stream_response(tracer)
        assert len(text.strip()) > 0, "Empty response"
        print(f"  PASS\n", flush=True)
        passed += 1
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}\n", flush=True)
        import traceback; traceback.print_exc()
        failed += 1

    # Test 5: Multiple steerings (shakespeare L28R + romance L22R)
    print("--- Test 5: Multiple steerings ---", flush=True)
    try:
        active = {}
        for name in ["shakespeare", "romance"]:
            layer, pos, exp, idx = fm.resolve_feature(name)
            ensure_downloaded(layer, pos, exp)
            active[name] = {"layer": layer, "position": pos, "expansion": exp, "index": idx, "scale": 10.0}
        sorted_steerings = build_sorted_steerings(active)

        with model.trace("Tell me about Paris", temperature=0.0, max_tokens=30) as tracer:
            apply_steerings(model, sorted_steerings)

        text = await stream_response(tracer)
        assert len(text.strip()) > 0, "Empty response"
        print(f"  PASS\n", flush=True)
        passed += 1
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}\n", flush=True)
        import traceback; traceback.print_exc()
        failed += 1

    print(f"\n{'='*40}", flush=True)
    print(f"Results: {passed} passed, {failed} failed", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
