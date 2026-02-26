"""Chat with a model using NNsight's async vLLM engine.

Tokens stream back as they're generated. Multi-turn conversation
history is maintained via the model's chat template.

Usage:
    python chat.py
    python chat.py --model meta-llama/Llama-3.1-8B-Instruct --tp 1
    python chat.py --model meta-llama/Llama-3.3-70B-Instruct --tp 4
"""

import argparse
import asyncio

from nnsight.modeling.vllm import VLLM


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with a model via NNsight async vLLM")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="HuggingFace model ID (default: Llama-3.3-70B-Instruct)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=4,
        help="Tensor parallel size (default: 4)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens per response (default: 1024)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--gpu-mem",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    print(f"Loading {args.model} (tp={args.tp})...")
    model = VLLM(
        args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        dispatch=True,
        mode="async",
    )
    print("Ready!\n")

    messages = []

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input.strip():
            continue
        if user_input.strip().lower() in ("exit", "quit"):
            print("Bye!")
            break

        messages.append({"role": "user", "content": user_input})

        prompt = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        with model.trace(
            prompt,
            temperature=args.temperature,
            top_p=0.9,
            max_tokens=args.max_tokens,
        ) as tracer:
            pass

        print("Assistant: ", end="", flush=True)
        full_text = ""
        async for output in tracer.backend():
            if output.outputs:
                new_text = output.outputs[0].text
                delta = new_text[len(full_text):]
                print(delta, end="", flush=True)
                full_text = new_text
        print()

        messages.append({"role": "assistant", "content": full_text})


if __name__ == "__main__":
    asyncio.run(main())
