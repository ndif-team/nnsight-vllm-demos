"""Chat with SAE feature steering via Llama Scope.

Steer model behavior during chat using SAE feature directions.
Type `/steer shakespeare 15` to make responses Shakespearean.

Usage:
    python chat_steer.py --model meta-llama/Llama-3.1-8B-Instruct --tp 1
    python chat_steer.py --model meta-llama/Llama-3.3-70B-Instruct --tp 4
"""

import argparse
import asyncio
from collections import defaultdict
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

from nnsight.modeling.vllm import VLLM
from sae import FeatureManager

# ANSI color codes
DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"

# Position priority for forward-pass ordering.
# Within a layer: attention runs first, then MLP, then residual (post-MLP).
POSITION_PRIORITY = {"A": 0, "M": 1, "R": 2}


class SteerCompleter(Completer):
    """Tab-complete slash commands and feature names."""

    COMMANDS = ["/steer", "/clear", "/features", "/active", "/reset", "/help"]

    def __init__(self, fm):
        self.fm = fm

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        words = text.split()

        # Complete commands when typing /
        if len(words) <= 1 and text.startswith("/"):
            for cmd in self.COMMANDS:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))
        # Complete feature names after "/steer " with partial input
        elif len(words) == 2 and words[0].lower() == "/steer":
            partial = words[1]
            for feat in self.fm.list_features():
                if feat["name"].startswith(partial):
                    yield Completion(feat["name"], start_position=-len(partial))
        # Complete all feature names after "/steer " with no partial yet
        elif len(words) == 1 and words[0].lower() == "/steer" and text.endswith(" "):
            for feat in self.fm.list_features():
                yield Completion(feat["name"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chat with SAE steering via NNsight async vLLM"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID (default: Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)",
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


def build_sorted_steerings(active_steerings, fm):
    """Group active steerings by SAE and sort by forward-pass execution order.

    Returns a list of (layer, position, sae, [(feature_index, scale), ...])
    sorted by (layer, POSITION_PRIORITY[position]) so module accesses
    respect forward-pass order within a single invoke.
    """
    grouped = defaultdict(list)
    for name, info in active_steerings.items():
        key = (info["layer"], info["position"], info["expansion"])
        grouped[key].append((info["index"], info["scale"]))

    result = []
    for (layer, pos, exp), mods in grouped.items():
        sae = fm.get_sae(layer, pos, exp)
        result.append((layer, pos, sae, mods))

    return sorted(result, key=lambda x: (x[0], POSITION_PRIORITY[x[1]]))


def show_features(fm):
    """Print the curated feature catalog."""
    print(f"\n{BOLD}Available features:{RESET}")
    for feat in fm.list_features():
        print(
            f"  {CYAN}{feat['name']:12s}{RESET}  L{feat['layer']}{feat['position']}-{feat['expansion']}  "
            f"idx {feat['index']:>6d}  {DIM}{feat['description']}{RESET}"
        )
    print(f"\nOr use numeric format: /steer <layer>:<index> <scale>")
    print()


def show_active(active_steerings):
    """Print currently active steerings."""
    if not active_steerings:
        print(f"\n{DIM}No active steerings.{RESET}\n")
        return
    print(f"\n{BOLD}Active steerings:{RESET}")
    for name, info in active_steerings.items():
        print(
            f"  {YELLOW}{name:12s}{RESET}  L{info['layer']}{info['position']}  "
            f"scale={info['scale']}"
        )
    print()


def show_help():
    """Print commands reference."""
    print(f"""
{BOLD}Commands:{RESET}
  {CYAN}/steer <feature> [scale=10]{RESET}   Activate a steering direction
  {CYAN}/clear{RESET}                         Remove all steerings
  {CYAN}/reset{RESET}                         Clear chat history
  {CYAN}/features{RESET}                      List available features
  {CYAN}/active{RESET}                        Show current steerings
  {CYAN}/help{RESET}                          Show this help
""")


def handle_command(line, active_steerings, messages, fm):
    """Handle a slash command. Returns True if the input was a command."""
    parts = line.strip().split()
    cmd = parts[0].lower()

    if cmd == "/features":
        show_features(fm)
        return True

    if cmd == "/active":
        show_active(active_steerings)
        return True

    if cmd == "/clear":
        active_steerings.clear()
        print(f"\n{YELLOW}All steerings cleared.{RESET}\n")
        return True

    if cmd == "/reset":
        messages.clear()
        print(f"\n{YELLOW}Chat history cleared.{RESET}\n")
        return True

    if cmd == "/help":
        show_help()
        return True

    if cmd == "/steer":
        if len(parts) < 2:
            print(f"\n{BOLD}Usage:{RESET} /steer <feature> [scale]")
            print(f"  e.g.  /steer shakespeare 15")
            print(f"  e.g.  /steer 28:8401 10")
            print(f"  {DIM}Default scale: 10{RESET}\n")
            return True
        feature_spec = parts[1]
        try:
            scale = float(parts[2]) if len(parts) >= 3 else 10.0
        except ValueError:
            print(f"\n{RED}Invalid scale: {parts[2]!r} (must be a number){RESET}\n")
            return True
        try:
            layer, pos, exp, idx = fm.resolve_feature(feature_spec)
        except ValueError as e:
            print(f"\n{RED}{e}{RESET}\n")
            return True
        # Pre-load the SAE so the first generation isn't slow
        print(f"{DIM}Loading SAE for L{layer}{pos}-{exp}...{RESET}", end="", flush=True)
        fm.get_sae(layer, pos, exp)
        print(f"\r\033[K", end="")  # Clear the loading message
        active_steerings[feature_spec] = {
            "layer": layer,
            "position": pos,
            "expansion": exp,
            "index": idx,
            "scale": scale,
        }
        print(f"\n{YELLOW}Steering active: {feature_spec} at scale {scale} (L{layer}{pos}){RESET}\n")
        return True

    print(f"\n{RED}Unknown command: {cmd}{RESET}")
    print(f"Commands: /steer, /clear, /reset, /features, /active, /help\n")
    return True


def print_banner(args):
    """Print the welcome banner."""
    print(f"""
  {BOLD}SAE Steering Chat{RESET}
  {DIM}Model: {args.model} (tp={args.tp}){RESET}

  {BOLD}Commands:{RESET}
    {CYAN}/steer <feature> [scale=10]{RESET}   Activate a steering direction
    {CYAN}/clear{RESET}                         Remove all steerings
    {CYAN}/reset{RESET}                         Clear chat history
    {CYAN}/features{RESET}                      List available features
    {CYAN}/active{RESET}                        Show current steerings
    {CYAN}/help{RESET}                          Show this help

  Type {CYAN}/features{RESET} to browse, or just start chatting.
""")


async def main():
    args = parse_args()
    fm = FeatureManager()

    print(f"\n{DIM}Loading {args.model} (tp={args.tp})...{RESET}")
    model = VLLM(
        args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        dispatch=True,
        mode="async",
    )
    print_banner(args)

    messages = []
    active_steerings = {}

    session = PromptSession(
        history=FileHistory(str(Path("~/.chat_steer_history").expanduser())),
        auto_suggest=AutoSuggestFromHistory(),
        completer=SteerCompleter(fm),
    )

    def toolbar_fn():
        if not active_steerings:
            return HTML("<b>No active steerings</b> | /help for commands")
        parts = [f"{n}={info['scale']}" for n, info in active_steerings.items()]
        return HTML(f"<b>Steerings:</b> {' | '.join(parts)} | /clear to reset")

    while True:
        try:
            with patch_stdout():
                user_input = await session.prompt_async(
                    HTML("<b>You:</b> "),
                    bottom_toolbar=toolbar_fn,
                )
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Bye!{RESET}")
            break

        if not user_input.strip():
            continue
        if user_input.strip().lower() in ("exit", "quit"):
            print(f"{DIM}Bye!{RESET}")
            break

        if user_input.strip().startswith("/"):
            handle_command(user_input, active_steerings, messages, fm)
            continue

        messages.append({"role": "user", "content": user_input})

        prompt = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if active_steerings:
            sorted_steerings = build_sorted_steerings(active_steerings, fm)

            with model.trace(
                prompt,
                temperature=args.temperature,
                top_p=0.9,
                max_tokens=args.max_tokens,
            ) as tracer:
                for iteration in tracer.iter[:]:
                    for layer, position, sae, mods in sorted_steerings:
                        # vLLM runs in inference_mode — clone to leave
                        # inference mode, then encode → modify → decode + error.
                        if position == "R":
                            layer_out = model.model.layers[layer].output
                            hidden = layer_out[0].clone()
                            model.model.layers[layer].output = (
                                sae.steer(hidden, mods),
                                *layer_out[1:],
                            )
                        elif position == "M":
                            out = model.model.layers[layer].mlp.output.clone()
                            model.model.layers[layer].mlp.output = sae.steer(out, mods)
                        elif position == "A":
                            out = model.model.layers[layer].self_attn.output.clone()
                            model.model.layers[layer].self_attn.output = sae.steer(out, mods)
        else:
            with model.trace(
                prompt,
                temperature=args.temperature,
                top_p=0.9,
                max_tokens=args.max_tokens,
            ) as tracer:
                pass

        print(f"{BOLD}{GREEN}Assistant:{RESET} ", end="", flush=True)
        full_text = ""
        async for output in tracer.backend():
            if output.outputs:
                new_text = output.outputs[0].text
                delta = new_text[len(full_text):]
                print(f"{GREEN}{delta}{RESET}", end="", flush=True)
                full_text = new_text
        print()

        messages.append({"role": "assistant", "content": full_text})


if __name__ == "__main__":
    asyncio.run(main())
