"""Llama Scope SAE loading and feature steering.

LlamaScopeSAE — Full JumpReLU SAE with encode → modify → decode steering.
FeatureManager — Loads SAEs on demand and resolves feature specs to SAE coordinates.
"""

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors import safe_open


# HuggingFace repo IDs for Llama Scope SAEs.
# Each repo contains all 32 layers for one (position, expansion) combination.
# Files inside: Llama3_1-8B-Base-L{layer}{pos}-{exp}/checkpoints/final.safetensors
_REPOS = {
    ("R", "8x"): "fnlp/Llama3_1-8B-Base-LXR-8x",
    ("R", "32x"): "fnlp/Llama3_1-8B-Base-LXR-32x",
    ("M", "8x"): "fnlp/Llama3_1-8B-Base-LXM-8x",
    ("M", "32x"): "fnlp/Llama3_1-8B-Base-LXM-32x",
    ("A", "8x"): "fnlp/Llama3_1-8B-Base-LXA-8x",
    ("A", "32x"): "fnlp/Llama3_1-8B-Base-LXA-32x",
}


def _repo_id(position: str, expansion: str) -> str:
    key = (position.upper(), expansion)
    if key not in _REPOS:
        raise ValueError(
            f"Unknown SAE variant: position={position}, expansion={expansion}. "
            f"Valid: {list(_REPOS.keys())}"
        )
    return _REPOS[key]


def _safetensors_path(layer: int, position: str, expansion: str) -> str:
    """Return the in-repo path to the safetensors file for a given layer."""
    return f"Llama3_1-8B-Base-L{layer}{position.upper()}-{expansion}/checkpoints/final.safetensors"


def _hyperparams_path(layer: int, position: str, expansion: str) -> str:
    return f"Llama3_1-8B-Base-L{layer}{position.upper()}-{expansion}/hyperparams.json"


class LlamaScopeSAE(nn.Module):
    """Full JumpReLU Sparse Autoencoder from Llama Scope.

    Loads ~256MB for 8x expansion, ~1GB for 32x. Used by FeatureManager
    for encode-modify-decode steering (see ``steer``).
    """

    def __init__(
        self,
        d_model: int,
        d_sae: int,
        jump_relu_threshold: float,
        dataset_average_activation_norm: Optional[float] = None,
        sparsity_include_decoder_norm: bool = True,
    ):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=True)
        self.jump_relu_threshold = jump_relu_threshold
        self.sparsity_include_decoder_norm = sparsity_include_decoder_norm
        self.dataset_average_activation_norm = dataset_average_activation_norm

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.dataset_average_activation_norm is not None:
            x = x * (self.dataset_average_activation_norm / x.norm(dim=-1, keepdim=True))
        pre_acts = self.encoder(x)
        # JumpReLU: zero out activations below threshold
        return pre_acts * (pre_acts > self.jump_relu_threshold).to(pre_acts.dtype)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, encoded_features)."""
        f = self.encode(x)
        return self.decode(f), f

    @torch.no_grad()
    def steer(
        self,
        x: torch.Tensor,
        modifications: list[tuple[int, float]],
        state: Optional[dict] = None,
    ) -> torch.Tensor:
        """Encode → modify features → decode, preserving reconstruction error.

        Additive steering with a count-based cutoff.  Pass a mutable
        ``state`` dict with a ``"max_steps"`` key to limit how many
        generation steps are steered.  After ``max_steps``, the model
        continues unsteered — the KV cache already contains steered
        context, which maintains the theme naturally.

        Args:
            x: Activation tensor [..., d_model].
            modifications: List of (feature_index, scale) pairs.
            state: Mutable dict shared across generation steps.
                ``state["max_steps"]``: stop steering after this many steps.
                Without state, steering is unconditional.

        Returns:
            Modified activations with SAE reconstruction error preserved.
        """
        if state is not None:
            state["step"] = state.get("step", 0) + 1
            if state["step"] > state.get("max_steps", float("inf")):
                return x

        # Move SAE to match input device/dtype (cached after first call)
        if self.encoder.weight.device != x.device or self.encoder.weight.dtype != x.dtype:
            self.to(device=x.device, dtype=x.dtype)
        encoded = self.encode(x)
        recon = self.decode(encoded)
        error = x - recon
        for idx, scale in modifications:
            if scale > 0:
                encoded[..., idx] += scale
            else:
                encoded[..., idx] = 0
        return self.decode(encoded) + error

    @classmethod
    def from_pretrained(
        cls,
        layer: int,
        position: str = "R",
        expansion: str = "8x",
        device: str = "cpu",
    ) -> "LlamaScopeSAE":
        """Download and load a full SAE from Llama Scope.

        Args:
            layer: Transformer layer index (0-31).
            position: Hook position — "R" (residual), "M" (MLP), or "A" (attention).
            expansion: "8x" (32K features) or "32x" (131K features).
            device: Target device.
        """
        repo = _repo_id(position, expansion)
        hp_path = hf_hub_download(repo, _hyperparams_path(layer, position, expansion))
        st_path = hf_hub_download(repo, _safetensors_path(layer, position, expansion))

        with open(hp_path) as f:
            hp = json.load(f)

        norm_in = hp.get("dataset_average_activation_norm", {})
        avg_norm = norm_in.get("in") if isinstance(norm_in, dict) else norm_in

        sae = cls(
            d_model=hp["d_model"],
            d_sae=hp["d_sae"],
            jump_relu_threshold=hp["jump_relu_threshold"],
            dataset_average_activation_norm=avg_norm,
            sparsity_include_decoder_norm=hp.get("sparsity_include_decoder_norm", True),
        )

        with safe_open(st_path, framework="pt") as f:
            sae.encoder.weight.data = f.get_tensor("encoder.weight").to(device)
            sae.encoder.bias.data = f.get_tensor("encoder.bias").to(device)
            sae.decoder.weight.data = f.get_tensor("decoder.weight").to(device)
            sae.decoder.bias.data = f.get_tensor("decoder.bias").to(device)

        return sae.to(device)


# ---------------------------------------------------------------------------
# Worker-side SAE cache and steering function.
#
# steer_activations() is meant to be called INSIDE a trace so it executes
# on the vLLM worker process.  SAEs are loaded from the HF disk cache on
# first use and kept in a module-level dict — no pickling across processes.
# ---------------------------------------------------------------------------

_worker_sae_cache: dict[tuple, LlamaScopeSAE] = {}


def steer_activations(
    x: torch.Tensor,
    layer: int,
    position: str,
    expansion: str,
    modifications: list[tuple[int, float]],
    state: Optional[dict] = None,
) -> torch.Tensor:
    """Load SAE on this process (from HF cache) and steer activations.

    Intended to be called inside an nnsight trace so that the SAE is loaded
    and cached on the vLLM worker process — nothing is pickled or sent over.

    Args:
        x: Activation tensor [..., d_model].
        layer: Transformer layer index.
        position: Hook position — "R", "M", or "A".
        expansion: "8x" or "32x".
        modifications: List of (feature_index, scale) pairs.
        state: Mutable dict shared across generation steps for cutoff.

    Returns:
        Modified activations.
    """
    key = (layer, position, expansion)
    if key not in _worker_sae_cache:
        _worker_sae_cache[key] = LlamaScopeSAE.from_pretrained(
            layer, position, expansion, device=str(x.device),
        )
    sae = _worker_sae_cache[key]
    return sae.steer(x, modifications, state=state)


def ensure_downloaded(layer: int, position: str, expansion: str) -> None:
    """Ensure SAE files are in the HF cache (no-op if already downloaded).

    Call this on the main process so the first trace doesn't block on a
    network download.
    """
    repo = _repo_id(position, expansion)
    hf_hub_download(repo, _safetensors_path(layer, position, expansion))
    hf_hub_download(repo, _hyperparams_path(layer, position, expansion))


# ---------------------------------------------------------------------------
# FeatureManager — resolves feature names/specs to SAE coordinates.
# ---------------------------------------------------------------------------


class FeatureManager:
    """Resolves feature names to SAE coordinates for steering.

    Does NOT hold SAE weights — those are loaded lazily on the worker
    process via ``steer_activations()``.
    """

    def __init__(self, catalog_path: Optional[str] = None):
        if catalog_path is None:
            catalog_path = str(Path(__file__).parent / "features.json")
        with open(catalog_path) as f:
            self.catalog: dict = json.load(f)

    def list_features(self) -> list[dict]:
        """Return the feature catalog for display."""
        result = []
        for name, info in self.catalog.items():
            result.append({"name": name, **info})
        return result

    def resolve_feature(
        self, feature_spec: str
    ) -> tuple[int, str, str, int]:
        """Resolve a feature spec to SAE coordinates.

        Args:
            feature_spec: Either a curated name ("shakespeare") or
                numeric "layer:index" format (defaults to R, 8x).

        Returns:
            (layer, position, expansion, feature_index).
        """
        if feature_spec in self.catalog:
            info = self.catalog[feature_spec]
            return info["layer"], info["position"], info["expansion"], info["index"]
        elif ":" in feature_spec:
            parts = feature_spec.split(":")
            return int(parts[0]), "R", "8x", int(parts[1])
        else:
            raise ValueError(
                f"Unknown feature: {feature_spec!r}. "
                f"Use a curated name ({', '.join(self.catalog.keys())}) "
                f"or 'layer:index' format (e.g. '28:8401')."
            )
