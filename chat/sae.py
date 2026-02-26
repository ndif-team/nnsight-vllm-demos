"""Llama Scope SAE loading and lightweight feature direction manager.

LlamaScopeSAE — Full JumpReLU SAE (for analysis/education, not used by steering).
FeatureManager — Extracts single decoder columns (~8KB each) for steering.
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

    This loads the complete SAE weights (~256MB for 8x, ~1GB for 32x).
    For steering you only need a single decoder column — use FeatureManager instead.
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
        return pre_acts * (pre_acts > self.jump_relu_threshold).float()

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, encoded_features)."""
        f = self.encode(x)
        return self.decode(f), f

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


class FeatureManager:
    """Lightweight manager that extracts single decoder columns for steering.

    The steering formula `decode(encode(x) + delta) + error` simplifies to
    `x + scale * decoder_column[feature_index]` because decode is linear and
    the error term cancels. So we only need one decoder column (~8KB per feature),
    not the full SAE.
    """

    def __init__(self, catalog_path: Optional[str] = None):
        if catalog_path is None:
            catalog_path = str(Path(__file__).parent / "features.json")
        with open(catalog_path) as f:
            self.catalog: dict = json.load(f)
        # Cache: (layer, pos, exp, idx) -> Tensor on CPU
        self._direction_cache: dict[tuple, torch.Tensor] = {}

    def list_features(self) -> list[dict]:
        """Return the feature catalog for display."""
        result = []
        for name, info in self.catalog.items():
            result.append({"name": name, **info})
        return result

    def get_direction(
        self, feature_spec: str
    ) -> tuple[torch.Tensor, int, str]:
        """Get a steering direction vector.

        Args:
            feature_spec: Either a curated name ("shakespeare") or
                numeric "layer:index" format (defaults to R, 8x).

        Returns:
            (direction_tensor, layer, position) where direction is on CPU.
        """
        if feature_spec in self.catalog:
            info = self.catalog[feature_spec]
            layer = info["layer"]
            pos = info["position"]
            exp = info["expansion"]
            idx = info["index"]
        elif ":" in feature_spec:
            parts = feature_spec.split(":")
            layer = int(parts[0])
            idx = int(parts[1])
            pos = "R"
            exp = "8x"
        else:
            raise ValueError(
                f"Unknown feature: {feature_spec!r}. "
                f"Use a curated name ({', '.join(self.catalog.keys())}) "
                f"or 'layer:index' format (e.g. '28:8401')."
            )

        direction = self._download_direction(layer, pos, exp, idx)
        return direction, layer, pos

    def _download_direction(
        self, layer: int, pos: str, exp: str, idx: int
    ) -> torch.Tensor:
        """Download a single decoder column from Llama Scope.

        Uses safetensors slicing to extract just one column (~8KB) without
        loading the full weight matrix into memory. The safetensors file is
        cached on disk by hf_hub_download.
        """
        cache_key = (layer, pos, exp, idx)
        if cache_key in self._direction_cache:
            return self._direction_cache[cache_key]

        repo = _repo_id(pos, exp)
        st_path = hf_hub_download(repo, _safetensors_path(layer, pos, exp))

        with safe_open(st_path, framework="pt") as f:
            # decoder.weight shape: [d_model, d_sae]
            direction = f.get_slice("decoder.weight")[:, idx]

        self._direction_cache[cache_key] = direction
        return direction
