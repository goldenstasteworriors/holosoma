"""Finite Scalar Quantization (FSQ).

SONIC uses a quantized universal token space (Sec. 3.2). This file implements
Finite Scalar Quantization following:

Finite Scalar Quantization: VQ-VAE Made Simple (Mentzer et al., 2023)

Implementation is adapted from the reference implementation bundled in this
repo at `reference/FSQ-pytorch/quantizers/fsq.py`, but rewritten to avoid extra
dependencies (einops) and to match Holosoma's quantizer interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from holosoma.utils.safe_torch_import import torch


@dataclass
class QuantizerOutputs:
    """Quantizer results."""

    z_q: torch.Tensor
    tokens: torch.Tensor
    aux: Optional[dict[str, torch.Tensor]] = None


class FSQQuantizer(torch.nn.Module):
    """Finite Scalar Quantizer (placeholder).

    Planned interface:
    - forward(z) returns quantized latent z_q and discrete tokens.
    """

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> QuantizerOutputs:
        raise NotImplementedError


def _round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with straight-through gradients."""

    zhat = torch.round(z)
    return z + (zhat - z).detach()


class FiniteScalarQuantizer(FSQQuantizer):
    """Finite Scalar Quantization (FSQ) with optional in/out projections.

    This mirrors `reference/FSQ-pytorch/quantizers/fsq.py` but is simplified for
    Holosoma's use (primarily 2D latents shaped (B, D)).

    Args:
        levels: Per-dimension quantization levels, e.g. [8, 5, 5, 5].
        dim: Input/output latent dimension. If it differs from
            `num_codebooks * len(levels)`, linear projections are used.
        num_codebooks: Number of independent codebooks (concatenated).
        keep_num_codebooks_dim: Whether to keep an explicit codebook dimension
            on the returned indices.
        eps: Epsilon used in the bounding transform.
    """

    def __init__(
        self,
        *,
        levels: list[int],
        dim: int,
        num_codebooks: int = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        eps: float = 1e-3,
    ):
        super().__init__()
        if not isinstance(levels, (list, tuple)) or len(levels) == 0:
            raise ValueError("levels must be a non-empty list")
        if any(int(l) < 2 for l in levels):
            raise ValueError("all levels must be >= 2")
        if num_codebooks < 1:
            raise ValueError("num_codebooks must be >= 1")

        self.codebook_levels = [int(l) for l in levels]
        self.codebook_dim = len(self.codebook_levels)
        self.num_codebooks = int(num_codebooks)
        self.effective_codebook_dim = self.codebook_dim * self.num_codebooks
        self.dim = int(dim)
        self.eps = float(eps)

        if keep_num_codebooks_dim is None:
            keep_num_codebooks_dim = self.num_codebooks > 1
        if self.num_codebooks > 1 and not keep_num_codebooks_dim:
            raise ValueError("num_codebooks > 1 requires keep_num_codebooks_dim=True")
        self.keep_num_codebooks_dim = bool(keep_num_codebooks_dim)

        levels_t = torch.tensor(self.codebook_levels, dtype=torch.int32)
        self.register_buffer("_levels", levels_t, persistent=False)

        basis = torch.cumprod(
            torch.tensor([1] + self.codebook_levels[:-1], dtype=torch.int32),
            dim=0,
        )
        self.register_buffer("_basis", basis, persistent=False)

        has_projections = self.dim != self.effective_codebook_dim
        self.project_in = (
            torch.nn.Linear(self.dim, self.effective_codebook_dim)
            if has_projections
            else torch.nn.Identity()
        )
        self.project_out = (
            torch.nn.Linear(self.effective_codebook_dim, self.dim)
            if has_projections
            else torch.nn.Identity()
        )

    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        # Matches reference implementation.
        half_l = (self._levels - 1) * (1 - self.eps) / 2
        offset = torch.where(self._levels % 2 == 0, torch.tensor(0.5, device=z.device), torch.tensor(0.0, device=z.device))
        shift = torch.tan(offset / half_l)
        return (z + shift).tanh() * half_l - offset

    def _quantize_codes(self, z: torch.Tensor) -> torch.Tensor:
        # z: (..., codebook_dim)
        quantized = _round_ste(self._bound(z))
        half_width = self._levels // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Convert normalized codes (..., codebook_dim) to integer indices."""

        if codes.shape[-1] != self.codebook_dim:
            raise ValueError(f"codes last dim must be {self.codebook_dim}, got {codes.shape[-1]}")
        codes_non_centered = self._scale_and_shift(codes)
        return (codes_non_centered.to(torch.int32) * self._basis).sum(dim=-1).to(torch.int32)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Inverse of `codes_to_indices` (returns normalized codes)."""

        indices = indices.to(torch.int32)
        idx = indices.unsqueeze(-1)
        codes_non_centered = (idx // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)
        return codes

    def forward(self, z: torch.Tensor) -> QuantizerOutputs:
        if z.shape[-1] != self.dim:
            raise ValueError(f"expected latent dim {self.dim}, got {z.shape[-1]}")

        z_proj = self.project_in(z)
        orig_shape = z_proj.shape[:-1]

        z_flat = z_proj.reshape(-1, self.effective_codebook_dim)
        z_flat = z_flat.reshape(-1, self.num_codebooks, self.codebook_dim)

        codes = self._quantize_codes(z_flat)
        indices = self.codes_to_indices(codes)

        codes_flat = codes.reshape(-1, self.effective_codebook_dim)
        out = self.project_out(codes_flat)
        out = out.reshape(*orig_shape, self.dim)

        indices = indices.reshape(*orig_shape, self.num_codebooks)
        if not self.keep_num_codebooks_dim:
            indices = indices.squeeze(-1)

        aux = {
            "fsq_indices": indices,
        }
        return QuantizerOutputs(z_q=out, tokens=out, aux=aux)


class UniformFSQQuantizer(FSQQuantizer):
    """Legacy uniform scalar quantizer.

    Kept for ablations/backward compatibility; prefer `FiniteScalarQuantizer`.
    """

    def __init__(self, *, levels: int, clip_range: float = 1.0):
        super().__init__()
        if levels < 2:
            raise ValueError("levels must be >= 2")
        self.levels = int(levels)
        self.clip_range = float(clip_range)

    def forward(self, z: torch.Tensor) -> QuantizerOutputs:
        z_clipped = torch.clamp(z, -self.clip_range, self.clip_range)
        scale = (self.levels - 1) / (2 * self.clip_range)
        z_scaled = (z_clipped + self.clip_range) * scale
        q_idx = torch.round(z_scaled)
        q_idx = torch.clamp(q_idx, 0, self.levels - 1)
        z_q = (q_idx / scale) - self.clip_range
        z_q_st = z + (z_q - z).detach()
        aux = {"fsq_indices": q_idx}
        return QuantizerOutputs(z_q=z_q_st, tokens=z_q_st, aux=aux)


class IdentityQuantizer(FSQQuantizer):
    """Passthrough quantizer for v0 integration.

    Returns continuous "tokens" equal to z.
    """

    def forward(self, z: torch.Tensor) -> QuantizerOutputs:
        return QuantizerOutputs(z_q=z, tokens=z, aux=None)
