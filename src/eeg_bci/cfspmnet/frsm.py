"""Fourier-Reorganized State Mamba (FRSM) encoder block (CFSPMNet paper, Eqs. 4-6).

Ported from ``temp/cfspmnet_frsmamba_model.py``. Only the Fourier-domain
token-state reorganization path is kept (the reference file's alternate
"conv" time-domain implementation was an ablation/comparison variant, not part
of the published method, and is out of scope for this pass -- see
``notebooks/custom_model/`` plan).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from torch import nn


class FourierRhythmicStateModeling(nn.Module):
    """Fourier-domain token-state reorganization (Eq. 4) with low/high split (Eq. 5)."""

    def __init__(
        self,
        emb_size,
        rhythm_branch_mode="full",
        num_blocks=None,
        sparsity_threshold=0.01,
        rhythm_low_ratio=0.45,
    ):
        super().__init__()
        if rhythm_branch_mode not in {"full", "low_only", "high_only"}:
            raise ValueError(f"Unsupported rhythm_branch_mode: {rhythm_branch_mode}")
        self.rhythm_branch_mode = rhythm_branch_mode
        self.emb_size = emb_size
        self.sparsity_threshold = sparsity_threshold
        self.rhythm_low_ratio = rhythm_low_ratio
        self.num_blocks = self._resolve_num_blocks(emb_size, preferred=num_blocks)
        self.block_size = emb_size // self.num_blocks
        self.scale = 0.02

        self.w = nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size, self.block_size, 2))
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, 1))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, 1))
        self.b = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.rhythm_filter = nn.Parameter(torch.ones(1, emb_size, 1))
        self.context_proj = nn.Sequential(
            nn.Conv1d(emb_size * 2, emb_size * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_size * 2),
            nn.SiLU(),
            nn.Conv1d(emb_size * 2, emb_size, kernel_size=1, bias=False),
        )
        self.context_norm = nn.LayerNorm(emb_size)

    @staticmethod
    def _resolve_num_blocks(channels: int, preferred=None) -> int:
        if preferred is not None:
            candidates = [preferred]
        else:
            candidates = [8, 6, 5, 4, 3, 2, 1]
        for candidate in candidates:
            if channels % candidate == 0:
                return candidate
        return 1

    def _fourier_mix(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = x_seq.dtype
        x_spectrum = torch.fft.rfft(x_seq.float(), dim=-1, norm="ortho")
        origin_spectrum = x_spectrum
        batch_size, channels, spectrum_bins = x_spectrum.shape
        x_spectrum = x_spectrum.reshape(batch_size, self.num_blocks, self.block_size, spectrum_bins)

        weight = torch.view_as_complex(self.w.contiguous())
        mixed = torch.einsum("bkif,kio->bkof", x_spectrum, weight)
        o_real = F.silu(
            mixed.real * self.w1[0].unsqueeze(0)
            - mixed.imag * self.w1[1].unsqueeze(0)
            + self.b[0, :, :, None]
        )
        o_imag = F.silu(
            mixed.imag * self.w2[0].unsqueeze(0)
            + mixed.real * self.w2[1].unsqueeze(0)
            + self.b[1, :, :, None]
        )
        mixed = torch.stack([o_real, o_imag], dim=-1)
        mixed = F.softshrink(mixed, lambd=self.sparsity_threshold)
        mixed = torch.view_as_complex(mixed).reshape(batch_size, channels, spectrum_bins)
        mixed = mixed * self.rhythm_filter + origin_spectrum
        enhanced = torch.fft.irfft(mixed, n=x_seq.shape[-1], dim=-1, norm="ortho").type(dtype)
        return enhanced + x_seq, mixed

    def _split_low_high(self, mixed_spectrum: torch.Tensor, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        spectrum_bins = mixed_spectrum.shape[-1]
        low_bins = max(1, min(spectrum_bins - 1, int(round(spectrum_bins * self.rhythm_low_ratio))))
        low_mask = torch.zeros_like(mixed_spectrum)
        low_mask[..., :low_bins] = 1.0
        high_mask = 1.0 - low_mask
        low = torch.fft.irfft(mixed_spectrum * low_mask, n=sequence_length, dim=-1, norm="ortho")
        high = torch.fft.irfft(mixed_spectrum * high_mask, n=sequence_length, dim=-1, norm="ortho")
        return low, high

    def forward(self, x):
        x_seq = x.transpose(1, 2)
        enhanced, mixed_spectrum = self._fourier_mix(x_seq)
        low_fused, high_fused = self._split_low_high(mixed_spectrum, x_seq.shape[-1])
        low_fused = low_fused + x_seq
        high_fused = high_fused + x_seq

        if self.rhythm_branch_mode == "low_only":
            low_used, high_used, residual = low_fused, torch.zeros_like(high_fused), low_fused
        elif self.rhythm_branch_mode == "high_only":
            low_used, high_used, residual = torch.zeros_like(low_fused), high_fused, high_fused
        else:
            low_used, high_used, residual = low_fused, high_fused, enhanced

        context = self.context_proj(torch.cat([low_used, high_used], dim=1))
        context = context + residual
        context = self.context_norm(context.transpose(1, 2))
        return low_fused.transpose(1, 2), high_fused.transpose(1, 2), context


class FourierRhythmicStateModulator(nn.Module):
    """Turns the Fourier-derived context into the Mamba conditioning terms S, B (Eq. 5)."""

    def __init__(self, d_model, d_inner):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.token_modulator = nn.Linear(d_model, d_inner)
        self.residual_modulator = nn.Linear(d_model, d_inner)

    def forward(self, rhythmic_context):
        context = self.norm(rhythmic_context)
        token_scale = torch.sigmoid(self.token_modulator(context))
        residual_bias = self.residual_modulator(context)
        return token_scale, residual_bias


class FRSMambaStateSpaceMixer(nn.Module):
    """Selective-scan (Mamba) branch, context-conditioned per Eq. 6."""

    def __init__(self, input_channels, use_fourier_rhythmic_modeling=True):
        super().__init__()
        self.d_model = input_channels
        self.d_inner = self.d_model * 2
        self.dt_rank = math.ceil(self.d_model / 16)
        self.d_state = 16
        self.use_fourier_rhythmic_modeling = use_fourier_rhythmic_modeling

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            groups=self.d_inner,
            padding=2,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        a = repeat(torch.arange(1, self.d_state + 1), "n -> d n", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model)
        self.fourier_rhythmic_modulator = (
            FourierRhythmicStateModulator(self.d_model, self.d_inner) if use_fourier_rhythmic_modeling else None
        )

    def ssm(self, x):
        _, n = self.A_log.shape
        a = -torch.exp(self.A_log.float())
        d = self.D.float()
        x_dbl = self.x_proj(x)
        delta, b, c = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        return self.selective_scan(x, delta, a, b, c, d)

    def selective_scan(self, u, delta, a, b, c, d):
        batch_size, sequence_length, inner_dim = u.shape
        state_dim = a.shape[1]
        delta_a = torch.exp(einsum(delta, a, "b l d, d n -> b l d n"))
        delta_b_u = einsum(delta, b, u, "b l d, b l n, b l d -> b l d n")

        state = torch.zeros((batch_size, inner_dim, state_dim), device=delta_a.device, dtype=delta_a.dtype)
        outputs = []
        for index in range(sequence_length):
            state = delta_a[:, index] * state + delta_b_u[:, index]
            y = einsum(state, c[:, index, :], "b d n, b n -> b d")
            outputs.append(y)
        y = torch.stack(outputs, dim=1)
        return y + u * d

    def forward(self, x, rhythmic_context=None, return_details=False):
        _, sequence_length, _ = x.shape
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        token_scale = None
        residual_bias = None
        if self.use_fourier_rhythmic_modeling and rhythmic_context is not None:
            assert self.fourier_rhythmic_modulator is not None
            token_scale, residual_bias = self.fourier_rhythmic_modulator(rhythmic_context)
            x = x * (1.0 + token_scale)
            res = res + residual_bias

        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :sequence_length]
        x = rearrange(x, "b d l -> b l d")

        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        out = self.out_proj(y)
        if not return_details:
            return out
        return out, {"token_scale": token_scale, "residual_bias": residual_bias}


class FRSMambaBlock(nn.Module):
    """One FRSM encoder block: Fourier reorganization + context-conditioned Mamba mixing."""

    def __init__(
        self,
        emb_size,
        drop_p=0.3,
        use_fourier_rhythmic_modeling=True,
        rhythm_branch_mode="full",
        rhythm_num_blocks=None,
        rhythm_sparsity_threshold=0.01,
        rhythm_low_ratio=0.45,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.use_fourier_rhythmic_modeling = use_fourier_rhythmic_modeling
        self.rhythmic_state_modeling = (
            FourierRhythmicStateModeling(
                emb_size,
                rhythm_branch_mode=rhythm_branch_mode,
                num_blocks=rhythm_num_blocks,
                sparsity_threshold=rhythm_sparsity_threshold,
                rhythm_low_ratio=rhythm_low_ratio,
            )
            if use_fourier_rhythmic_modeling
            else None
        )
        self.temporal_mixer = FRSMambaStateSpaceMixer(
            input_channels=emb_size, use_fourier_rhythmic_modeling=use_fourier_rhythmic_modeling
        )
        self.drop = nn.Dropout(drop_p)

    def forward(self, x, return_details=False):
        x_norm = self.norm(x)
        rhythmic_context = None
        low_fused = None
        high_fused = None

        if self.use_fourier_rhythmic_modeling:
            assert self.rhythmic_state_modeling is not None
            low_fused, high_fused, rhythmic_context = self.rhythmic_state_modeling(x_norm)

        if return_details:
            out, mixer_details = self.temporal_mixer(x_norm, rhythmic_context, return_details=True)
        else:
            out = self.temporal_mixer(x_norm, rhythmic_context)
            mixer_details = None

        residual_out = x + self.drop(out)
        if not return_details:
            return residual_out
        return residual_out, {
            "low_fused": low_fused,
            "high_fused": high_fused,
            "rhythmic_context": rhythmic_context,
            "token_scale": None if mixer_details is None else mixer_details["token_scale"],
            "residual_bias": None if mixer_details is None else mixer_details["residual_bias"],
        }
