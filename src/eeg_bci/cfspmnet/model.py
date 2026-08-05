"""CFSPMNet: physiological tokenizer + FRSM encoder + prediction head.

Ported from ``temp/cfspmnet_frsmamba_model.py``. Default hyperparameters come
from the CFSPMNet paper's XW-Stroke column (Table 2) since Liu2024 *is*
XW-Stroke (see ``src/eeg_bci/cfspmnet/__init__.py``).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from eeg_bci.cfspmnet.frsm import FRSMambaBlock
from eeg_bci.cfspmnet.tokenizer import LearnablePositionEncoding, MultiScalePhysiologicalTokenizer
from eeg_bci.data.types import DatasetInfo


class PredictionHead(nn.Module):
    def __init__(self, flatten_number, n_classes):
        super().__init__()
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(flatten_number, n_classes))

    def forward(self, x):
        return self.fc(x)


class CFSPMNet(nn.Module):
    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: int,
        emb_size: int = 30,
        depth: int = 2,
        eeg_f1: int = 8,
        eeg_D: int = 3,
        eeg_pooling_size1: int = 8,
        eeg_pooling_size2: int = 8,
        eeg_dropout_rate: float = 0.3,
        temporal_kernel_sizes: tuple[int, int, int] = (36, 24, 18),
        fusion_kernel_size: int = 16,
        use_fourier_rhythmic_modeling: bool = True,
        rhythm_branch_mode: str = "full",
        rhythm_num_blocks: int | None = None,
        rhythm_sparsity_threshold: float = 0.01,
        rhythm_low_ratio: float = 0.45,
    ):
        super().__init__()
        self.emb_size = emb_size

        self.physiological_tokenizer = MultiScalePhysiologicalTokenizer(
            f1=eeg_f1,
            D=eeg_D,
            pooling_size1=eeg_pooling_size1,
            pooling_size2=eeg_pooling_size2,
            dropout_rate=eeg_dropout_rate,
            number_channel=n_chans,
            emb_size=emb_size,
            temporal_kernel_sizes=temporal_kernel_sizes,
            fusion_kernel_size=fusion_kernel_size,
        )
        self.position_encoding = LearnablePositionEncoding(emb_size, dropout=0.1)
        self.temporal_encoder = nn.Sequential(
            *[
                FRSMambaBlock(
                    emb_size,
                    drop_p=eeg_dropout_rate,
                    use_fourier_rhythmic_modeling=use_fourier_rhythmic_modeling,
                    rhythm_branch_mode=rhythm_branch_mode,
                    rhythm_num_blocks=rhythm_num_blocks,
                    rhythm_sparsity_threshold=rhythm_sparsity_threshold,
                    rhythm_low_ratio=rhythm_low_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(emb_size)
        self.flatten = nn.Flatten()

        flatten_size = self._infer_flatten_size(n_chans, n_times)
        self.prediction_head = PredictionHead(flatten_size, n_outputs)

    def _infer_flatten_size(self, n_chans: int, n_times: int) -> int:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_chans, n_times)
            features = self._encode(dummy)
            flatten_size = int(self.flatten(features).shape[1])
        self.train(was_training)
        return flatten_size

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.physiological_tokenizer(x)
        scaled_tokens = tokens * math.sqrt(self.emb_size)
        hidden = self.position_encoding(scaled_tokens)
        hidden = self.temporal_encoder(hidden)
        return self.final_norm(hidden)

    def forward_features(self, x: torch.Tensor, return_block_details: bool = False) -> dict:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        physiological_tokens = self.physiological_tokenizer(x)
        scaled_tokens = physiological_tokens * math.sqrt(self.emb_size)
        positioned_tokens = self.position_encoding(scaled_tokens)

        hidden = positioned_tokens
        block_details = []
        for block in self.temporal_encoder:
            if return_block_details:
                hidden, details = block(hidden, return_details=True)
                block_details.append(details)
            else:
                hidden = block(hidden)

        features = self.final_norm(hidden)
        flatten_features = self.flatten(features)
        logits = self.prediction_head(flatten_features)

        export = {
            "physiological_tokens": physiological_tokens,
            "positioned_tokens": positioned_tokens,
            "encoder_features": features,
            "flatten_features": flatten_features,
            "logits": logits,
        }
        if return_block_details and block_details:
            export["encoder_block_details"] = block_details
            export.update(block_details[-1])
        return export

    def forward(self, x, return_features: bool = False, return_block_details: bool = False):
        export = self.forward_features(x, return_block_details=return_block_details)
        if return_features:
            return export
        return export["logits"]

    @classmethod
    def from_dataset_info(cls, dataset_info: DatasetInfo, **hparams) -> "CFSPMNet":
        return cls(
            n_chans=dataset_info.n_chans,
            n_outputs=dataset_info.n_outputs,
            n_times=dataset_info.n_times,
            **hparams,
        )


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
