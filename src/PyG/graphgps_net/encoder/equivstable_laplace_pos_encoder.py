import torch
import torch.nn as nn


class EquivStableLapPENodeEncoder(torch.nn.Module):
    """Equivariant and Stable Laplace Positional Embedding node encoder.

    Transforms the k-dim node LapPE to d-dim; it is NOT concatenated to x but
    stored in `batch.pe_EquivStableLapPE` and used inside the local MPNN
    (GatedGCN / GINE) to reweight messages by PE distance.
    Paper: https://openreview.net/pdf?id=e95i1IHcWj

    Args:
        dim_emb: Size of the PE embedding (= hidden dim)
        pecfg: PosEncConfig (cfg.posenc_EquivStableLapPE)
    """

    def __init__(self, dim_emb, pecfg):
        super().__init__()
        max_freqs = pecfg.max_freqs
        norm_type = pecfg.raw_norm_type.lower()

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(max_freqs)
        else:
            self.raw_norm = None

        self.linear_encoder_eigenvec = nn.Linear(max_freqs, dim_emb)

    def forward(self, batch):
        if not (hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs')):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}")
        pos_enc = batch.EigVecs

        empty_mask = torch.isnan(pos_enc)  # N x k
        pos_enc[empty_mask] = 0.

        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)

        pos_enc = self.linear_encoder_eigenvec(pos_enc)

        batch.pe_EquivStableLapPE = pos_enc
        return batch
