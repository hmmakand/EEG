import torch
import torch.nn as nn


class KernelPENodeEncoder(torch.nn.Module):
    """Configurable kernel-based Positional Encoding node encoder.

    Reads the precomputed statistics `batch.pestat_<kernel_type>`
    [N, len(kernel_times)], encodes them to dim_pe and appends them to x.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with PE.

    Args:
        dim_in: Input node feature dim (only used if expand_x)
        dim_emb: Size of final node embedding
        pecfg: PosEncConfig (cfg.posenc_<kernel_type>)
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    kernel_type = None  # Instantiated type of the KernelPE, e.g. RWSE

    def __init__(self, dim_in, dim_emb, pecfg, expand_x=True):
        super().__init__()
        if self.kernel_type is None:
            raise ValueError(f"{self.__class__.__name__} has to be "
                             f"preconfigured by setting 'kernel_type' class"
                             f"variable before calling the constructor.")

        dim_pe = pecfg.dim_pe  # Size of the kernel-based PE embedding
        num_rw_steps = len(pecfg.kernel_times)
        model_type = pecfg.model.lower()  # Encoder NN model type for PEs
        n_layers = pecfg.layers  # Num. layers in PE encoder model
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type
        self.pass_as_var = pecfg.pass_as_var

        if dim_emb - dim_pe < 0:
            raise ValueError(f"PE dim size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(num_rw_steps)
        else:
            self.raw_norm = None

        activation = nn.ReLU
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(num_rw_steps, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(num_rw_steps, 2 * dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(num_rw_steps, dim_pe)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, batch):
        pestat_var = f"pestat_{self.kernel_type}"
        if not hasattr(batch, pestat_var):
            raise ValueError(f"Precomputed '{pestat_var}' variable is "
                             f"required for {self.__class__.__name__}")

        pos_enc = getattr(batch, pestat_var)  # N x num kernel times
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)  # N x dim_pe

        h = self.linear_x(batch.x) if self.expand_x else batch.x
        batch.x = torch.cat((h, pos_enc), 1)
        if self.pass_as_var:
            setattr(batch, f'pe_{self.kernel_type}', pos_enc)
        return batch


class RWSENodeEncoder(KernelPENodeEncoder):
    """Random Walk Structural Encoding node encoder."""
    kernel_type = 'RWSE'
