"""EEGNet configured for the preprocessed Liu2024 motor-imagery windows."""

import torch
from braindecode.models import EEGNet as BraindecodeEEGNet


class EEGNet(BraindecodeEEGNet):
    """Two-class EEGNet accepting tensors shaped ``(batch, 29, 2000)``."""

    def __init__(self) -> None:
        super().__init__(
            n_chans=29,
            n_outputs=2,
            n_times=2000,
            sfreq=500.0,
            F1=8,
            D=2,
            F2=16,
            kernel_length=250,
            depthwise_kernel_length=64,
            drop_prob=0.25,
            final_conv_length="auto",
        )


if __name__ == "__main__":
    model = EEGNet()
    dummy_batch = torch.zeros(2, 29, 2000)
    model.eval()
    with torch.inference_mode():
        logits = model(dummy_batch)

    print(model)
    print(f"\nTrainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Dummy input shape: {tuple(dummy_batch.shape)}")
    print(f"Output logits shape: {tuple(logits.shape)}")
