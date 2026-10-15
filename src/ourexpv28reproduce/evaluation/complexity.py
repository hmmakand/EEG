"""Size and cost of a model, and the software and hardware of a run, for reporting.

Shared by every paper: `model_complexity` takes a function that builds the
model and one input batch. It does not draw from the random number generators
used in training.
"""
import io
from importlib.metadata import PackageNotFoundError, version
import platform
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

PACKAGES = ("torch", "numpy", "scipy", "scikit-learn", "mne")

FLOP_NOTE = ("FLOPs counted by torch.utils.flop_counter.FlopCounterMode for one trial: convolutions, matrix "
             "products and attention. Element-wise operations (activations, normalisation, pooling) are not "
             "counted. MACs = FLOPs / 2.")


def synchronize(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def inference_milliseconds(model, x, repeats: int) -> float:
    """Median time of one forward pass of `x`, in milliseconds, after 5 warm-up passes."""
    for _ in range(5):
        model(x)
    times = []
    for _ in range(repeats):
        synchronize(x.device)
        start = time.perf_counter()
        model(x)
        synchronize(x.device)
        times.append(time.perf_counter() - start)
    return 1000 * sorted(times)[len(times) // 2]


def model_complexity(build, trials, device, batch_size: int, repeats: int = 50) -> dict:
    """Parameters, FLOPs, inference latency and file size. `build()` returns the model (untrained weights; the
    cost does not depend on them); `trials` [n, ...] is a float array of prepared trials, of which the first
    and the first `batch_size` are timed."""
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        model = build().to(device).eval()
    one = torch.as_tensor(trials[:1]).to(device)
    batch = torch.as_tensor(trials[:batch_size]).to(device)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(one)
    flops = counter.get_total_flops()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    batch_ms = inference_milliseconds(model, batch, repeats)
    return dict(parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                flops_per_trial=flops, macs_per_trial=flops // 2, flop_note=FLOP_NOTE,
                model_file_bytes=buffer.getbuffer().nbytes,
                latency_ms_per_trial=inference_milliseconds(model, one, repeats),
                latency_ms_per_batch=batch_ms, latency_batch_size=len(batch),
                latency_ms_per_trial_in_batch=batch_ms / len(batch),
                latency_note=f"Median of {repeats} forward passes in evaluation mode, after 5 warm-up passes, on "
                             f"{device}; data preparation is not included.")


def environment(device) -> dict:
    """Package versions, Python, and the processor that trained the models."""
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(device=str(device),
                device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
                or "CPU",
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version() if device.type == "cuda" else None,
                python=platform.python_version(), packages=packages)
