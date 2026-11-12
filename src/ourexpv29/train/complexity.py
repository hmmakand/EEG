"""Size and cost of the model, and the features it computes, for reporting.

model_complexity: parameters, floating-point operations, inference time and
file size of one model, measured once per run on a test sample.
embeddings: the inputs of the two output layers (graph head and temporal
encoder) for the test samples, for a t-SNE plot.

None of this draws from the random number generators used in training.
"""
import io
import time

import torch
from torch.utils.flop_counter import FlopCounterMode
from torch_geometric.loader import DataLoader

from ..model import EEGGraphModel
from .engine import predict

FLOP_NOTE = ("FLOPs counted by torch.utils.flop_counter.FlopCounterMode for one sample: matrix products, "
             "convolutions and einsum. The scatter and index_add operations of the graph layers and the "
             "element-wise operations are not counted, so the figure is a lower bound. MACs = FLOPs / 2.")


def synchronize(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def inference_milliseconds(model, data, device, repeats: int) -> float:
    """Median time of one forward pass of `data`, in milliseconds."""
    for _ in range(5):                                   # warm-up
        predict(model, data)
    times = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        predict(model, data)
        synchronize(device)
        times.append(time.perf_counter() - start)
    return 1000 * sorted(times)[len(times) // 2]


def model_complexity(model_config, graphs, device, batch_size: int, repeats: int = 50) -> dict:
    """Parameters, FLOPs, inference latency and file size of one model (untrained weights; the cost does not
    depend on them). `graphs`: test samples; the first one, and the first batch of `batch_size`, are timed."""
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        model = EEGGraphModel(model_config).to(device).eval()
    one = next(iter(DataLoader(graphs[:1], batch_size=1))).to(device)
    batch = next(iter(DataLoader(graphs[:batch_size], batch_size=batch_size))).to(device)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        predict(model, one)
    flops = counter.get_total_flops()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    parameters = sum(p.numel() for p in model.parameters())
    per_path = {name: sum(p.numel() for p in module.parameters())
                for name, module in dict(temporal=model.temporal, wide=model.wide).items()}
    per_path["graph"] = parameters - sum(per_path.values())
    batch_ms = inference_milliseconds(model, batch, device, repeats)
    return dict(parameters=parameters, trainable_parameters=sum(p.numel() for p in model.parameters()
                                                                if p.requires_grad),
                parameters_per_path=per_path, flops_per_sample=flops, macs_per_sample=flops // 2,
                flop_note=FLOP_NOTE, model_file_bytes=buffer.getbuffer().nbytes,
                latency_ms_per_sample=inference_milliseconds(model, one, device, repeats),
                latency_ms_per_batch=batch_ms, latency_batch_size=batch.num_graphs,
                latency_ms_per_sample_in_batch=batch_ms / batch.num_graphs,
                latency_note=f"Median of {repeats} forward passes in evaluation mode, after 5 warm-up passes, "
                             f"on {device}; graph building and filtering are not included. Ensemble of k models: "
                             f"k times the cost.")


@torch.no_grad()
def embeddings(model, batches, device) -> dict:
    """The input of the graph head's MLP (pooled nodes and pairs) and of the temporal encoder's output layer
    for every test sample, in the order of the batches; a path that the model does not have is left out."""
    model.eval()
    captured = {}
    hooks = []
    layers = dict(graph=model.head, temporal=model.temporal.out)
    for name, layer in layers.items():
        captured[name] = []
        hooks.append(layer.register_forward_hook(lambda module, inputs, output, name=name:
                                                 captured[name].append(inputs[0].cpu())))
    try:
        for data in batches:
            predict(model, data.to(device))
    finally:
        for hook in hooks:
            hook.remove()
    return {name: torch.cat(values).numpy() for name, values in captured.items() if values}
