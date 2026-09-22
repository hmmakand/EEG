"""Write deterministic NumPy arrays and sample metadata from subject checkpoints."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def save_dataset(output_dir, checkpoints, edge_index, metadata_builder):
    """Assemble ordered subject checkpoints into memory-mappable dataset files.

    Parameters
    ----------
    output_dir : path-like
        Empty staging directory to create. Existing contents are rejected.
    checkpoints : sequence of path-like
        NPZ files in subject order. Each has two node arrays, six edge arrays,
        labels and a JSON sidecar with sample records, montage and provenance.
    edge_index : ndarray, shape (2, 812)
        Shared zero-based electrode connections.
    metadata_builder : callable
        Called with (samples, subject_records, montage, array_schema); returns
        a JSON-compatible dataset metadata dictionary.

    Returns
    -------
    dict
        The metadata written to metadata.json.

    Notes
    -----
    Writes to a staging directory. The caller validates it before publishing.
    Input checkpoints are read without modification.
    """
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f'Staging directory must be empty: {destination}')
    destination.mkdir(parents=True, exist_ok=True)
    sidecars = [json.loads(Path(path).with_suffix('.json').read_text()) for path in checkpoints]
    samples = []
    for sidecar in sidecars:
        for record in sidecar['samples']:
            samples.append({**record, 'sample_index': len(samples)})
    count = len(samples)
    specs: dict[str, tuple[tuple[int, ...], str]] = {
        f'node_features_{variant}': ((count, 29, len(config.NODE_FEATURE_NAMES)), 'float32') for variant in config.NODE_VARIANTS
    }
    specs.update({f'edge_attr_{method}_{variant}': ((count, 812, 5), 'float32')
                  for variant in config.NODE_VARIANTS for method in config.METHODS})
    specs['labels'] = ((count,), 'int64')
    arrays = {}
    for key, (shape, dtype) in specs.items():
        mapped = np.lib.format.open_memmap(destination / f'{key}.npy', mode='w+', dtype=dtype, shape=shape)
        offset = 0
        for path in checkpoints:
            with np.load(path, allow_pickle=False) as checkpoint:
                values = checkpoint[key]
                mapped[offset:offset + len(values)] = values
                offset += len(values)
        if offset != count:
            raise ValueError(f'{key}: checkpoint row count does not match samples')
        mapped.flush()
        del mapped
        arrays[f'{key}.npy'] = {'shape': list(shape), 'dtype': dtype}
    np.save(destination / 'edge_index.npy', np.asarray(edge_index, dtype=np.int64), allow_pickle=False)
    arrays['edge_index.npy'] = {'shape': [2, 812], 'dtype': 'int64'}
    pd.DataFrame(samples).to_csv(destination / 'samples.tsv', sep='\t', index=False)
    metadata = metadata_builder(samples, [row['provenance'] for row in sidecars], sidecars[0]['montage'], arrays)
    (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    return metadata
