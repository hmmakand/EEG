"""Save every eligible original EEG trial without permanent split assignments."""
from dataclasses import asdict
import numpy as np
from .extraction import extract_original
from .storage import write_json, write_rows, sha256


def save_originals(data_dir, output, config):
    output.mkdir(parents=True, exist_ok=True)
    all_rows, exclusions = [], []
    counts = {}
    for subject in config.subjects:
        source = data_dir / f'A{subject:02d}T.mat'
        signals, rows, omitted = extract_original(source, subject, config)
        labels = np.asarray([r['class_label'] for r in rows], dtype=np.int64)
        if set(labels.tolist()) != {0, 1} or not np.isfinite(signals).all():
            raise ValueError(f'Subject {subject}: expected finite EEG with both classes')
        name = f'subject_{subject:02d}.npz'
        for row in rows:
            row.pop('split_seed')
            row['trial_id'] = f'original:S{subject:02d}:{row["session_id"]}:{row["original_trial_index"]}'
            row['signal_file'] = f'original/{name}'
        np.savez_compressed(output / name, signals=signals, labels=labels,
                            subject_ids=np.full(len(rows), subject, dtype=np.int64),
                            session_ids=np.array([r['session_id'] for r in rows]),
                            original_trial_indices=np.array([r['original_trial_index'] for r in rows]),
                            trial_ids=np.array([r['trial_id'] for r in rows]))
        all_rows.extend(rows)
        exclusions.extend(omitted)
        counts[str(subject)] = {str(c): int(sum(labels == c)) for c in (0, 1)}
    write_rows(output / 'trials.csv', all_rows)
    write_json(output / 'metadata.json', dict(
        format_version=2, extraction=asdict(config.trials), sample_rate=config.sample_rate,
        counts=counts, exclusions=exclusions, signal_axes=['trial', 'sample', 'channel'],
        processing='original EEG extraction only; no filtering, scaling, graphs, or fixed partitions',
        artifact_sha256={p.name: sha256(p) for p in output.iterdir() if p.is_file()}))
    return all_rows
