"""Small, dependency-light artifact and manifest helpers."""
import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f'Cannot save an empty manifest: {path}')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def verify_bundle(root):
    root = Path(root)
    metadata = json.loads((root / 'metadata.json').read_text())
    if metadata.get('format_version') != 2 or metadata.get('workflow') != 'fold_first_signals':
        raise ValueError('Expected a version-fourteen fold-first signal bundle')
    for name, digest in metadata['artifact_sha256'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'Prepared artifact changed: {name}')
    return metadata
