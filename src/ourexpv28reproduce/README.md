# Version twenty-eight: reproducing published experiments

Each paper gets its own folder under `papers/`, with its data preparation,
model, training and settings, and a README that sets out what the paper says,
what was missing, what was assumed, and the reported against the reproduced
result. The file readers in `datascripts/` and the evaluation are shared.

| Paper | Folder | Status |
|---|---|---|
| PA-TCNet (Wang et al. 2026) | [papers/patcnet](papers/patcnet/README.md) | Ready to run; no full run yet |

## Structure

```
datascripts/        file readers, shared: Liu 2024, BCI IV 2a, Jia 2019
                    (subjects, trials, left/right labels, 250 Hz; copied from version 27)
papers/<paper>/     one folder per paper
evaluation/         shared: metrics.py (metrics), complexity.py (parameters, FLOPs, inference time,
                    software and hardware), figures.py (ROC curves, training curves, model table)
run.py              python -m src.ourexpv28reproduce.run --paper patcnet --dataset jia2019 --data-dir ... --output-dir ...
tests/
```

Results will go to
`databci/bciupperlimbactionexperiment/ourexpv28reproduce/outputs/<paper>/<dataset>/`.

The package imports nothing from other versions; `datascripts/` is a copy.

Figures and the model table of any result files:

```bash
python -m src.ourexpv28reproduce.evaluation.figures "$OUT"/patcnet/liu2024/PATCNet_liu2024_published_LOSO_seed2.json
```

Tests: `python -m unittest discover -s src/ourexpv28reproduce/tests -t .`
(9 tests; two read real recordings and are skipped when those are not on the
machine).
