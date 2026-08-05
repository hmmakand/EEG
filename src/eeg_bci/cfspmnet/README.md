# CFSPMNet comparison experiments

This package contains the CFSPMNet/FRSM model, SPPM adaptation logic, and the
four controlled Liu2024 LOSO experiments under
`scripts/custom_model_scripts/cfspmnet/`.

## Comparison matrix

| Entry point | Data profile | Labels/montage | Training protocol | Default recipe |
| --- | --- | --- | --- | --- |
| `train_project_source_only_loso.py` | Project MOABB processed EDF | Raw left/right, no flip | Source only; target is first read by the model at final evaluation | 200 source epochs, AdamW + cosine, batch 16 |
| `train_canonical_source_only_loso.py` | Project MOABB processed EDF | Affected/unaffected + left-paralysis hemisphere flip | Source only; target is first read by the model at final evaluation | Same 200-epoch project recipe |
| `train_sppm_transductive_loso.py` | Project MOABB processed EDF | Affected/unaffected + hemisphere flip | 25 source epochs, then 175 transductive SPPM epochs | Same optimizer, scheduler, batch size, and total duration as the two controls |
| `train_paper_aligned_sppm_loso.py` | Direct Figshare v5 raw MAT | Raw left/right, no hemisphere flip, including CPz | Full-target transductive SPPM | 25 + 175 epochs, Adam, no scheduler, batch 40 |

The first three runs form the controlled project-preprocessed ablation. Their
equal optimizer settings and 200 source-update epochs isolate label/montage
canonicalization and SPPM target adaptation. The fourth changes the data
pipeline and training recipe together, so compare it as paper alignment, not
as a pure SPPM ablation.

## Cohort and LOSO boundary

The scientific cohort is always Liu2024 subjects 1 through 50. Every fold
trains on all trials from the other 49 subjects and evaluates all 40 trials
from the held-out subject.

- Source-only folds do not construct a target-adaptation dataset.
- Transductive SPPM uses all 40 held-out trials without labels for adaptation,
  then uses their labels only once for final evaluation.
- `--subjects` schedules held-out folds; it never shrinks the 50-subject
  source cohort. `--subjects smoke` therefore runs only held-out subject 1
  while retaining the other 49 source subjects.
- Dynamic pseudo-label refresh replaces the whole accepted set. Samples that
  no longer pass confidence/signature gates return to an all-zero label.

## Data profiles

The project profile calls the repository's existing Liu2024 MOABB pipeline:
processed 29-channel EDF, microvolt conversion, 8–30 Hz filtering, exponential
moving standardization, and one 4-second/2,000-sample trial window.

The paper-aligned profile downloads the checksummed Figshare v5
`sourcedata.zip` and `participants.tsv` directly. For every raw eight-second
trial it preserves the paper-stated filter/downsample/CAR/baseline/ICA order
while making otherwise unspecified scope and crop choices explicit:

1. second-order zero-phase Butterworth 8–30 Hz filtering;
2. full-trial downsampling from 500 Hz to 250 Hz;
3. common-average reference over all 30 EEG electrodes;
4. last-one-second pre-MI baseline correction;
5. per-subject EOG-guided FastICA on the full baseline-corrected trial;
6. four-second MI crop.

It retains the recorded left-hand/right-hand labels and original channel
orientation. Unlike the project SPPM experiment, it does not remap classes to
affected/unaffected or flip hemispheres, because the paper describes its task
and results in the left/right label space.

The paper does not specify its baseline interval, ICA selection settings, where
ICA is fit relative to MI cropping, or Stage-II epoch count. Those choices are
explicit in run metadata, so this is named `paper_aligned`, not an exact
reproduction. The paper reports a clinically selected 24-subject subcohort;
this project intentionally keeps all 50 subjects as requested.

## Commands

```bash
# 1) Project source-only baseline:
#    MOABB processed 29-channel EDF, original left/right labels, no target adaptation.
#    Trains on the other 49 subjects; held-out labels are used only for final evaluation.
python scripts/custom_model_scripts/cfspmnet/train_project_source_only_loso.py

# 2) Canonical source-only control:
#    Same data/training recipe as experiment 1, but maps labels to
#    affected/unaffected and flips hemispheres for left-paralysis subjects.
#    Isolates canonicalization; still performs no target adaptation.
python scripts/custom_model_scripts/cfspmnet/train_canonical_source_only_loso.py

# 3) Project transductive SPPM:
#    Same MOABB profile and canonicalization as experiment 2. Stage I is source-only;
#    Stage II adapts with all held-out trials unlabeled, then evaluates those trials.
#    Compare with experiment 2 to isolate the effect of SPPM adaptation.
python scripts/custom_model_scripts/cfspmnet/train_sppm_transductive_loso.py

# 4) Paper-aligned transductive SPPM:
#    Loads direct Figshare raw MAT trials (30 EEG channels including CPz), applies the
#    paper-aligned preprocessing profile, retains raw left/right labels without a
#    hemisphere flip, then runs full-target transductive SPPM.
#    Changes both data and training recipe, so this is not a pure SPPM ablation.
python scripts/custom_model_scripts/cfspmnet/train_paper_aligned_sppm_loso.py
```

Inspect a smoke-fold configuration without loading or downloading data:

```bash
python scripts/custom_model_scripts/cfspmnet/train_sppm_transductive_loso.py \
  --subjects smoke --print-config --no-download
```

The existing project `liu2024_loso` benchmark uses two epochs. For a
two-epoch, budget-matched CFSPMNet comparison against those rows, run:

```bash
python scripts/custom_model_scripts/cfspmnet/train_project_source_only_loso.py \
  --stage-i-epochs 2 --stage-ii-epochs 0
```

That command matches the project data/optimizer/scheduler/batch/LR/weight-decay
budget, but not the older runner's random sequence: the existing benchmark uses
seed 42 once, while this runner uses seed 2 and reseeds each fold. For a strict
statistical comparison, rerun every model under one seed policy. The two-epoch
result has its own fingerprint; do not use it as the source-only control for
the 200-epoch SPPM ablation.

Likewise, reducing the paper-aligned `25 + 175` schedule is useful for checking
that loading, preprocessing, Stage I, Stage II, and artifact writing all run.
It is not a reliable preview of reproduction accuracy: SPPM depends on Stage I
first learning a confident source classifier, and a very short warm-up can leave
the target pseudo-label set empty. The console reports Stage I and Stage II
separately and warns whenever the locked epoch preset is overridden.

The raw archive is about 1.87 GiB and is downloaded only by the paper-aligned
entry point. Use `--no-download` to require an existing verified cache,
`--figshare-cache-dir PATH` for a custom cache, and `--skip-ica` only for
diagnostic checks.

## Reproducibility and outputs

Each fold is reseeded with `base_seed + held_out_subject`. Resume keys include
the scientific configuration, relevant source-file hashes, project YAML
hashes, library versions, and preprocessing version. Operational options such as
the scheduled fold subset and output/cache paths do not change the scientific
fingerprint.

Artifacts are written beneath `outputs/runs/`, TensorBoard data beneath
`outputs/tensorboard/`, and master rows beneath
`outputs/results/liu2024/leave_one_subject_out/`. A failed fold is persisted
with `status=error`, later folds continue, and the CLI returns a nonzero status
if any newly requested fold fails.
