# Version five: overfitting and poor-results audit

Date: 2026-09-15 (updated same day after a controlled rerun; see [Update](#0-update-2026-09-15-controlled-rerun-with-self-only-edges-z-score-normalization-linear-head-lower-learning-rate)). Scope: current model, loaders, split construction, training/selection, metrics, upstream feature construction, both saved runs, and local dataset arrays. Training implementation was not changed.

## Conclusion

**There is a substantial training/generalization gap, but overfitting alone is not an adequate diagnosis.** Both saved within-subject experiments perform at chance. Many folds also fit their training data poorly. Strong graph mixing, unscaled inputs, very small validation sets, and train/evaluation behavior differences deserve attention before another large sweep.

A separate, confirmed data-integrity issue exists: subjects 4 and 5 share 15 exactly identical raw trial windows. Subject-ID-disjoint LOSO therefore does not guarantee independent signal content.

## 0. Update 2026-09-15: controlled rerun with self-only edges, z-score normalization, linear head, lower learning rate

After this audit, the within-subject CLI's defaults were changed to
`--node-normalization zscore --edge-mode self_only --classifier linear
--learning-rate 0.001 --no-early-stopping` (all other settings, including
seed 42 and `without_csd_alpha_wpli`, unchanged), and `within_subject_full`
was regenerated with these new defaults, **overwriting the previous run in
place** (`overwrite: true` in the manifest). This combines four of this
audit's Section 8 comparisons (normalization, self-only edges, a smaller
linear head, a lower learning rate) into one run rather than isolating them
individually, so it cannot attribute the change to any single factor.

**The regenerated run stays at chance on discrimination, but calibration and
class-collapse improved substantially:**

| Measurement | Old `within_subject_full` (weighted, mlp, none, lr 0.01, patience 10) | New `within_subject_full` (self_only, linear, zscore, lr 0.001, no early stopping) |
|---|---:|---:|
| Balanced accuracy | 50.20% (95% CI 48.00-52.55%) | 52.30% (95% CI 49.15-55.60%) |
| Mean fold AUC | 0.5133 (95% CI 0.478-0.549) | 0.5178 (95% CI 0.473-0.564) |
| Cohen's kappa | 0.004 (95% CI -0.041-0.050) | 0.046 (95% CI -0.018-0.113) |
| Held-out mean loss | 1.1935 | **0.7850** |
| Last online training accuracy | 70.16% | 70.58% |
| Final-training (eval-mode) accuracy | 70.35% | 76.64% |
| Train/eval accuracy gap | 20.15 points | **24.34 points** |
| Train/eval loss gap | 0.548 | 0.327 |
| Folds predicting only one class | 42/250 (from the earlier "full" row) | **17/250** |
| Folds with training accuracy >= 90% | 30/250 | 69/250 |
| Folds with training accuracy <= 60% | 63/250 | 99/250 |
| Held-out accuracy for the >=90%-training-accuracy subset | not previously recorded | 60.33% |
| Median selected epoch (of up to 50) | 8 | 10 |
| Selected epoch <= 2 despite a full 50-epoch selection budget | 33/250 | **85/250 (34%)** |

Old-column numbers above are read directly from the `within_subject_full`
summary as it existed immediately before this update's rerun overwrote it
(the exact file examined earlier in this session), not retyped from Section
1's table. Its balanced-accuracy CI (48.00-52.55%) differs slightly from
Section 1's quoted accuracy CI for the same run (47.95-53.10%) even though
bootstrap resampling is seeded (`seed=42` in
`summarize_within_subject_results`); since a seeded bootstrap over identical
per-subject means is deterministic, this points to the two documents
describing two different training executions of the same configuration
(consistent with training-time non-determinism in the underlying GCN
training, not a transcription error), not to CI methodology. This is a
reproducibility gap worth closing (pin and record a training-determinism
seed/mode, not just the data-split and bootstrap seeds) before treating any
single run's numbers as exact.

Both CI intervals for balanced accuracy, AUC, and kappa still straddle their
no-discrimination value (50%, 0.5, 0 respectively), so **this rerun does not
establish real predictive signal any more than the original run did.** The
held-out loss improvement (1.19 -> 0.79) mostly reflects better-calibrated,
less-confidently-wrong probabilities from the lower learning rate, z-score
input scaling, and smaller/simpler head -- not improved discrimination.
Held-out loss (0.785) is still worse than the constant-0.5-probability
baseline of ln(2) = 0.693, i.e. a hard-coded coin flip that reports no
confidence still beats this model's calibrated loss.

Two findings sharpen, rather than resolve, prior open questions:

- **Epoch-selection noise persists even with the early-stopping patience
  removed and the selection pass always running the full 50-epoch budget.**
  34% of folds still pick their minimum-validation-loss epoch at epoch 1 or
  2, versus 13% before. This is evidence *against* "not enough epochs" or
  "early stopping cuts training short" as the cause of erratic epoch
  selection; the 8-trial inner-validation split is still noisy enough that
  loss keeps rising, epoch over epoch, in a third of folds, well before the
  budget is exhausted. This isolates the inner-validation-split size (see
  Section 5) as the dominant open issue, independent of architecture,
  normalization, classifier size, or learning rate.
- **The train/eval accuracy gap widened slightly (20.15 -> 24.34 points)**
  even though the parameter count fell from 13,721 to about 1,106 and
  learning rate fell 10x. More folds now reach >=90% training accuracy
  (69/250 vs. 30/250), and that high-training-accuracy subset still only
  reaches 60.33% held-out accuracy. A smaller model and gentler optimization
  reduced confident wrong answers (loss) but did not reduce the
  training/held-out gap in accuracy terms; with 24-32 training trials per
  fold, ~1,106 parameters can still fit training noise.

**Reproducibility caveat:** because `within_subject_full` was overwritten
in place rather than given a new `--run-name`, the raw per-fold artifacts
backing the original "later run" column in Section 1's table (and the
`within_subject_full_fixed` run referenced throughout this document) no
longer exist anywhere on disk -- `outputs/` is untracked (see git status)
and `audit/check_results.py` only sees whatever run directories currently
exist, so `audit/evidence.json` has already been regenerated to reflect only
the current run. The numbers quoted for those two historical runs elsewhere
in this document are preserved only as static text here, not as
independently reproducible evidence. Future controlled comparisons should
follow the README's own advice (`training/README.md`'s "Use a new run name
and keep your normalized comparison's settings") and give every comparison
a distinct `--run-name`, never reusing one that other sections of this
audit still cite as evidence.

This rerun does not test any Section 8 comparison in isolation (it changed
four settings at once), does not address the subject 4/5 duplicate-trial
issue (Section 2), and does not run LOSO. The next controlled step is still
to isolate the inner-validation-split-size fix (larger
`inner_validation_fraction`, repeated inner splits, or inner k-fold) from
the other four changes already bundled into this rerun.

## 1. What the saved runs actually show

Both runs use without-CSD nodes, alpha wPLI, eight features, 50 subjects, five folds, one repeat, and seed 42. Each subject contributes 40 trials; every outer fold has 32 training and eight evaluation trials. Selection uses 24 training and eight inner-validation trials.

| Measurement | `within_subject_full_fixed` | `within_subject_full` (later run) |
|---|---:|---:|
| Batch size | 32 | 8 |
| Selection patience | 10 | disabled |
| Held-out accuracy / balanced accuracy | 50.40% | 50.50% |
| Saved 95% subject-bootstrap accuracy interval | 48.80–52.15% | 47.95–53.10% |
| Mean fold AUC | 0.5126 | 0.5128 |
| Cohen's kappa | 0.008 | 0.010 |
| Held-out negative log likelihood | 1.5221 | 1.3128 |
| Last recorded online training accuracy | 65.29% | 70.16% |
| Online training minus evaluation accuracy | 14.89 points | 19.66 points |
| Last online training loss | 0.6092 | 0.5959 |
| Folds with online training accuracy ≥90% | 15/250 | 30/250 |
| Folds with online training accuracy ≤60% | 106/250 | 63/250 |
| Median selected epoch | 4 | 8 |
| Selected epoch ≤2 | 91/250 | 33/250 |
| Folds predicting only one class | 149/250 | 42/250 |

The intervals include 50%; neither run establishes useful aggregate discrimination. A constant 0.5 probability has log loss ln(2) = 0.6931: both models make substantially worse probability predictions. This indicates confident mistakes, although formal calibration curves were not available.

**Training-metric caveat:** `training/engine.py::_run_loader` records predictions during optimization with dropout enabled and BatchNorm using batch statistics. They come from successive model states, before each update. Evaluation uses a fixed final model with dropout disabled and running BatchNorm statistics. These gaps are warning signs, not clean same-checkpoint overfitting measurements. Even the ≥90% training subsets achieve only 50.83% and 58.75% evaluation accuracy, respectively; those are descriptive, post-hoc subsets.

The run name `fixed` is misleading chronologically: it started at 01:27 UTC, whereas `full` started at 02:06 UTC. Both already record `inner_selection`. Historical comments saying training-loss checkpointing caused the failure do not describe these saved runs. The two runs changed both batch size and patience, so their differences cannot isolate either effect.

No saved LOSO results were present in this experiment's output directory. These results cannot establish cross-subject performance.

## 2. Confirmed data-integrity issue: duplicate signals across subjects

Exact byte hashing of all 2,000 without-CSD node tensors found 15 duplicate pairs, all between subjects 4 and 5. The corresponding complete five-band without-CSD wPLI tensors and labels are identical too.

Zero-based trial indices: **5, 9, 13, 14, 19, 20, 22, 26, 27, 29, 31, 32, 35, 36, 38**.

I then loaded the two original local EDF files and compared all channels over all 40 corresponding saved trial windows. Those same 15 windows are exactly equal in the raw recordings. This establishes that the duplication is already present in the local source signals; it is not merely a graph-array serialization collision. It does not establish why the source recordings contain these copies.

Consequences:

- Ordinary subject-ID splits can place identical signal content in different LOSO partitions, including train/test or validation/test.
- Within-subject folds do not mix these two subjects, so this does **not** explain chance-level within-subject performance. It does compromise the assumption that all 50 subject results are independent for bootstrap inference.
- Trace source-file provenance and compare against the original dataset distribution. Pending resolution, group these subjects together for cross-subject evaluation or report a prespecified exclusion sensitivity analysis. Grouping changes strict single-subject LOSO and should be named explicitly. Do not silently delete trials to improve scores.
- Add content-duplicate checks to dataset validation; present shape/finiteness/subject-ID checks do not catch this.

Evidence: [duplicate pairs](audit/duplicates.json), [raw-window comparison](audit/raw_duplicate_check.json).

## 3. Strongest model issue: spatial information is heavily averaged

`model/gcn.py` applies one `GCNConv(8, 16)` on all 812 directed edges, then BatchNorm, activation, dropout, flattening, and a dense classifier. `data/without_csd_alpha_wpli.py` passes all alpha weights through without sparsification. All observed alpha weights are positive; median 0.3562, interquartile range 0.2397–0.4987.

I reconstructed the normalized adjacency used by this configuration:

`P = D^(-1/2) (A + I) D^(-1/2)`

For each feature, I measured mean across-trial electrode variance after `P @ X`, divided by the corresponding variance before mixing. Feature columns were standardized globally **only for this descriptive calculation**, never for predictive evaluation.

| Feature | Spatial variance retained |
|---|---:|
| Delta power | 1.80% |
| Theta power | 1.66% |
| Alpha power | 1.98% |
| Beta power | 1.43% |
| Gamma power | 1.41% |
| Spectral entropy | 2.45% |
| Hjorth mobility | 2.24% |
| Hjorth complexity | 2.31% |

Thus this operation removes about **97.5–98.6% of this spatial-variance measure before the nonlinear classifier**. This is a direct measurement of input mixing, not a proof that precisely that fraction of class information is lost. Learned channel mixing and BatchNorm can rescale residual variation. Nevertheless, for an electrode-specific task, it is a strong reason to test an identity/self-only graph, a raw-node residual path, and a sparse graph.

The architecture has 13,721 trainable parameters; 13,485 are in the 464-to-29 dense layer. That is considerable fitting capacity for 24 selection-training trials or 32 final-training trials. Flattening retains electrode order, but cannot fully undo information lost by earlier averaging. Increasing depth or training longer is not the first experiment to run.

## 4. Feature scale and optimization risks

Both runs use `node_normalization=none`. Dataset-wide standard deviations are approximately:

`[7.070, 5.073, 4.942, 4.599, 5.675, 0.2225, 0.05538, 6.305]`.

The largest-to-smallest ratio is about 128. Entropy and mobility enter the first learned mixture at much smaller scales than band powers and complexity. BatchNorm occurs **after** this mixing and does not independently normalize the eight input features.

Train-only z-scoring already exists and is correctly scoped in the normal orchestration paths. It should be a controlled comparison. Large observed power values (up to 64.51 dB) and complexity up to 73 warrant trial-level outlier inspection, but their magnitudes alone do not prove artifacts. Upstream `prepare_eeg` explicitly applies no extra artifact correction, filtering, or reference change; Hjorth features use the broadband signal.

AdamW learning rate 0.01, dropout 0.5, and weight decay 0.0005 are untuned defaults. AdamW's direct decay multiplier is about 0.999995 per optimizer step, so this decay setting is weak by itself. The combination of high dropout, tiny data, and a large classifier can produce both poor fitting and memorization in different folds. A smaller learning rate and input normalization are hypotheses to test, not established fixes.

## 5. Epoch selection and BatchNorm need better diagnostics

In `training/within_subject.py::_run_selection_pass`, a single eight-trial validation set selects among as many as 50 epochs. One classification error changes validation accuracy by 12.5 percentage points; loss is also vulnerable to individual confident errors. Selecting the minimum of many noisy losses can overfit validation even when the outer evaluation split is properly isolated.

The final pass reinitializes and transfers the selected **epoch count**, not the selected model. At batch size 8, selection has three updates per epoch and final training has four. This is a 33% change in update count for the same selected epoch, along with changed training data and potentially different normalization statistics. This is conventional refitting, not leakage, but the selected budget may transfer poorly.

BatchNorm is another plausible contributor: small numbers of optimizer batches update running statistics, while training uses current batch statistics. Very early selection can evaluate with poorly adapted running statistics. The observed class collapse and poor held-out loss are consistent with this possibility; they do not prove it.

Needed measurements:

1. Evaluate the selected final model on its training set in `eval()` mode, alongside outer evaluation.
2. Save selection-pass training and validation curves, not just the minimum validation loss and selected epoch.
3. Save per-trial labels, predicted probabilities, fold/subject/trial IDs, and final model checkpoints for within-subject runs.
4. Record BatchNorm diagnostics; compare a normalization alternative or training-only recalibration in a controlled experiment. Never update normalization statistics using held-out trials.
5. Compare repeated inner splits or inner CV and a robust selected training budget; separately report split-seed and initialization-seed sensitivity.

Current within-subject output lacks checkpoints and per-trial probabilities, preventing an exact retrospective same-model train/evaluation check.

## 6. New baseline experiments performed in this audit

I ran fixed, untuned `StandardScaler + LogisticRegression(C=1, max_iter=2000)` on all 50 subjects, with five stratified folds and seed 42, matching the saved outer split construction. Scaling was fitted separately on each fold's 32 training trials. Each outer trial was predicted once per baseline. These baselines use all outer training trials directly and require no epoch selection.

| Representation | Mean subject accuracy | Mean fold AUC |
|---|---:|---:|
| All eight node features, flattened (232 inputs) | **56.55%** | **0.5740** |
| Alpha and beta node powers only (58 inputs) | 52.05% | 0.5153 |
| Alpha wPLI unique edges (406 inputs) | 52.90% | 0.5385 |
| Existing later GCN run | 50.50% | 0.5128 |

This supplies evidence that the current GCN pipeline fails to exploit some information available to a simpler baseline. It does **not** isolate graph mixing from scaling, optimizer, regularization, or epoch-selection effects. These are exploratory comparisons on an already-inspected evaluation dataset, not a newly independent final benchmark. Shared source content also affects independence across subjects. Do not conclude that there is no signal in the features, or that 56.55% is deployment-quality performance.

## 7. What appears correct, and remaining scope limits

- Saved outer training/evaluation graph indices are disjoint in all 500 folds.
- Current LOSO code validates subject and graph separation; inner search excludes the outer test subject.
- Within-subject epoch selection uses only the outer training partition.
- Optional scaling is fitted to training indices; subject-keyed normalization dictionaries intentionally contain shared training statistics, not held-out subject fits.
- LOSO deep-copies the best validation-loss state, restores it, and evaluates test afterward.
- Log-softmax + NLLLoss is consistent; AUC uses exponentiated positive-class log probability, not hard labels or an extra softmax.
- Reverse edges carry symmetric weights. Two directed entries are the expected representation for undirected message passing.
- Selected node arrays and alpha edges are finite, with 1,000 labels of each class globally.
- Source code reads class labels from the event schedule and uses MI start/stop markers, rather than treating the stage marker as a hand class. Independent physiological/event-annotation verification across all EDFs was not performed.

Index separation does not address the confirmed content duplication. Random within-subject folds also do not estimate future-session performance or rule out temporal confounding. Binary positive-class F1 can be misleading under class collapse; balanced accuracy, AUC, both-class recall, and confusion matrices should lead reporting.

All **111 tests and 24 subtests passed** (`python -m pytest src/ourexperimentversionfive -q`, 35.66 s). One upstream Torch deprecation warning occurred. Passing tests establish tested implementation behavior, not scientific validity or predictive quality.

## 8. Prioritized action plan

### First: establish trustworthy data and observability

- Resolve duplicated source trials and define an evaluation grouping/exclusion policy before publishing LOSO estimates.
- Add saved same-checkpoint training metrics, complete selection curves, per-trial predictions, and within-subject checkpoints.
- Preserve run code and dataset hashes. Current manifests record a dirty git commit while the experiment directory is untracked; a commit ID alone cannot reconstruct the producing code.

### Next: isolate the cause with small controlled comparisons

Use prespecified development subjects and fixed folds/seeds, with one change per comparison:

1. ~~Current GCN with training-only z-score versus raw inputs.~~ Bundled into the [2026-09-15 rerun](#0-update-2026-09-15-controlled-rerun-with-self-only-edges-z-score-normalization-linear-head-lower-learning-rate) with three other changes at once, not isolated. Still needs a z-score-only-vs-raw comparison holding edge mode/classifier/learning rate fixed.
2. ~~Same scaled GCN with self-only edges versus the complete weighted graph~~ — same caveat: bundled, not isolated. A raw-node residual or sparse graph remains untested.
3. ~~Smaller learning rate (e.g. 0.001 versus 0.01) with all other settings fixed.~~ — same caveat: bundled, not isolated.
4. ~~Smaller classifier~~ (`--classifier linear`, 13,721 -> 1,106 params) tested, bundled with the three changes above; a BatchNorm alternative is still untested.
5. Keep the fixed linear baseline. Run a tiny-training-set memorization check and shuffled-label negative control, keeping these diagnostics separate from reported held-out evaluation.
6. **New, higher priority after the rerun:** increase the inner-validation split size (larger `inner_validation_fraction`, repeated inner splits, or inner k-fold) and re-measure how often the selected epoch sits at 1-2 out of a full budget — the rerun shows this got *worse* (13% to 34% of folds) even after removing early stopping and changing four other settings, so it is not explained by any of them.

None of the five comparisons above were actually isolated by the 2026-09-15 rerun: it changed normalization, edge mode, classifier, and learning rate together in one run. A genuine one-change-at-a-time sweep, ideally on a small prespecified subject subset before spending a full 50-subject run per cell, is still outstanding.

Do not select the best result from repeated inspection of all 50 subjects and call it unbiased. Put architecture/normalization choices inside the development selection process or reserve fresh evaluation data. Use subject/group-level uncertainty, accounting for duplicates, rather than treating 250 overlapping-training folds as independent observations.

### Finally: full evaluation

After the data issue and controlled comparisons, run a locked within-subject protocol across seeds, then the corrected cross-subject protocol. Report accuracy, balanced accuracy, AUC, loss, confusion matrices, uncertainty, and same-model training gaps. More epochs alone, stronger dropout alone, or a large nested sweep on the existing setup is not supported as the next fix by this audit.

## Reproduction and evidence

- [Audit script](audit/check_results.py): run `OPENBLAS_NUM_THREADS=1 python src/ourexperimentversionfive/audit/check_results.py` from repository root. It reads the dataset and saved runs, checks fold overlap, calculates graph mixing, and reruns the three fixed baselines. It writes only `audit/evidence.json`, scoped to whatever run directories currently exist under `outputs/`.
- [Numeric evidence](audit/evidence.json) — **regenerated on 2026-09-15 against only the current `within_subject_full` run** (see the Update section above); it no longer contains `within_subject_full_fixed`, which was already absent from `outputs/` before this update and has no surviving raw artifacts. [Duplicate pairs](audit/duplicates.json) and [raw-window verification](audit/raw_duplicate_check.json) are unaffected by the rerun (dataset-level, not run-level).
- Only `outputs/within_subject_full` (now the self-only/z-score/linear/lr-0.001/no-early-stopping rerun) remains on disk; the original weighted/mlp/none/lr-0.01 run this document's Section 1 and Section 6 tables describe was overwritten in place and cannot be reproduced from saved artifacts, only from its manifest's recorded configuration.

The audit did not retrain the GCN, run a new LOSO sweep, regenerate features, or alter training code beyond the CLI-default change already described in the Update section. The proposed model changes remain controlled experiments, not promised performance improvements.
