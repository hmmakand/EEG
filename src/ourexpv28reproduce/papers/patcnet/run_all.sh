#!/usr/bin/env bash
# Run from the EEG repository root. Version 27 subjects, tasks, samples and folds for all six comparisons.
set -euo pipefail
LIU=${LIU:-/root/autodl-tmp/Experiment/EEG/data/MNE-liu2024-data}
BCI=${BCI:-/root/autodl-tmp/Experiment/EEG/data/bci-competition-iv-data-sets-2a}
JIA=${JIA:-/root/autodl-tmp/Experiment/EEG/data/jia2019}
OUT=${OUT:-/root/autodl-tmp/Experiment/EEG/outputs/ourexpv28reproduce}
LOG="$OUT/logs"
mkdir -p "$LOG"
for dataset in liu2024 bci2a jia2019; do
  case "$dataset" in
    liu2024) data_dir="$LIU" ;;
    bci2a) data_dir="$BCI" ;;
    jia2019) data_dir="$JIA" ;;
  esac
  for protocol in loso within; do
    python -u -m src.ourexpv28reproduce.run \
      --paper patcnet --dataset "$dataset" --protocol "$protocol" --cohort version27 --quiet \
      --data-dir "$data_dir" --output-dir "$OUT" \
      2>&1 | tee "$LOG/patcnet_${dataset}_version27_${protocol}.log"
  done
  python -m src.ourexpv28reproduce.evaluation.figures \
    "$OUT"/patcnet/"$dataset"/PATCNet_*_version27_*.json
done
