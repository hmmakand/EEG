#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: bash src/ourexpv29/run_all.sh LIU_DATA BCI_DATA JIA_DATA [OUTPUT_DIR]" >&2
  exit 2
fi
liu_data=$1
bci_data=$2
jia_data=$3
output_dir=${4:-outputs/v29}
for dataset in liu2024 bci2a jia2019; do
  case "$dataset" in
    liu2024) data_dir=$liu_data ;;
    bci2a) data_dir=$bci_data ;;
    jia2019) data_dir=$jia_data ;;
  esac
  for protocol in loso within; do
    python -u -m "src.ourexpv29.train.run_${protocol}" \
      --dataset "$dataset" --data-dir "$data_dir" --output-dir "$output_dir" \
      --epochs 100 --patience 15 --device "${DEVICE:-cuda}" --save-models
  done
done
