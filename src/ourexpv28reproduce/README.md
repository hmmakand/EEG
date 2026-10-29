# Version 28: commands for the server

Every paper runs on version 27's subjects, tasks, samples and test folds (leave one subject out, and
within subject with version 27's 10 folds). Details of each paper are in `papers/<paper>/README.md`.

## Setup (run once in every new terminal or tmux session)

```bash
cd /root/autodl-tmp/Experiment/EEG
LIU=/root/autodl-tmp/Experiment/EEG/data/MNE-liu2024-data
BCI=/root/autodl-tmp/Experiment/EEG/data/bci-competition-iv-data-sets-2a
JIA=/root/autodl-tmp/Experiment/EEG/data/jia2019
OUT=/root/autodl-tmp/Experiment/EEG/outputs/ourexpv28reproduce
LOG="$OUT/logs"; mkdir -p "$LOG"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 1. PA-TCNet

```bash
# Liu 2024, imagery against rest
python -u -m src.ourexpv28reproduce.run --paper patcnet --cohort version27 --quiet --dataset liu2024 --protocol loso  --data-dir "$LIU" --output-dir "$OUT" 2>&1 | tee "$LOG/patcnet_liu2024_loso.log"
python -u -m src.ourexpv28reproduce.run --paper patcnet --cohort version27 --quiet --dataset liu2024 --protocol within --data-dir "$LIU" --output-dir "$OUT" 2>&1 | tee "$LOG/patcnet_liu2024_within.log"

# BCI IV 2a, left against right hand
python -u -m src.ourexpv28reproduce.run --paper patcnet --cohort version27 --quiet --dataset bci2a --protocol loso  --data-dir "$BCI" --output-dir "$OUT" 2>&1 | tee "$LOG/patcnet_bci2a_loso.log"
python -u -m src.ourexpv28reproduce.run --paper patcnet --cohort version27 --quiet --dataset bci2a --protocol within --data-dir "$BCI" --output-dir "$OUT" 2>&1 | tee "$LOG/patcnet_bci2a_within.log"

# Jia 2019, left against right hand
python -u -m src.ourexpv28reproduce.run --paper patcnet --cohort version27 --quiet --dataset jia2019 --protocol loso  --data-dir "$JIA" --output-dir "$OUT" 2>&1 | tee "$LOG/patcnet_jia2019_loso.log"
python -u -m src.ourexpv28reproduce.run --paper patcnet --cohort version27 --quiet --dataset jia2019 --protocol within --data-dir "$JIA" --output-dir "$OUT" 2>&1 | tee "$LOG/patcnet_jia2019_within.log"

# Figures and model tables, after the runs
python -m src.ourexpv28reproduce.evaluation.figures "$OUT"/patcnet/*/PATCNet_*.json
```

## 2. EEGNet

```bash
# Liu 2024, imagery against rest
python -u -m src.ourexpv28reproduce.run --paper eegnet --quiet --save-models --dataset liu2024 --protocol loso  --data-dir "$LIU" --output-dir "$OUT" 2>&1 | tee "$LOG/eegnet_liu2024_loso.log"
python -u -m src.ourexpv28reproduce.run --paper eegnet --quiet --save-models --dataset liu2024 --protocol within --data-dir "$LIU" --output-dir "$OUT" 2>&1 | tee "$LOG/eegnet_liu2024_within.log"

# BCI IV 2a, left against right hand
python -u -m src.ourexpv28reproduce.run --paper eegnet --quiet --save-models --dataset bci2a --protocol loso  --data-dir "$BCI" --output-dir "$OUT" 2>&1 | tee "$LOG/eegnet_bci2a_loso.log"
python -u -m src.ourexpv28reproduce.run --paper eegnet --quiet --save-models --dataset bci2a --protocol within --data-dir "$BCI" --output-dir "$OUT" 2>&1 | tee "$LOG/eegnet_bci2a_within.log"

# Jia 2019, left against right hand
python -u -m src.ourexpv28reproduce.run --paper eegnet --quiet --save-models --dataset jia2019 --protocol loso  --data-dir "$JIA" --output-dir "$OUT" 2>&1 | tee "$LOG/eegnet_jia2019_loso.log"
python -u -m src.ourexpv28reproduce.run --paper eegnet --quiet --save-models --dataset jia2019 --protocol within --data-dir "$JIA" --output-dir "$OUT" 2>&1 | tee "$LOG/eegnet_jia2019_within.log"

# Figures and model tables, after the runs
python -m src.ourexpv28reproduce.evaluation.figures "$OUT"/eegnet/*/EEGNet_*.json
```

## 3. M-ShallowConvNet

```bash
# Liu 2024, imagery against rest
python -u -m src.ourexpv28reproduce.run --paper mshallowconvnet --quiet --save-models --dataset liu2024 --protocol loso  --data-dir "$LIU" --output-dir "$OUT" 2>&1 | tee "$LOG/mshallowconvnet_liu2024_loso.log"
python -u -m src.ourexpv28reproduce.run --paper mshallowconvnet --quiet --save-models --dataset liu2024 --protocol within --data-dir "$LIU" --output-dir "$OUT" 2>&1 | tee "$LOG/mshallowconvnet_liu2024_within.log"

# BCI IV 2a, left against right hand
python -u -m src.ourexpv28reproduce.run --paper mshallowconvnet --quiet --save-models --dataset bci2a --protocol loso  --data-dir "$BCI" --output-dir "$OUT" 2>&1 | tee "$LOG/mshallowconvnet_bci2a_loso.log"
python -u -m src.ourexpv28reproduce.run --paper mshallowconvnet --quiet --save-models --dataset bci2a --protocol within --data-dir "$BCI" --output-dir "$OUT" 2>&1 | tee "$LOG/mshallowconvnet_bci2a_within.log"

# Jia 2019, left against right hand
python -u -m src.ourexpv28reproduce.run --paper mshallowconvnet --quiet --save-models --dataset jia2019 --protocol loso  --data-dir "$JIA" --output-dir "$OUT" 2>&1 | tee "$LOG/mshallowconvnet_jia2019_loso.log"
python -u -m src.ourexpv28reproduce.run --paper mshallowconvnet --quiet --save-models --dataset jia2019 --protocol within --data-dir "$JIA" --output-dir "$OUT" 2>&1 | tee "$LOG/mshallowconvnet_jia2019_within.log"

# Figures and model tables, after the runs
python -m src.ourexpv28reproduce.evaluation.figures "$OUT"/mshallowconvnet/*/MShallowConvNet_*.json
```

## 4. EEG Conformer

```bash
# Liu 2024, imagery against rest
python -u -m src.ourexpv28reproduce.run --paper conformer --quiet --save-models --dataset liu2024 --protocol loso  --data-dir "$LIU" --output-dir "$OUT" 2>&1 | tee "$LOG/conformer_liu2024_loso.log"
python -u -m src.ourexpv28reproduce.run --paper conformer --quiet --save-models --dataset liu2024 --protocol within --data-dir "$LIU" --output-dir "$OUT" 2>&1 | tee "$LOG/conformer_liu2024_within.log"

# BCI IV 2a, left against right hand
python -u -m src.ourexpv28reproduce.run --paper conformer --quiet --save-models --dataset bci2a --protocol loso  --data-dir "$BCI" --output-dir "$OUT" 2>&1 | tee "$LOG/conformer_bci2a_loso.log"
python -u -m src.ourexpv28reproduce.run --paper conformer --quiet --save-models --dataset bci2a --protocol within --data-dir "$BCI" --output-dir "$OUT" 2>&1 | tee "$LOG/conformer_bci2a_within.log"

# Jia 2019, left against right hand
python -u -m src.ourexpv28reproduce.run --paper conformer --quiet --save-models --dataset jia2019 --protocol loso  --data-dir "$JIA" --output-dir "$OUT" 2>&1 | tee "$LOG/conformer_jia2019_loso.log"
python -u -m src.ourexpv28reproduce.run --paper conformer --quiet --save-models --dataset jia2019 --protocol within --data-dir "$JIA" --output-dir "$OUT" 2>&1 | tee "$LOG/conformer_jia2019_within.log"

# Figures and model tables, after the runs
python -m src.ourexpv28reproduce.evaluation.figures "$OUT"/conformer/*/Conformer_*.json
```

## 5. MSCFormer

```bash
# Liu 2024, imagery against rest
python -u -m src.ourexpv28reproduce.run --paper mscformer --quiet --save-models --dataset liu2024 --protocol loso  --data-dir "$LIU" --output-dir "$OUT" 2>&1 | tee "$LOG/mscformer_liu2024_loso.log"
python -u -m src.ourexpv28reproduce.run --paper mscformer --quiet --save-models --dataset liu2024 --protocol within --data-dir "$LIU" --output-dir "$OUT" 2>&1 | tee "$LOG/mscformer_liu2024_within.log"

# BCI IV 2a, left against right hand
python -u -m src.ourexpv28reproduce.run --paper mscformer --quiet --save-models --dataset bci2a --protocol loso  --data-dir "$BCI" --output-dir "$OUT" 2>&1 | tee "$LOG/mscformer_bci2a_loso.log"
python -u -m src.ourexpv28reproduce.run --paper mscformer --quiet --save-models --dataset bci2a --protocol within --data-dir "$BCI" --output-dir "$OUT" 2>&1 | tee "$LOG/mscformer_bci2a_within.log"

# Jia 2019, left against right hand
python -u -m src.ourexpv28reproduce.run --paper mscformer --quiet --save-models --dataset jia2019 --protocol loso  --data-dir "$JIA" --output-dir "$OUT" 2>&1 | tee "$LOG/mscformer_jia2019_loso.log"
python -u -m src.ourexpv28reproduce.run --paper mscformer --quiet --save-models --dataset jia2019 --protocol within --data-dir "$JIA" --output-dir "$OUT" 2>&1 | tee "$LOG/mscformer_jia2019_within.log"

# Figures and model tables, after the runs
python -m src.ourexpv28reproduce.evaluation.figures "$OUT"/mscformer/*/MSCFormer_*.json
```

## 6. DBConformer

```bash
# Liu 2024, imagery against rest
python -u -m src.ourexpv28reproduce.run --paper dbconformer --quiet --save-models --dataset liu2024 --protocol loso  --data-dir "$LIU" --output-dir "$OUT" 2>&1 | tee "$LOG/dbconformer_liu2024_loso.log"
python -u -m src.ourexpv28reproduce.run --paper dbconformer --quiet --save-models --dataset liu2024 --protocol within --data-dir "$LIU" --output-dir "$OUT" 2>&1 | tee "$LOG/dbconformer_liu2024_within.log"

# BCI IV 2a, left against right hand
python -u -m src.ourexpv28reproduce.run --paper dbconformer --quiet --save-models --dataset bci2a --protocol loso  --data-dir "$BCI" --output-dir "$OUT" 2>&1 | tee "$LOG/dbconformer_bci2a_loso.log"
python -u -m src.ourexpv28reproduce.run --paper dbconformer --quiet --save-models --dataset bci2a --protocol within --data-dir "$BCI" --output-dir "$OUT" 2>&1 | tee "$LOG/dbconformer_bci2a_within.log"

# Jia 2019, left against right hand
python -u -m src.ourexpv28reproduce.run --paper dbconformer --quiet --save-models --dataset jia2019 --protocol loso  --data-dir "$JIA" --output-dir "$OUT" 2>&1 | tee "$LOG/dbconformer_jia2019_loso.log"
python -u -m src.ourexpv28reproduce.run --paper dbconformer --quiet --save-models --dataset jia2019 --protocol within --data-dir "$JIA" --output-dir "$OUT" 2>&1 | tee "$LOG/dbconformer_jia2019_within.log"

# Figures and model tables, after the runs
python -m src.ourexpv28reproduce.evaluation.figures "$OUT"/dbconformer/*/DBConformer_*.json
```

## 7. SlimSeiz

```bash
# Liu 2024, imagery against rest
python -u -m src.ourexpv28reproduce.run --paper slimseiz --quiet --save-models --dataset liu2024 --protocol loso  --data-dir "$LIU" --output-dir "$OUT" 2>&1 | tee "$LOG/slimseiz_liu2024_loso.log"
python -u -m src.ourexpv28reproduce.run --paper slimseiz --quiet --save-models --dataset liu2024 --protocol within --data-dir "$LIU" --output-dir "$OUT" 2>&1 | tee "$LOG/slimseiz_liu2024_within.log"

# BCI IV 2a, left against right hand
python -u -m src.ourexpv28reproduce.run --paper slimseiz --quiet --save-models --dataset bci2a --protocol loso  --data-dir "$BCI" --output-dir "$OUT" 2>&1 | tee "$LOG/slimseiz_bci2a_loso.log"
python -u -m src.ourexpv28reproduce.run --paper slimseiz --quiet --save-models --dataset bci2a --protocol within --data-dir "$BCI" --output-dir "$OUT" 2>&1 | tee "$LOG/slimseiz_bci2a_within.log"

# Jia 2019, left against right hand
python -u -m src.ourexpv28reproduce.run --paper slimseiz --quiet --save-models --dataset jia2019 --protocol loso  --data-dir "$JIA" --output-dir "$OUT" 2>&1 | tee "$LOG/slimseiz_jia2019_loso.log"
python -u -m src.ourexpv28reproduce.run --paper slimseiz --quiet --save-models --dataset jia2019 --protocol within --data-dir "$JIA" --output-dir "$OUT" 2>&1 | tee "$LOG/slimseiz_jia2019_within.log"

# Figures and model tables, after the runs
python -m src.ourexpv28reproduce.evaluation.figures "$OUT"/slimseiz/*/SlimSeiz_*.json
```

## 8. UA-DAAN

```bash
# Liu 2024, imagery against rest
python -u -m src.ourexpv28reproduce.run --paper uadaan --quiet --save-models --dataset liu2024 --protocol loso  --data-dir "$LIU" --output-dir "$OUT" 2>&1 | tee "$LOG/uadaan_liu2024_loso.log"
python -u -m src.ourexpv28reproduce.run --paper uadaan --quiet --save-models --dataset liu2024 --protocol within --data-dir "$LIU" --output-dir "$OUT" 2>&1 | tee "$LOG/uadaan_liu2024_within.log"

# BCI IV 2a, left against right hand
python -u -m src.ourexpv28reproduce.run --paper uadaan --quiet --save-models --dataset bci2a --protocol loso  --data-dir "$BCI" --output-dir "$OUT" 2>&1 | tee "$LOG/uadaan_bci2a_loso.log"
python -u -m src.ourexpv28reproduce.run --paper uadaan --quiet --save-models --dataset bci2a --protocol within --data-dir "$BCI" --output-dir "$OUT" 2>&1 | tee "$LOG/uadaan_bci2a_within.log"

# Jia 2019, left against right hand
python -u -m src.ourexpv28reproduce.run --paper uadaan --quiet --save-models --dataset jia2019 --protocol loso  --data-dir "$JIA" --output-dir "$OUT" 2>&1 | tee "$LOG/uadaan_jia2019_loso.log"
python -u -m src.ourexpv28reproduce.run --paper uadaan --quiet --save-models --dataset jia2019 --protocol within --data-dir "$JIA" --output-dir "$OUT" 2>&1 | tee "$LOG/uadaan_jia2019_within.log"

# Figures and model tables, after the runs
python -m src.ourexpv28reproduce.evaluation.figures "$OUT"/uadaan/*/UADAAN_*.json
```

## Tests (optional, before the runs)

```bash
python -m unittest discover -s src/ourexpv28reproduce/tests -t .
```
