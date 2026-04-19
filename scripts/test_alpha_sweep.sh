#!/bin/bash

set -euo pipefail

# Batch test for alpha experiments:
# - experiment folders: origin_alpha{1,3,5,7,9}
# - alpha values: 0.1/0.3/0.5/0.7/0.9
# - SNR: 5, 30
# - channel: custom + 802

export CUDA_VISIBLE_DEVICES=1

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="test_results"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/alpha_sweep_${TIMESTAMP}.log"

ALPHA_TAGS=(1 3 5 7 9)
SNR_LIST=(5 30)
USE_802_LIST=(False True)

echo "========================================" | tee -a "$LOG_FILE"
echo "Alpha sweep test start: $(date)" | tee -a "$LOG_FILE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

for tag in "${ALPHA_TAGS[@]}"; do
    exp_name="origin_alpha${tag}"
    alpha_val="0.${tag}"
    exp_dir="experiments/${exp_name}"

    echo "" | tee -a "$LOG_FILE"
    echo ">>> Experiment: ${exp_name}, alpha=${alpha_val}" | tee -a "$LOG_FILE"

    if [[ ! -d "$exp_dir" ]]; then
        echo "[SKIP] Missing experiment folder: $exp_dir" | tee -a "$LOG_FILE"
        continue
    fi

    # Quick weight check (default test.py path: sr.w/ra.w/rb.w)
    if [[ ! -f "${exp_dir}/sr.w" || ! -f "${exp_dir}/ra.w" || ! -f "${exp_dir}/rb.w" ]]; then
        echo "[SKIP] Missing weights in $exp_dir (need sr.w, ra.w, rb.w)" | tee -a "$LOG_FILE"
        continue
    fi

    for snr in "${SNR_LIST[@]}"; do
        for use_802 in "${USE_802_LIST[@]}"; do
            echo "[RUN] name=${exp_name}, alpha=${alpha_val}, snr=${snr}, use_802=${use_802}" | tee -a "$LOG_FILE"
            python test.py \
                --name "$exp_name" \
                --alpha "$alpha_val" \
                --snr "$snr" \
                --use_802 "$use_802" \
                --device gpu 2>&1 | tee -a "$LOG_FILE"
        done
    done
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Alpha sweep test done: $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

