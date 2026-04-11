#!/bin/bash

# Waveform design experiments without attention / e2e

export CUDA_VISIBLE_DEVICES=1

WAVEFORMS=("ones" "zc" "gray" "mseq")
SNR_LIST=(0 5 10 15 20 25 30)
TRAIN_SNR="high"
NUM_TEST=100

LOG_DIR="test_results/waveform"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/waveform_exp_${TIMESTAMP}.txt"

echo "========================================" | tee -a "$LOG_FILE"
echo "Waveform experiments start - $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

for wf in "${WAVEFORMS[@]}"; do
    EXP_NAME="wf_${wf}_${TRAIN_SNR}"

    echo "" | tee -a "$LOG_FILE"
    echo "===== Train: ${EXP_NAME} =====" | tee -a "$LOG_FILE"
    python train.py --name "$EXP_NAME" --snr "$TRAIN_SNR" --waveform "$wf" \
        --use_attention False --use_e2e False --device gpu 2>&1 | tee -a "$LOG_FILE"

    echo "----- Test custom channel: ${EXP_NAME} -----" | tee -a "$LOG_FILE"
    for snr in "${SNR_LIST[@]}"; do
        python test.py --name "$EXP_NAME" --snr "$snr" --waveform "$wf" \
            --use_attention False --use_e2e False --device gpu --num_test "$NUM_TEST" \
            2>&1 | tee -a "$LOG_FILE"
    done

    echo "----- Test 802.15.4a channel: ${EXP_NAME} -----" | tee -a "$LOG_FILE"
    for snr in "${SNR_LIST[@]}"; do
        python test.py --name "$EXP_NAME" --snr "$snr" --waveform "$wf" \
            --use_attention False --use_e2e False --use_802 True \
            --device gpu --num_test "$NUM_TEST" 2>&1 | tee -a "$LOG_FILE"
    done
done

echo "========================================" | tee -a "$LOG_FILE"
echo "Waveform experiments finished - $(date)" | tee -a "$LOG_FILE"
echo "Results saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
