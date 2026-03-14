#!/bin/bash

# 综合测试脚本：测试所有实验配置在不同SNR下的性能
# 实验配置：
# 1. origin_high: 原始方法（High SNR训练）
# 2. origin_low: 原始方法（Low SNR训练）
# 3. attention_high: 加入注意力机制（High SNR训练）
# 4. attention_low: 加入注意力机制（Low SNR训练）
# 每个配置测试两种权重：原始权重 vs E2E权重
# 并且会测试自定义通道和802.15.4a通道的性能

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=7

# 设置输出日志文件
LOG_DIR="test_results"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/test_log_${TIMESTAMP}.txt"

echo "========================================" | tee -a $LOG_FILE
echo "开始综合测试 - $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

# 定义SNR测试点
SNR_LIST=(0 5 10 15 20 25 30)

# 定义实验配置
declare -a EXP_NAMES=("origin_high" "origin_low" "attention_high" "attention_low")
declare -a USE_ATTENTION=("False" "False" "True" "True")

# 循环测试每个实验配置
for idx in "${!EXP_NAMES[@]}"; do
    exp_name="${EXP_NAMES[$idx]}"
    use_att="${USE_ATTENTION[$idx]}"
    
    echo "" | tee -a $LOG_FILE
    echo "========================================" | tee -a $LOG_FILE
    echo "测试实验: $exp_name (Attention=$use_att)" | tee -a $LOG_FILE
    echo "========================================" | tee -a $LOG_FILE
    
    # 测试原始权重（不使用E2E）
    echo "" | tee -a $LOG_FILE
    echo "--- 测试原始权重（不使用E2E）---" | tee -a $LOG_FILE
    for snr_val in "${SNR_LIST[@]}"; do
        echo "[Original] Testing $exp_name at SNR=$snr_val dB" | tee -a $LOG_FILE
        python test.py --name $exp_name --snr $snr_val \
            --use_attention $use_att --use_e2e False \
            --device gpu --num_test 100 2>&1 | tee -a $LOG_FILE
    done
    
    # 测试E2E权重
    echo "" | tee -a $LOG_FILE
    echo "--- 测试E2E优化权重 ---" | tee -a $LOG_FILE
    for snr_val in "${SNR_LIST[@]}"; do
        echo "[E2E] Testing $exp_name at SNR=$snr_val dB" | tee -a $LOG_FILE
        python test.py --name $exp_name --snr $snr_val \
            --use_attention $use_att --use_e2e True \
            --device gpu --num_test 100 2>&1 | tee -a $LOG_FILE
    done
done

echo "" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "开始测试泛化能力（IEEE 802.15.4a 标准通道）" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

# 测试802.15.4a通道（泛化能力测试）
for idx in "${!EXP_NAMES[@]}"; do
    exp_name="${EXP_NAMES[$idx]}"
    use_att="${USE_ATTENTION[$idx]}"
    
    echo "" | tee -a $LOG_FILE
    echo "========================================" | tee -a $LOG_FILE
    echo "测试实验 (802.15.4a): $exp_name (Attention=$use_att)" | tee -a $LOG_FILE
    echo "========================================" | tee -a $LOG_FILE
    
    # 测试原始权重（不使用E2E）
    echo "" | tee -a $LOG_FILE
    echo "--- [802.15.4a] 测试原始权重（不使用E2E）---" | tee -a $LOG_FILE
    for snr_val in "${SNR_LIST[@]}"; do
        echo "[802.15.4a Original] Testing $exp_name at SNR=$snr_val dB" | tee -a $LOG_FILE
        python test.py --name $exp_name --snr $snr_val \
            --use_attention $use_att --use_e2e False --use_802 True \
            --device gpu --num_test 100 2>&1 | tee -a $LOG_FILE
    done
    
    # 测试E2E权重
    echo "" | tee -a $LOG_FILE
    echo "--- [802.15.4a] 测试E2E优化权重 ---" | tee -a $LOG_FILE
    for snr_val in "${SNR_LIST[@]}"; do
        echo "[802.15.4a E2E] Testing $exp_name at SNR=$snr_val dB" | tee -a $LOG_FILE
        python test.py --name $exp_name --snr $snr_val \
            --use_attention $use_att --use_e2e True --use_802 True \
            --device gpu --num_test 100 2>&1 | tee -a $LOG_FILE
    done
done

echo "" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "所有测试完成 - $(date)" | tee -a $LOG_FILE
echo "结果保存在: $LOG_FILE" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE