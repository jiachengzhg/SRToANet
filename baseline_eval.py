import numpy as np
import scipy.io as sio
import argparse
import os
from scipy.fft import fft
from scipy.signal import find_peaks

def music_estimate(cfr, n_tones, sub_size, search_range, bw):
    """
    MUSIC 算法实现 (带空间平滑以处理相干多径)
    """
    # 1. 构造空间平滑协方差矩阵
    num_subarrays = n_tones - sub_size + 1
    R = np.zeros((sub_size, sub_size), dtype=complex)
    
    for i in range(num_subarrays):
        x = cfr[i : i + sub_size]
        R += np.outer(x, x.conj())
    
    # 前向后向平滑
    J = np.fliplr(np.eye(sub_size))
    R = (R + J @ R.conj() @ J) / (2 * num_subarrays)
    
    # 2. 特征值分解
    eig_vals, eig_vecs = np.linalg.eigh(R)
    
    # 假设有 K 个主要路径 (可根据 eig_vals 的突变点做 MDL 估计，这里按基线固定为3)
    K = 3
    Un = eig_vecs[:, :-K] # 噪声子空间 (特征值默认从小到大排列)
    
    # 3. 搜索伪谱
    C = 3e8
    distances = np.arange(search_range[0], search_range[1], 0.1)
    
    # 频率间隔
    df = (bw * 1e6) / n_tones
    freq_indices = np.arange(sub_size)
    
    # 【优化】矩阵化计算所有距离的导向矢量，避免 for 循环
    taus = distances / C
    # Steering Matrix: shape = (sub_size, len(distances))
    A = np.exp(-1j * 2 * np.pi * df * np.outer(freq_indices, taus))
    
    # 伪谱 P = 1 / ||a^H * Un||^2
    # 等价于沿列求范数
    projection = Un.conj().T @ A  # shape = (sub_size-K, len(distances))
    spectrum = 1.0 / np.sum(np.abs(projection)**2, axis=0)
    
    # 4. 找第一个显著峰值 (ToA 通常关注最早到达的径)
    peaks, properties = find_peaks(spectrum, prominence=np.max(spectrum)*0.1) # 增加阈值避免噪声假峰
    
    if len(peaks) > 0:
        # 提取第一个到达的有效峰值（距离最短的）
        peak_idx = peaks[0]
    else:
        # 如果没找到局部峰值，退化为全局最大值
        peak_idx = np.argmax(spectrum)
        
    return distances[peak_idx]

def run_baselines():
    parser = argparse.ArgumentParser(description='Baseline ToA Estimation Methods')
    parser.add_argument('--mat_path', type=str, required=True, help='Path to test .mat file')
    parser.add_argument('--bw', type=float, default=40.0, help='Bandwidth in MHz')
    parser.add_argument('--up', type=int, default=2, help='Upsample rate')
    parser.add_argument('--threshold', type=float, default=0.3, help='Threshold for peak detection')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.mat_path):
        print(f"File not found: {args.mat_path}")
        return

    data = sio.loadmat(args.mat_path)
    cir_l = data['cir_l'] # (N, 2, L)
    dist_gt = data['dist'].flatten() # (N,)
    
    N, _, L = cir_l.shape
    C = 3e8
    n_tones = int(args.bw / 0.3125) # 40MHz -> 128
    
    err_ifft_peak = []
    err_ifft_threshold = []
    err_music = []

    print(f"Evaluating {args.mat_path} ({N} samples)...")
    dist_res = C / (args.bw * 1e6 * args.up)

    for i in range(min(N, 500)): 
        real = cir_l[i, 0, :]
        imag = cir_l[i, 1, :]
        cir_complex = real + 1j * imag
        amp = np.abs(cir_complex)
        
        # 1. IFFT Peak Detection
        idx_peak = np.argmax(amp)
        est_peak = idx_peak * dist_res
        err_ifft_peak.append(est_peak - dist_gt[i])
        
        # 2. IFFT Thresholding
        threshold_val = args.threshold * np.max(amp)
        indices = np.where(amp > threshold_val)[0]
        idx_threshold = indices[0] if len(indices) > 0 else 0
        est_threshold = idx_threshold * dist_res
        err_ifft_threshold.append(est_threshold - dist_gt[i])
        
        # 3. MUSIC
        cfr_full = fft(cir_complex)
        
        # 【关键修正】：确保提取的频率是从负频到正频单调递增的，保证连续性
        cfr_active = np.concatenate([cfr_full[-n_tones//2:], cfr_full[:n_tones//2]])
        
        search_min = max(0, est_peak - 20)
        search_max = est_peak + 20
        est_music = music_estimate(cfr_active, n_tones, n_tones//2, (search_min, search_max), args.bw)
        err_music.append(est_music - dist_gt[i])
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{min(N, 500)} samples...")

    def calc_rmse(err_list):
        return np.sqrt(np.mean(np.array(err_list)**2))

    print("\n" + "="*40)
    print(f"Results for: {os.path.basename(args.mat_path)}")
    print(f"IFFT Peak Detection RMSE:    {calc_rmse(err_ifft_peak):.4f} m")
    print(f"IFFT Thresholding RMSE:      {calc_rmse(err_ifft_threshold):.4f} m")
    print(f"MUSIC Algorithm RMSE:        {calc_rmse(err_music):.4f} m")
    print("="*40)

if __name__ == "__main__":
    run_baselines()