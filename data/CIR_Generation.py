######################################################################
# This code is used to generate CIR
# 
# Author: Jiacheng Zhang
# Python implementation
######################################################################

import os
import numpy as np
from scipy.io import loadmat, savemat
import torch
import sys

# Add root to sys.path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.waveform_net import WaveformNet

def sum_exponentials(freq_grid, paths, coef):
    """
    计算指数和的频率响应
    
    Parameters:
        freq_grid: 频率网格 (1D array)
        paths: 路径延迟 (1D array)
        coef: 路径系数 (1D array)
    
    Returns:
        freq_resp: 频率响应 (1D array)
    """
    N = len(paths)
    freq_resp = np.zeros(freq_grid.shape, dtype=complex)
    for i in range(N):
        freq_resp = freq_resp + coef[i] * np.exp(-1j * 2 * np.pi * freq_grid * paths[i])
    
    return freq_resp


def generate_dataset(saved_dist, saved_mag, saved_paths, SNR_list, ofdm_bw, upsample, path, debug, alpha_val=None):
    """
    生成数据集
    
    Parameters:
        saved_dist: 保存的距离数据
        saved_mag: 保存的幅度数据
        saved_paths: 保存的路径数据
        SNR_list: SNR列表
        ofdm_bw: OFDM带宽
        upsample: 上采样率
        path: 保存路径
        debug: 调试模式 (1 显示, 0 生成)
        alpha_val: 波形设计偏好因子 (0: 通信优先, 1: 定位优先), None 表示不使用波形设计
    """
    import matplotlib.pyplot as plt
    
    DISPLAY = debug  # 1 for display mode, 0 for generation mode
    
    tones_gap = 312.5e3  # 802.11 g/n/ac standard
    N_tones = round(ofdm_bw / tones_gap)  # number of subcarriers
    
    # Load WaveformNet if alpha_val is provided
    waveform_net = None
    if alpha_val is not None:
        waveform_net = WaveformNet(N_tones)
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 '/data2/jiacheng.zhang/SRToANet/experiments/waveform_design/waveform_net.w')
        if os.path.exists(model_path):
            waveform_net.load_state_dict(torch.load(model_path, map_location='cpu'))
            waveform_net.eval()
            print(f"Loaded WaveformNet with alpha={alpha_val}")
        else:
            print(f"Warning: WaveformNet weights not found at {model_path}. Using uniform power.")
            waveform_net = None
    
    signal_pwr_dB = 10
    signal_pwr = 10 ** (signal_pwr_dB / 10)
    
    C = 3e8  # speed of light
    
    ################## Generate dataset ###################
    
    N = len(saved_dist)
    cir_l = np.zeros((N, 2, N_tones * upsample))
    cir_h = np.zeros((N, 2, N_tones * upsample))
    cfr_h = np.zeros((N, 2, N_tones * upsample))
    
    print(f'Generating {path}...,  Upsample: {upsample}, Bandwidth: {int(ofdm_bw / 1e6)}')
    
    for kk in range(len(saved_dist)):
        ##################################################
        # Channel Delay Spread Setting
        ##################################################
        SNR = np.random.choice(SNR_list)
        noise_pwr_dB = signal_pwr_dB - SNR
        noise_pwr = 10 ** (noise_pwr_dB / 10)
        
        # Sampling points for small and large bands
        dd = saved_dist[kk]
        
        # Sampling grid for the channel frequency response
        # MATLAB: -1/2 : 1/N_tones : 1/2 - 1/N_tones  生成 N_tones 个点
        grid_l = ofdm_bw * np.arange(-1/2, 1/2, 1/N_tones)
        grid_h = ofdm_bw * upsample * np.arange(-1/2, 1/2, 1/N_tones/upsample)
        
        # load the path information
        # 处理 MATLAB cell array 的情况
        paths = saved_paths[kk]
        coef = saved_mag[kk]
        
        # 如果是嵌套数组，需要展平
        if isinstance(paths, np.ndarray) and paths.ndim > 1:
            paths = paths.flatten()
        if isinstance(coef, np.ndarray) and coef.ndim > 1:
            coef = coef.flatten()
        
        # Sample the ground truth channel frequency response
        freq_resp_l = np.fft.fftshift(sum_exponentials(grid_l, paths * 1e-9, coef))
        freq_resp_h = np.fft.fftshift(sum_exponentials(grid_h, paths * 1e-9, coef))
        
        # Normalization to unit power
        power = np.mean(np.abs(freq_resp_l) ** 2)
        freq_resp_l = np.sqrt(signal_pwr) * freq_resp_l / np.sqrt(power)
        freq_resp_h = np.sqrt(signal_pwr) * freq_resp_h / np.sqrt(power)
        
        # Waveform Design: Power Allocation
        if waveform_net is not None:
            with torch.no_grad():
                h_sq = torch.from_numpy(np.abs(freq_resp_l)**2).float().unsqueeze(0)
                sigma_sq = torch.ones_like(h_sq) * (noise_pwr / signal_pwr)
                alpha = torch.tensor([[alpha_val]]).float()
                p = waveform_net(h_sq, sigma_sq, alpha).squeeze(0).numpy()
                # Apply power allocation: Multiply by sqrt(P * N_tones)
                freq_resp_l = freq_resp_l * np.sqrt(p * N_tones)
                # For high-res ground truth, we also apply it (interpolated if necessary)
                # but for simplicity, let's just use the same P if upsample=1 or handle upsample
                p_h = np.interp(np.linspace(0, 1, len(freq_resp_h)), np.linspace(0, 1, len(p)), p)
                freq_resp_h = freq_resp_h * np.sqrt(p_h * len(p_h))
        
        # add noise
        noise = np.sqrt(noise_pwr / 2) * (np.random.randn(N_tones) + 1j * np.random.randn(N_tones))
        freq_resp_obs = freq_resp_l + noise
        
        # Zero padding the observed channel frequency response
        # MATLAB索引: freq_resp_obs(1:N_tones/2) -> Python: freq_resp_obs[0:N_tones//2]
        # MATLAB索引: freq_resp_obs(end-N_tones/2+1:end) -> Python: freq_resp_obs[-N_tones//2:]
        freq_resp_obs_pad = np.concatenate([
            freq_resp_obs[0:N_tones//2],
            np.zeros((upsample - 1) * N_tones, dtype=complex),
            freq_resp_obs[-N_tones//2:]
        ])
        
        if DISPLAY == 1:
            time_h = np.arange(N_tones * upsample) / ofdm_bw / 2
            dist_h = time_h * C
            
            plt.figure()
            plt.plot(dist_h, np.abs(np.fft.ifft(freq_resp_obs_pad)))
            plt.xlabel("Distance")
            plt.title(f'SNR (dB): {SNR}  Bandwidth (MHz): {int(ofdm_bw / 1e6)}')
            plt.plot(dist_h, np.abs(np.fft.ifft(freq_resp_h)))
            plt.plot(dd * np.ones(100), np.linspace(0, 1.25 * np.max(np.abs(np.fft.ifft(freq_resp_obs_pad))), 100))
            plt.legend(['low resolution', 'high resolution', 'ground truth'])
            
            plt.figure()
            plt.plot(grid_h, np.abs(np.fft.fftshift(freq_resp_obs_pad)))
            plt.xlabel("Distance")
            plt.title(f'SNR (dB): {SNR}  Bandwidth (MHz): {int(ofdm_bw / 1e6)}')
            plt.plot(grid_h, np.abs(np.fft.fftshift(freq_resp_h)))
            plt.legend(['low resolution', 'high resolution'])
            
            plt.show()
            input("Press Enter to continue...")  # 等效于 MATLAB 的 keyboard
        
        cir_pad = np.fft.ifft(freq_resp_obs_pad)
        cirh = np.fft.ifft(freq_resp_h)
        cfrh = freq_resp_h
        
        cir_l[kk, 0, :] = np.real(cir_pad)
        cir_l[kk, 1, :] = np.imag(cir_pad)
        cfr_h[kk, 0, :] = np.real(cfrh)
        cfr_h[kk, 1, :] = np.imag(cfrh)
        cir_h[kk, 0, :] = np.real(cirh)
        cir_h[kk, 1, :] = np.imag(cirh)
    
    dist = saved_dist.reshape(-1, 1)  # 转换为列向量 (N, 1) 以匹配 MATLAB 格式
    savemat(path, {'cir_l': cir_l, 'cir_h': cir_h, 'cfr_h': cfr_h, 'dist': dist})


def load_cell_array(mat_data, key):
    """
    加载 MATLAB cell array 数据
    MATLAB cell array 在 scipy 中被加载为 object 类型的 numpy 数组
    """
    data = mat_data[key]
    # 如果是 object 数组 (cell array)，保持原样返回
    if data.dtype == object:
        # 展平并返回列表
        return [item.flatten() if isinstance(item, np.ndarray) else item 
                for item in data.flatten()]
    else:
        return data.flatten()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate CIR dataset with Waveform Design')
    parser.add_argument('--alpha', type=float, default=None, help='Waveform design alpha (0-1). None for uniform power.')
    parser.add_argument('--bandwidth', type=int, default=40, help='OFDM bandwidth in MHz')
    parser.add_argument('--upsample', type=int, default=2, help='Upsampling rate')
    args = parser.parse_args()

    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 创建 traindata 目录
    if not os.path.exists('traindata'):
        os.makedirs('traindata')
    
    # 加载训练数据
    train_data = loadmat('Pathset_train.mat')
    
    DISPLAY = 0  # 1 for display mode, 0 for generation mode
    ofdm_bw = args.bandwidth * 1e6  # Target ofdm bandwidth
    upsample = args.upsample  # Up-sampling rate for super-resolution
    alpha_val = args.alpha
    suffix = f"_alpha{alpha_val}" if alpha_val is not None else ""
    
    # 提取数据
    saved_dist = train_data['saved_dist'].flatten()
    saved_mag = load_cell_array(train_data, 'saved_mag')
    saved_paths = load_cell_array(train_data, 'saved_paths')
    
    # *_A for CIR enhancement and *_B for ToA estimation stage
    saved_dist_A = saved_dist[0:100000]
    saved_dist_B = saved_dist[100000:]
    saved_mag_A = saved_mag[0:100000]
    saved_mag_B = saved_mag[100000:]
    saved_paths_A = saved_paths[0:100000]
    saved_paths_B = saved_paths[100000:]
    
    ############## Generate low SNR training sets ################
    SNR_list = np.arange(-2.5, 7.5 + 0.1, 0.1)
    save_path = f'traindata/Train_x{upsample}_low_{int(ofdm_bw * 1e-6)}MHz_A{suffix}.mat'
    generate_dataset(saved_dist_A, saved_mag_A, saved_paths_A, SNR_list, ofdm_bw, upsample, save_path, DISPLAY, alpha_val)
    save_path = f'traindata/Train_x{upsample}_low_{int(ofdm_bw * 1e-6)}MHz_B{suffix}.mat'
    generate_dataset(saved_dist_B, saved_mag_B, saved_paths_B, SNR_list, ofdm_bw, upsample, save_path, DISPLAY, alpha_val)
    
    ############## Generate high SNR training sets ################
    SNR_list = np.arange(7.5, 32.5 + 0.1, 0.1)
    save_path = f'traindata/Train_x{upsample}_high_{int(ofdm_bw * 1e-6)}MHz_A{suffix}.mat'
    generate_dataset(saved_dist_A, saved_mag_A, saved_paths_A, SNR_list, ofdm_bw, upsample, save_path, DISPLAY, alpha_val)
    save_path = f'traindata/Train_x{upsample}_high_{int(ofdm_bw * 1e-6)}MHz_B{suffix}.mat'
    generate_dataset(saved_dist_B, saved_mag_B, saved_paths_B, SNR_list, ofdm_bw, upsample, save_path, DISPLAY, alpha_val)
    
    ############## Generate test sets ##############################
    
    if not os.path.exists('testdata'):
        os.makedirs('testdata')
    
    test_data = loadmat('Pathset_test.mat')
    SNR_set = np.arange(0, 30 + 5, 5)  # [0, 5, 10, 15, 20, 25, 30]
    
    saved_dist = test_data['saved_dist'].flatten()
    saved_mag = load_cell_array(test_data, 'saved_mag')
    saved_paths = load_cell_array(test_data, 'saved_paths')
    
    for i in range(len(SNR_set)):
        SNR_list = np.array([SNR_set[i], SNR_set[i]])
        save_path = f'testdata/Test_x{upsample}_{SNR_set[i]}dB_{int(ofdm_bw * 1e-6)}MHz{suffix}.mat'
        generate_dataset(saved_dist, saved_mag, saved_paths, SNR_list, ofdm_bw, upsample, save_path, DISPLAY, alpha_val)
    
    test_data_802 = loadmat('Pathset_test_802.mat')
    saved_dist = test_data_802['saved_dist'].flatten()
    saved_mag = load_cell_array(test_data_802, 'saved_mag')
    saved_paths = load_cell_array(test_data_802, 'saved_paths')
    
    for i in range(len(SNR_set)):
        SNR_list = np.array([SNR_set[i], SNR_set[i]])
        save_path = f'testdata/Test_x{upsample}_{SNR_set[i]}dB_{int(ofdm_bw * 1e-6)}MHz_802{suffix}.mat'
        generate_dataset(saved_dist, saved_mag, saved_paths, SNR_list, ofdm_bw, upsample, save_path, DISPLAY, alpha_val)


if __name__ == '__main__':
    main()

