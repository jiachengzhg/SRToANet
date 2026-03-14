'''
ISAC指的是通感一体化 (Integrated Sensing and Communication)
用于衡量不同 alpha 时的通信性能和感知性能
usage: python evaluate_isac.py
会加载训练好的 waveform_net.w
TODO: 这两个性能的衡量方式可能还需要再研究
'''

import torch
import numpy as np
from models.waveform_net import WaveformNet

def evaluate():
    n_subcarriers = 128
    net = WaveformNet(n_subcarriers)
    try:
        net.load_state_dict(torch.load('experiments/waveform_design/waveform_net.w', map_location='cpu'))
    except:
        print("Error: WaveformNet weights not found.")
        return
    net.eval()

    # 模拟 1000 个随机信道环境
    n_samples = 1000
    h_sq = torch.exp(torch.randn(n_samples, n_subcarriers) * 0.5)
    sigma_sq = torch.ones(n_samples, n_subcarriers) * 0.01 # 固定噪声功率

    alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    print(f"{'Alpha':<10} | {'Spectral Efficiency (bits/s/Hz)':<35} | {'Sensing Reward (Var)':<20}")
    print("-" * 75)

    for a in alphas:
        alpha_tensor = torch.ones(n_samples, 1) * a
        with torch.no_grad():
            p = net(h_sq, sigma_sq, alpha_tensor)
            
            # 1. 计算通信速率 (Shannon)
            snr = (p * h_sq) / sigma_sq
            se = torch.log2(1 + snr).mean()
            
            # 2. 计算感知性能 (方差)
            freq_grid = torch.linspace(-1, 1, n_subcarriers)
            f_mean = (p * freq_grid).sum(dim=-1, keepdim=True)
            f_var = (p * (freq_grid - f_mean)**2).sum(dim=-1, keepdim=True).mean()

            print(f"{a:<10.1f} | {se:<35.4f} | {f_var:<20.4f}")

if __name__ == "__main__":
    evaluate()

