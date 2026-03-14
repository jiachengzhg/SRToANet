'''
Plot the result of the waveform net
会加载训练好的 waveform_net.w, 画出不同alpha时的功率分配
usage: python plot_waveform.py
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from models.waveform_net import WaveformNet

def visualize():
    n_subcarriers = 128
    net = WaveformNet(n_subcarriers)
    # 请确保路径正确，如果报错请检查文件路径
    net.load_state_dict(torch.load('experiments/waveform_design/waveform_net.w', map_location='cpu'))
    net.eval()

    # 模拟一个简单的信道 (全 1)
    h_sq = torch.ones(1, n_subcarriers)
    sigma_sq = torch.ones(1, n_subcarriers) * 0.01

    plt.figure(figsize=(10, 6))
    
    # 1. 修改 Alpha 列表
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # 2. 设置对应的颜色 (5种颜色)
    colors = ['blue', 'cyan', 'green', 'orange', 'red']

    for a, c in zip(alphas, colors):
        alpha_tensor = torch.tensor([[a]]).float()
        with torch.no_grad():
            p = net(h_sq, sigma_sq, alpha_tensor)
        
        # 动态生成标签
        label_text = f'Alpha={a}'
        plt.plot(p[0].numpy(), color=c, label=label_text, linewidth=2)

    plt.xlabel('Subcarrier Index')
    plt.ylabel('Power Allocation')
    plt.title('Waveform Design: Power Allocation vs Alpha')
    
    # 3. 设置纵轴区间到 0.0 - 0.5
    plt.ylim(0.0, 0.06)
    
    plt.legend()
    plt.grid(True)
    plt.savefig('waveform_comparison.png')
    print("Comparison plot saved as 'waveform_comparison.png'")
    plt.show()

if __name__ == "__main__":
    visualize()