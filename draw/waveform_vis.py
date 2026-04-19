"""不同alpha下的子载波功率分配可视化"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
matplotlib.rcParams['axes.unicode_minus'] = False

from models.waveform_net import WaveformNet

def main():
    n_subcarriers = 128
    net = WaveformNet(n_subcarriers)
    weight_path = 'experiments/waveform_design/waveform_net.w'
    net.load_state_dict(torch.load(weight_path, map_location='cpu'))
    net.eval()

    h_sq = torch.ones(1, n_subcarriers)
    sigma_sq = torch.ones(1, n_subcarriers) * 0.01

    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = ['#4e79a7', '#76b7b2', '#59a14f', '#f28e2b', '#e15759']

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for a, c in zip(alphas, colors):
        alpha_t = torch.tensor([[a]]).float()
        with torch.no_grad():
            p = net(h_sq, sigma_sq, alpha_t)
        ax.plot(range(n_subcarriers), p[0].numpy(), color=c, linewidth=1.5, label=f'α = {a}')

    ax.set_xlabel('子载波索引')
    ax.set_ylabel('功率分配')
    ax.set_title('不同 α 下的子载波功率分配')
    ax.set_ylim(bottom=0, top=0.08)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'results', 'waveform_vis.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved to {out}')

if __name__ == '__main__':
    main()
