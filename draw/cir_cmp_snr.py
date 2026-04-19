"""不同SNR下CIR幅度对比图"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
matplotlib.rcParams['axes.unicode_minus'] = False

def main():
    snrs = [0, 5, 10, 20, 30]
    bw = 40e6; up = 2; c = 3e8
    res = c / (bw * up)
    x_axis = np.arange(256) * res

    fig, axes = plt.subplots(len(snrs), 1, figsize=(8, 2.2 * len(snrs)), sharex=True)
    idx = 0

    for i, snr in enumerate(snrs):
        path = f'data/testdata/Test_x2_{snr}dB_40MHz.mat'
        data = sio.loadmat(path)
        cir_l = data['cir_l']
        dist = data['dist']
        amp = np.sqrt(cir_l[idx, 0] ** 2 + cir_l[idx, 1] ** 2)
        gt = dist[idx, 0]

        axes[i].plot(x_axis, amp, color='steelblue', linewidth=0.8)
        axes[i].axvline(gt, color='red', linestyle='--', linewidth=1, label=f'GT = {gt:.1f} m')
        axes[i].set_ylabel('幅值')
        axes[i].set_title(f'SNR = {snr} dB', fontsize=10)
        axes[i].legend(fontsize=8, loc='upper right')
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('距离 (m)')
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'results', 'cir_cmp_snr.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved to {out}')

if __name__ == '__main__':
    main()
