"""低分辨率 vs 高分辨率 CIR 对比图，带GT时延红色虚线"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
matplotlib.rcParams['axes.unicode_minus'] = False

def main():
    data = sio.loadmat('data/testdata/Test_x2_30dB_40MHz.mat')
    cir_l = data['cir_l']  # (N, 2, 256)
    cir_h = data['cir_h']
    dist = data['dist']    # (N, 1) in meters

    bw = 40e6
    up = 2
    c = 3e8
    res = c / (bw * up)
    x_axis = np.arange(256) * res  # meters

    idx = 0
    amp_l = np.sqrt(cir_l[idx, 0] ** 2 + cir_l[idx, 1] ** 2)
    amp_h = np.sqrt(cir_h[idx, 0] ** 2 + cir_h[idx, 1] ** 2)
    gt_dist = dist[idx, 0]

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    axes[0].plot(x_axis, amp_l, color='steelblue', linewidth=1)
    axes[0].axvline(gt_dist, color='red', linestyle='--', linewidth=1.2, label=f'GT = {gt_dist:.1f} m')
    axes[0].set_ylabel('幅值')
    axes[0].set_title('低分辨率 CIR')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_axis, amp_h, color='darkorange', linewidth=1)
    axes[1].axvline(gt_dist, color='red', linestyle='--', linewidth=1.2, label=f'GT = {gt_dist:.1f} m')
    axes[1].set_ylabel('幅值')
    axes[1].set_xlabel('距离 (m)')
    axes[1].set_title('高分辨率 CIR')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'results', 'cir_cmp_landh.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved to {out}')

if __name__ == '__main__':
    main()
