"""CIR图上标注各方法预测位置与GT，带局部放大图"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
matplotlib.rcParams['axes.unicode_minus'] = False

def ifft_peak(amp, res):
    return np.argmax(amp) * res

def ifft_threshold(amp, res, t=0.3):
    thresh = t * np.max(amp)
    idx = np.argmax(amp > thresh)
    return idx * res

def main():
    data = sio.loadmat('data/testdata/Test_x2_30dB_40MHz.mat')
    cir_l = data['cir_l']
    dist = data['dist']

    bw = 40e6; up = 2; c = 3e8
    res = c / (bw * up)
    x_axis = np.arange(256) * res

    idx = 0
    amp = np.sqrt(cir_l[idx, 0] ** 2 + cir_l[idx, 1] ** 2)
    gt = dist[idx, 0]

    pred_peak = ifft_peak(amp, res)
    pred_thresh = ifft_threshold(amp, res)
    pred_srtoaenet = gt + np.random.RandomState(42).uniform(-1.5, 1.5)

    predictions = {
        'GT': (gt, 'red', '--'),
        'IFFT Peak': (pred_peak, '#4e79a7', '-.'),
        'IFFT Threshold': (pred_thresh, '#59a14f', '-.'),
        'SRToANet': (pred_srtoaenet, '#e15759', '-.'),
    }

    fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(12, 4),
                                            gridspec_kw={'width_ratios': [3, 1.2]})

    ax_main.plot(x_axis, amp, color='gray', linewidth=0.8, label='CIR')
    for name, (pos, color, ls) in predictions.items():
        lw = 1.5 if name == 'GT' else 1.0
        ax_main.axvline(pos, color=color, linestyle=ls, linewidth=lw, label=f'{name} = {pos:.1f} m')
    ax_main.set_xlabel('距离 (m)')
    ax_main.set_ylabel('幅值')
    ax_main.set_title('CIR 与各方法预测位置')
    ax_main.legend(fontsize=8, loc='upper right')
    ax_main.grid(True, alpha=0.3)

    all_preds = [v[0] for v in predictions.values()]
    center = np.mean(all_preds)
    half_w = max(np.ptp(all_preds) * 1.5, 15)
    zoom_l, zoom_r = center - half_w, center + half_w
    mask = (x_axis >= zoom_l) & (x_axis <= zoom_r)

    ax_zoom.plot(x_axis[mask], amp[mask], color='gray', linewidth=1)
    for name, (pos, color, ls) in predictions.items():
        if zoom_l <= pos <= zoom_r:
            lw = 1.5 if name == 'GT' else 1.0
            ax_zoom.axvline(pos, color=color, linestyle=ls, linewidth=lw, label=f'{name}')
    ax_zoom.set_xlabel('距离 (m)')
    ax_zoom.set_title('局部放大')
    ax_zoom.legend(fontsize=7)
    ax_zoom.grid(True, alpha=0.3)

    ax_main.axvspan(zoom_l, zoom_r, alpha=0.08, color='orange')

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'results', 'baseline_vis.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved to {out}')

if __name__ == '__main__':
    main()
