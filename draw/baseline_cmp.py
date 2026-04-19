"""基线方法RMSE柱状图：5dB和30dB两组，不同方法不同颜色"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
matplotlib.rcParams['axes.unicode_minus'] = False

def main():
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'baseline.csv')
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    methods = df.iloc[:, 0].tolist()
    rmse_5 = df['5dB'].tolist()
    rmse_30 = df['30dB'].tolist()

    display_names = {
        'Peak Detection': 'IFFT Peak',
        'Thresholding': 'IFFT Threshold',
        'MUSIC_M64': 'MUSIC (M=64)',
        'origin_low': 'SRToANet (Low)',
        'origin_high': 'SRToANet (High)',
    }
    labels = [display_names.get(m, m) for m in methods]
    colors = ['#4e79a7', '#59a14f', '#f28e2b', '#e15759', '#b07aa1']

    x = range(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    from matplotlib.patches import Patch
    bars1 = ax.bar([i - width / 2 for i in x], rmse_5, width, color=colors, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar([i + width / 2 for i in x], rmse_30, width, color=colors, edgecolor='white', linewidth=0.5, alpha=0.6)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_ylabel('RMSE (m)')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(handles=[Patch(facecolor='#555555', label='SNR = 5 dB'),
                       Patch(facecolor='#bbbbbb', label='SNR = 30 dB')])
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('基线方法测距RMSE对比')

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'results', 'baseline_cmp.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved to {out}')

if __name__ == '__main__':
    main()
