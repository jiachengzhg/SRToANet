"""消融实验：2x2折线图 (coarse/fine × custom/802)"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
matplotlib.rcParams['axes.unicode_minus'] = False

METHOD_DISPLAY = {
    'origin_low': 'Baseline (Low)',
    'origin_high': 'Baseline (High)',
    'attention_low': '+Attention (Low)',
    'attention_high': '+Attention (High)',
    'e2e_low': '+E2E (Low)',
    'e2e_high': '+E2E (High)',
    'both_low': '+Both (Low)',
    'both_high': '+Both (High)',
}

COLORS = {
    'origin_low': '#4e79a7', 'origin_high': '#4e79a7',
    'attention_low': '#59a14f', 'attention_high': '#59a14f',
    'e2e_low': '#f28e2b', 'e2e_high': '#f28e2b',
    'both_low': '#e15759', 'both_high': '#e15759',
}

STYLES = {}
for k in METHOD_DISPLAY:
    STYLES[k] = '--' if '_low' in k else '-'

MARKERS = {}
for k in METHOD_DISPLAY:
    MARKERS[k] = 's' if '_low' in k else 'o'

def load(name):
    path = os.path.join(os.path.dirname(__file__), 'data', name)
    df = pd.read_csv(path, encoding='utf-8-sig')
    return df

def plot_panel(ax, df, title):
    methods = df.iloc[:, 0].tolist()
    snrs = [int(c.replace('dB', '')) for c in df.columns[1:]]
    for m in methods:
        row = df[df.iloc[:, 0] == m]
        vals = row.iloc[0, 1:].values.astype(float)
        ax.plot(snrs, vals, color=COLORS[m], linestyle=STYLES[m],
                marker=MARKERS[m], markersize=5, linewidth=1.5,
                label=METHOD_DISPLAY[m])
    ax.set_title(title, fontsize=10)
    ax.set_xticks(snrs)
    ax.set_xticklabels([f'{s} dB' for s in snrs])
    ax.grid(True, alpha=0.3)

def main():
    files = {
        ('Coarse', '自定义信道'): 'coarse_custom.csv',
        ('Coarse', '802.15.4a 信道'): 'coarse_802.csv',
        ('Fine', '自定义信道'): 'fine_custom.csv',
        ('Fine', '802.15.4a 信道'): 'fine_802.csv',
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = [
        ((0, 0), ('Coarse', '自定义信道')),
        ((0, 1), ('Coarse', '802.15.4a 信道')),
        ((1, 0), ('Fine', '自定义信道')),
        ((1, 1), ('Fine', '802.15.4a 信道')),
    ]

    for (r, c_), key in panels:
        df = load(files[key])
        title = f'{key[0]} — {key[1]}'
        plot_panel(axes[r][c_], df, title)

    axes[1][0].set_xlabel('SNR')
    axes[1][1].set_xlabel('SNR')
    axes[0][0].set_ylabel('RMSE (m)')
    axes[1][0].set_ylabel('RMSE (m)')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10,
              bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = os.path.join(os.path.dirname(__file__), 'results', 'attention_e2e.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved to {out}')

if __name__ == '__main__':
    main()
