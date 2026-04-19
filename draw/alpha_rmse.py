"""横向柱状图：不同alpha值对应的RMSE"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
matplotlib.rcParams['axes.unicode_minus'] = False

def main():
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'alpha_rmse.csv')
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    alphas = df['alpha'].tolist()
    rmses = df['RMSE'].tolist()
    labels = [f'α = {a}' for a in alphas]
    colors = ['#4e79a7', '#76b7b2', '#59a14f', '#f28e2b', '#e15759']

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(labels, rmses, color=colors[:len(alphas)], edgecolor='white', height=0.5)

    for bar, val in zip(bars, rmses):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.set_xlabel('RMSE (m)')
    ax.set_title('波形设计 α 扫描 — TOA估计RMSE')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'results', 'alpha_rmse.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved to {out}')

if __name__ == '__main__':
    main()
