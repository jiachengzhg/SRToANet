'''
Show the CIR of the dataset
usage: python show_cir.py --mat <path_to_mat_file> --id <sample_id> --bandwidth <bandwidth> --upsample <upsample_rate> --snr <snr> --alpha <alpha>
'''

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat

# Add root to sys.path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.waveform_net import WaveformNet


def sum_exponentials(freq_grid, paths, coef):
    """
    计算指数和的频率响应
    """
    N = len(paths)
    freq_resp = np.zeros(freq_grid.shape, dtype=complex)
    for i in range(N):
        freq_resp = freq_resp + coef[i] * np.exp(-1j * 2 * np.pi * freq_grid * paths[i])
    return freq_resp


def load_cell_array(mat_data, key):
    """
    加载 MATLAB cell array 数据
    MATLAB cell array 在 scipy 中被加载为 object 类型的 numpy 数组
    """
    data = mat_data[key]
    if data.dtype == object:
        return [item.flatten() if isinstance(item, np.ndarray) else item
                for item in data.flatten()]
    return data.flatten()


def get_waveform_net(n_tones, alpha_val):
    if alpha_val is None:
        return None

    waveform_net = WaveformNet(n_tones)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_model_path = os.path.join(
        repo_root,
        "experiments",
        "waveform_design",
        "waveform_net.w",
    )
    fallback_model_path = os.path.join(
        repo_root,
        "/data2/jiacheng.zhang/SRToANet/experiments/waveform_design/waveform_net.w",
    )

    model_path = default_model_path if os.path.exists(default_model_path) else fallback_model_path
    if os.path.exists(model_path):
        waveform_net.load_state_dict(torch.load(model_path, map_location="cpu"))
        waveform_net.eval()
        print(f"Loaded WaveformNet with alpha={alpha_val}")
        return waveform_net

    print(f"Warning: WaveformNet weights not found at {model_path}. Using uniform power.")
    return None


def compute_cir_from_sample(mat_path, sample_id, ofdm_bw, upsample, snr_db, alpha_val=None):
    data = loadmat(mat_path)
    if "saved_dist" not in data or "saved_mag" not in data or "saved_paths" not in data:
        raise KeyError(f"{mat_path} 缺少 saved_dist / saved_mag / saved_paths")

    saved_dist = data["saved_dist"].flatten()
    saved_mag = load_cell_array(data, "saved_mag")
    saved_paths = load_cell_array(data, "saved_paths")

    if sample_id < 0 or sample_id >= len(saved_dist):
        raise IndexError(f"{mat_path} sample_id={sample_id} 超出范围 (0~{len(saved_dist)-1})")

    tones_gap = 312.5e3
    n_tones = round(ofdm_bw / tones_gap)

    dd = saved_dist[sample_id]
    grid_l = ofdm_bw * np.arange(-1/2, 1/2, 1/n_tones)
    grid_h = ofdm_bw * upsample * np.arange(-1/2, 1/2, 1/n_tones/upsample)

    paths = saved_paths[sample_id]
    coef = saved_mag[sample_id]
    if isinstance(paths, np.ndarray) and paths.ndim > 1:
        paths = paths.flatten()
    if isinstance(coef, np.ndarray) and coef.ndim > 1:
        coef = coef.flatten()

    freq_resp_l = np.fft.fftshift(sum_exponentials(grid_l, paths * 1e-9, coef))
    freq_resp_h = np.fft.fftshift(sum_exponentials(grid_h, paths * 1e-9, coef))

    signal_pwr_db = 10
    signal_pwr = 10 ** (signal_pwr_db / 10)
    power = np.mean(np.abs(freq_resp_l) ** 2)
    freq_resp_l = np.sqrt(signal_pwr) * freq_resp_l / np.sqrt(power)
    freq_resp_h = np.sqrt(signal_pwr) * freq_resp_h / np.sqrt(power)

    waveform_net = get_waveform_net(n_tones, alpha_val)
    if waveform_net is not None:
        import torch
        with torch.no_grad():
            h_sq = torch.from_numpy(np.abs(freq_resp_l) ** 2).float().unsqueeze(0)
            noise_pwr_db = signal_pwr_db - snr_db
            noise_pwr = 10 ** (noise_pwr_db / 10)
            sigma_sq = torch.ones_like(h_sq) * (noise_pwr / signal_pwr)
            alpha = torch.tensor([[alpha_val]]).float()
            p = waveform_net(h_sq, sigma_sq, alpha).squeeze(0).numpy()
            freq_resp_l = freq_resp_l * np.sqrt(p * n_tones)
            p_h = np.interp(np.linspace(0, 1, len(freq_resp_h)), np.linspace(0, 1, len(p)), p)
            freq_resp_h = freq_resp_h * np.sqrt(p_h * len(p_h))

    noise_pwr_db = signal_pwr_db - snr_db
    noise_pwr = 10 ** (noise_pwr_db / 10)
    noise = np.sqrt(noise_pwr / 2) * (np.random.randn(n_tones) + 1j * np.random.randn(n_tones))
    freq_resp_obs = freq_resp_l + noise

    freq_resp_obs_pad = np.concatenate([
        freq_resp_obs[0:n_tones // 2],
        np.zeros((upsample - 1) * n_tones, dtype=complex),
        freq_resp_obs[-n_tones // 2:]
    ])

    cir_low = np.fft.ifft(freq_resp_obs_pad)
    cir_high = np.fft.ifft(freq_resp_h)

    C = 3e8
    time_h = np.arange(n_tones * upsample) / ofdm_bw / 2
    dist_h = time_h * C

    return dist_h, dd, cir_low, cir_high


def parse_items(args):
    items = []
    if args.item:
        for raw in args.item:
            parts = raw.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"--item 需要 path:sample_id[:alpha] 格式, 收到: {raw}")
            path, idx = parts[0], parts[1]
            alpha = parts[2] if len(parts) == 3 else None
            items.append((path, int(idx), float(alpha) if alpha is not None else None))
        return items

    if args.mat is None or args.id is None:
        raise ValueError("请使用 --item 或同时提供 --mat 和 --id")
    if len(args.mat) != len(args.id):
        raise ValueError("--mat 和 --id 数量必须一致")
    if args.alpha_list is not None and len(args.alpha_list) != len(args.mat):
        raise ValueError("--alpha-list 数量必须与 --mat 相同")

    for i, (path, idx) in enumerate(zip(args.mat, args.id)):
        alpha = args.alpha_list[i] if args.alpha_list is not None else None
        items.append((path, idx, alpha))
    return items


def main():
    parser = argparse.ArgumentParser(description="可视化 CIR（参考 CIR_Generation.py 的变换逻辑）")
    parser.add_argument("--item", action="append", default=[],
                        help="单项格式: /path/to/file.mat:sample_id，可重复指定")
    parser.add_argument("--mat", nargs="*", default=None, help="多个 .mat 文件路径")
    parser.add_argument("--id", nargs="*", type=int, default=None, help="对应的 sample_id 列表")
    parser.add_argument("--bandwidth", type=int, default=40, help="OFDM bandwidth in MHz")
    parser.add_argument("--upsample", type=int, default=2, help="Upsampling rate")
    parser.add_argument("--snr", type=float, default=10.0, help="SNR (dB)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="全局 Waveform design alpha (0-1). None for uniform power.")
    parser.add_argument("--alpha-list", nargs="*", type=float, default=None,
                        help="与 --mat 对应的 alpha 列表（可选）")
    parser.add_argument("--resolution", choices=["high", "low", "both"], default="high",
                        help="绘制分辨率: high/low/both")
    parser.add_argument("--no-gt", action="store_true", help="不绘制 ground truth 距离标记")
    parser.add_argument("--save", type=str, default=None, help="保存图像路径（可选）")
    args = parser.parse_args()

    items = parse_items(args)
    show_gt = not args.no_gt
    show_low = args.resolution in ("low", "both")
    show_high = args.resolution in ("high", "both")

    ofdm_bw = args.bandwidth * 1e6
    upsample = args.upsample

    plt.figure()
    gt_seen = []
    for mat_path, sample_id, item_alpha in items:
        alpha_val = item_alpha if item_alpha is not None else args.alpha
        dist_h, dd, cir_low, cir_high = compute_cir_from_sample(
            mat_path, sample_id, ofdm_bw, upsample, args.snr, alpha_val
        )
        label_suffix = f" a={alpha_val}" if alpha_val is not None else ""
        label_base = f"{os.path.basename(mat_path)}#{sample_id}{label_suffix}"
        if show_low:
            plt.plot(dist_h, np.abs(cir_low), label=f"{label_base} low")
        if show_high:
            plt.plot(dist_h, np.abs(cir_high), label=f"{label_base} high")
        if show_gt:
            if not any(np.isclose(dd, seen, rtol=0, atol=1e-6) for seen in gt_seen):
                gt_seen.append(dd)
                plt.axvline(dd, linestyle="--", label=f"{label_base} gt")

    plt.xlabel("Distance")
    plt.title(f"SNR (dB): {args.snr}  Bandwidth (MHz): {args.bandwidth}")
    plt.legend()
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()

