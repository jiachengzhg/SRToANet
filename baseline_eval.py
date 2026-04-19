import numpy as np
import scipy.io as sio
import argparse
import os
from scipy.fft import fft
from scipy.signal import find_peaks

def extract_active_cfr_from_cir(cir_complex, n_tones):
    cfr_full = fft(cir_complex)
    # Keep active tones in monotonic order from negative to positive.
    return np.concatenate([cfr_full[-n_tones//2:], cfr_full[:n_tones//2]])


def reorder_cfr_to_monotonic(cfr_snapshots, n_tones):
    """
    Reorder CFR from legacy FFT-shifted layout to monotonic (-f to +f).
    Input shape can be (N_tones,) or (M, N_tones).
    """
    if cfr_snapshots.ndim == 1:
        return np.concatenate([cfr_snapshots[-n_tones//2:], cfr_snapshots[:n_tones//2]])
    return np.concatenate([cfr_snapshots[:, -n_tones//2:], cfr_snapshots[:, :n_tones//2]], axis=1)


def build_music_snapshots(data, sample_idx, cir_l, n_tones, use_snapshots):
    if use_snapshots and 'cfr_l_snap' in data:
        real = data['cfr_l_snap'][sample_idx, :, 0, :]
        imag = data['cfr_l_snap'][sample_idx, :, 1, :]
        cfr_snapshots = real + 1j * imag
        if cfr_snapshots.ndim == 1:
            cfr_snapshots = cfr_snapshots[np.newaxis, :]
        return reorder_cfr_to_monotonic(cfr_snapshots, n_tones)

    if use_snapshots and 'cir_l_snap' in data:
        real = data['cir_l_snap'][sample_idx, :, 0, :]
        imag = data['cir_l_snap'][sample_idx, :, 1, :]
        cir_snapshots = real + 1j * imag
        if cir_snapshots.ndim == 1:
            cir_snapshots = cir_snapshots[np.newaxis, :]

        cfr_snapshots = np.zeros((cir_snapshots.shape[0], n_tones), dtype=complex)
        for sym_idx in range(cir_snapshots.shape[0]):
            cfr_snapshots[sym_idx, :] = extract_active_cfr_from_cir(cir_snapshots[sym_idx], n_tones)
        return cfr_snapshots

    # Fallback for legacy data: use single snapshot (M=1).
    real = cir_l[sample_idx, 0, :]
    imag = cir_l[sample_idx, 1, :]
    cir_complex = real + 1j * imag
    cfr_single = extract_active_cfr_from_cir(cir_complex, n_tones)
    return cfr_single[np.newaxis, :]


def resolve_music_k(data, music_k_arg, sample_idx):
    if music_k_arg == 'gt':
        if 'gt_num_paths' not in data:
            raise ValueError("music_k=gt requires 'gt_num_paths' in dataset.")
        gt_paths = data['gt_num_paths'].flatten()
        return int(gt_paths[sample_idx])
    return int(music_k_arg)


def music_estimate(cfr_snapshots, n_tones, sub_size, search_range, bw, k_paths=3, fb_smoothing=True):
    """
    Unified MUSIC estimation for both single and multi snapshots.
    """
    if cfr_snapshots.ndim == 1:
        cfr_snapshots = cfr_snapshots[np.newaxis, :]

    if cfr_snapshots.shape[1] != n_tones:
        raise ValueError(f"Expected n_tones={n_tones}, got snapshots shape={cfr_snapshots.shape}")

    # 1. Build covariance with subarray averaging and snapshot averaging.
    num_subarrays = n_tones - sub_size + 1
    R = np.zeros((sub_size, sub_size), dtype=complex)

    for sym_idx in range(cfr_snapshots.shape[0]):
        cfr = cfr_snapshots[sym_idx]
        for i in range(num_subarrays):
            x = cfr[i: i + sub_size]
            R += np.outer(x, x.conj())

    R /= (cfr_snapshots.shape[0] * num_subarrays)

    if fb_smoothing:
        J = np.fliplr(np.eye(sub_size))
        R = (R + J @ R.conj() @ J) / 2.0

    # 2. Eigen decomposition
    _, eig_vecs = np.linalg.eigh(R)

    # Keep K valid for all configs.
    k_paths = max(1, min(k_paths, sub_size - 1))
    Un = eig_vecs[:, :sub_size - k_paths]

    # 3. Pseudospectrum search
    C = 3e8
    distances = np.arange(search_range[0], search_range[1], 0.1)
    if distances.size == 0:
        distances = np.array([search_range[0]])

    df = (bw * 1e6) / n_tones
    freq_indices = np.arange(sub_size)
    taus = distances / C
    A = np.exp(-1j * 2 * np.pi * df * np.outer(freq_indices, taus))

    projection = Un.conj().T @ A
    denominator = np.sum(np.abs(projection) ** 2, axis=0)
    denominator = np.maximum(denominator, 1e-12)
    spectrum = 1.0 / denominator

    # 4. ToA: select the earliest significant peak.
    peaks, _ = find_peaks(spectrum, prominence=np.max(spectrum) * 0.1)

    if len(peaks) > 0:
        peak_idx = peaks[0]
    else:
        peak_idx = np.argmax(spectrum)

    return distances[peak_idx]


def run_baselines():
    parser = argparse.ArgumentParser(description='Baseline ToA Estimation Methods')
    parser.add_argument('--mat_path', type=str, required=True, help='Path to test .mat file')
    parser.add_argument('--bw', type=float, default=40.0, help='Bandwidth in MHz')
    parser.add_argument('--up', type=int, default=2, help='Upsample rate')
    parser.add_argument('--threshold', type=float, default=0.3, help='Threshold for peak detection')
    parser.add_argument(
        '--music_k',
        type=str,
        default='3',
        help="Number of signal paths for MUSIC, or 'gt' to use gt_num_paths",
    )
    parser.add_argument('--music_sub_size', type=int, default=0, help='Subarray size for MUSIC (0 means n_tones//2)')
    parser.add_argument(
        '--music_use_snapshots',
        choices=['auto', 'on', 'off'],
        default='auto',
        help='Use multi-snapshot fields for MUSIC: auto/on/off',
    )
    parser.add_argument('--music_fb_smoothing', dest='music_fb_smoothing', action='store_true', help='Enable forward-backward smoothing')
    parser.add_argument('--no_music_fb_smoothing', dest='music_fb_smoothing', action='store_false', help='Disable forward-backward smoothing')
    parser.set_defaults(music_fb_smoothing=True)

    args = parser.parse_args()

    if not os.path.exists(args.mat_path):
        print(f"File not found: {args.mat_path}")
        return

    data = sio.loadmat(args.mat_path)
    cir_l = data['cir_l']  # (N, 2, L)
    dist_gt = data['dist'].flatten()  # (N,)

    N, _, L = cir_l.shape
    C = 3e8
    n_tones = int(args.bw / 0.3125)  # 40MHz -> 128
    sub_size = args.music_sub_size if args.music_sub_size > 0 else n_tones // 2
    sub_size = max(2, min(sub_size, n_tones))

    has_cfr_snap = 'cfr_l_snap' in data
    has_cir_snap = 'cir_l_snap' in data
    if args.music_use_snapshots == 'auto':
        use_snapshots = has_cfr_snap or has_cir_snap
    elif args.music_use_snapshots == 'on':
        if not (has_cfr_snap or has_cir_snap):
            print("Requested snapshot mode, but no cfr_l_snap/cir_l_snap in data.")
            return
        use_snapshots = True
    else:
        use_snapshots = False

    err_ifft_peak = []
    err_ifft_threshold = []
    err_music = []

    print(f"Evaluating {args.mat_path} ({N} samples)...")
    dist_res = C / (args.bw * 1e6 * args.up)
    print(
        f"MUSIC config: K={args.music_k}, sub_size={sub_size}, "
        f"fb_smoothing={args.music_fb_smoothing}, use_snapshots={use_snapshots}"
    )

    if args.music_k == 'gt' and 'gt_num_paths' not in data:
        print("music_k=gt requires gt_num_paths in dataset, but it was not found.")
        return

    for i in range(min(N, 500)):
        real = cir_l[i, 0, :]
        imag = cir_l[i, 1, :]
        cir_complex = real + 1j * imag
        amp = np.abs(cir_complex)

        # 1. IFFT Peak Detection
        idx_peak = np.argmax(amp)
        est_peak = idx_peak * dist_res
        err_ifft_peak.append(est_peak - dist_gt[i])

        # 2. IFFT Thresholding
        threshold_val = args.threshold * np.max(amp)
        indices = np.where(amp > threshold_val)[0]
        idx_threshold = indices[0] if len(indices) > 0 else 0
        est_threshold = idx_threshold * dist_res
        err_ifft_threshold.append(est_threshold - dist_gt[i])

        # 3. MUSIC
        cfr_snapshots = build_music_snapshots(data, i, cir_l, n_tones, use_snapshots)
        k_paths = resolve_music_k(data, args.music_k, i)
        search_min = max(0, est_peak - 20)
        search_max = est_peak + 20
        est_music = music_estimate(
            cfr_snapshots,
            n_tones,
            sub_size,
            (search_min, search_max),
            args.bw,
            k_paths=k_paths,
            fb_smoothing=args.music_fb_smoothing,
        )
        err_music.append(est_music - dist_gt[i])

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{min(N, 500)} samples...")

    def calc_rmse(err_list):
        return np.sqrt(np.mean(np.array(err_list)**2))

    print("\n" + "="*40)
    print(f"Results for: {os.path.basename(args.mat_path)}")
    print(f"IFFT Peak Detection RMSE:    {calc_rmse(err_ifft_peak):.4f} m")
    print(f"IFFT Thresholding RMSE:      {calc_rmse(err_ifft_threshold):.4f} m")
    print(f"MUSIC Algorithm RMSE:        {calc_rmse(err_music):.4f} m")
    print("="*40)

if __name__ == "__main__":
    run_baselines()