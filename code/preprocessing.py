from math import gcd
from statistics import mode

import numpy as np
import torch
from einops import rearrange, repeat
from scipy import signal
from scipy.signal import resample_poly


def preprocess(
    data,
    model_config,
    sf,
    labels,
    channels,
    time_axis=-1,
    subtrial_size=1,
    DEBUG=False,
    SUBTRIAL=False,
):
    def extract_subset(
        data,
        labels,
        channels,
        n_trials=20,
        n_channels=5,
        n_samples=4 * sf,
    ):
        for split in data:
            data[split] = data[split][:n_trials, :n_channels, :n_samples]
        labels = labels[:n_trials]
        channels = channels[:n_channels]
        return data, labels, channels

    if DEBUG:
        data, labels, channels = extract_subset(data, labels, channels)

    splits = list(data.keys())

    # Filter signal
    if "filter" in model_config:
        print("Filtering...")
        filter_dict = model_config.get("filter", None)
        if filter_dict:
            # Build filter
            b, a = build_filter(
                highpass=filter_dict["highpass"],
                lowpass=filter_dict["lowpass"],
                order=filter_dict["order"],
                sf=sf,
            )

            for split in splits:
                # Apply filter
                data[split] = np.apply_along_axis(
                    filter_signal,
                    axis=time_axis,
                    arr=data[split],
                    b=b,
                    a=a,
                )

    # Resample
    if "resample" in model_config and model_config["resample"]["sf"] != sf:
        print(f"Resampling...")
        for split in splits:
            data[split] = resample_signal(
                data[split],
                fs_orig=sf,
                fs_target=filter_dict["sf"],
                axis=time_axis,
            )
        sf = filter_dict["sf"]

    # Extract largest central window
    print(f"Extracting largest window...")
    w_size, lengths, min_len_global, max_len_global = compute_largest_window(data, sf)

    for split in splits:
        data[split] = extract_central_window(
            data[split],
            w_size=w_size,
            lengths=lengths[split],
        )

    # Normalize
    means, stds = {}, {}
    if "normalize" in model_config:
        print("Normalizing signals...")
        norm_dict = model_config["normalize"]
        if "zscore" in norm_dict:
            print("Applying zscore normalization...")
        elif "remap" in norm_dict:
            print("Applying remapping normalization...")
        for split in splits:
            if "zscore" in norm_dict:
                data[split], means[split], stds[split] = zscore_sig(
                    data[split],
                    axis=time_axis,
                )
            elif "remap" in norm_dict:
                data[split] = remap_sig(
                    data[split],
                    min=norm_dict["remap"]["min"],
                    max=norm_dict["remap"]["max"],
                    axis=time_axis,
                )

    # Clip
    if "clip_std" in model_config:
        print("Applying clip_std clipping...")
        scale = model_config["clip_std"]["scale"]
        for split in splits:
            if split not in stds:
                stds[split] = np.nanstd(data[split], axis=time_axis, keepdims=True)
            if "zscore" in norm_dict:
                c_scale = scale
            else:
                c_scale = scale * stds[split]
            data[split] = np.clip(
                data[split],
                -c_scale,
                c_scale,
            )

    # Extract subtrials
    trials = {}
    n_subtrials_per_trial = {}
    if SUBTRIAL:
        for split in splits:
            (
                data[split],
                trials[split],
                n_subtrials_per_trial[split],
            ) = create_subtrials(
                data[split],
                subtrial_samples=subtrial_size * sf,
            )

    # Get final shape
    _, n_channels, n_times = data["train"].shape
    data_spec = {"n_channels": n_channels, "n_times": n_times, "sf": sf}
    return data, data_spec, labels, channels, trials, n_subtrials_per_trial


def create_subtrials(x, subtrial_samples):
    x = torch.tensor(x)
    n_trials, n_channels, n_times = x.shape

    n_subtrials_per_trial = n_times // subtrial_samples
    trimmed_len = n_subtrials_per_trial * subtrial_samples

    x = x[:, :, :trimmed_len]

    x = rearrange(
        x,
        "trial chan (window time) -> (trial window) chan time",
        time=subtrial_samples,
    )

    trials = repeat(
        torch.arange(n_trials),
        "trial -> (trial window)",
        window=n_subtrials_per_trial,
    )
    return x.numpy(), trials.numpy(), n_subtrials_per_trial


def subtrials2trials(
    subtrial_predictions,
    trials,
    n_subtrials_per_trial,
    modality="mode",
):
    def reduce(y):
        match modality:
            case "mode":
                # print(f"subtrial predictions: {y}")
                # print(f"trial predictions: {mode(y)}")
                return mode(y)
            case _:
                raise ValueError(f"Unrecognized modality {modality}")

    n_tot_predictions = len(subtrial_predictions)
    n_trials = n_tot_predictions // n_subtrials_per_trial
    start = 0
    trial_predictions = []
    for t in range(n_trials):
        stop = start + n_subtrials_per_trial
        trial_predictions.append(reduce(subtrial_predictions[start:stop]))
        start = stop
    return trial_predictions


def remap_sig(x, min=-1.0, max=1.0, axis=-1, eps=1e-8):
    x_min = np.min(x, axis=axis, keepdims=True)
    x_max = np.max(x, axis=axis, keepdims=True)

    denom = x_max - x_min
    x_scaled = (x - x_min) / (denom + eps)
    x_remapped = x_scaled * (max - min) + min

    return x_remapped


def resample_signal(x, fs_orig, fs_target, axis=-1):
    g = gcd(int(fs_orig), int(fs_target))
    up = int(fs_target // g)
    down = int(fs_orig // g)
    return resample_poly(x, up=up, down=down, axis=axis)


def build_filter(highpass, lowpass, order, sf):
    [b, a] = signal.butter(N=order, Wn=[highpass, lowpass], btype="bandpass", fs=sf)
    return b, a


def filter_signal(x, b, a):
    # Bandpass
    first_invalid_idx = np.where(~np.isnan(x))[0][-1] + 1
    x[:first_invalid_idx] = signal.filtfilt(b, a, x[:first_invalid_idx], padlen=100)
    return x


def compute_largest_window(data, sf):
    min_len_global, max_len_global = np.inf, 0
    lengths = {}
    for split in data:
        lengths[split], min_len, max_len = compute_len(data[split])
        min_len_global = min(min_len, min_len_global)
        max_len_global = max(max_len, max_len_global)
    w_size = int((min_len_global // sf) * sf)  # Largest valid window multiple of sf
    return w_size, lengths, min_len_global, max_len_global


def compute_len(x):
    trials, channels, times = x.shape
    lengths = np.empty((trials, channels))
    for t in range(trials):
        for c in range(channels):
            lengths[t, c] = np.where(~np.isnan(x[t, c]))[0][-1] + 1
    return lengths, np.min(lengths), np.max(lengths)


def zscore_sig(x, means=None, stds=None, eps=1e-8, axis=-1):
    if not means:
        means = np.nanmean(x, axis=axis, keepdims=True)
    if not stds:
        stds = np.nanstd(x, axis=axis, keepdims=True)

    x_norm = (x - means) / (stds + eps)
    return x_norm, means, stds


def zscore_and_clip_eeg(x, clip_std=15.0, eps=1e-8, means=None, stds=None):
    if not means:
        means = np.nanmean(x, axis=-1, keepdims=True)
    if not stds:
        stds = np.nanstd(x, axis=-1, keepdims=True)

    x_norm = (x - means) / (stds + eps)

    x_norm = np.clip(x_norm, -clip_std * stds, clip_std * stds)

    return x_norm


def extract_central_window_1d(x, w_size, length):
    length = int(length)

    if length < w_size:
        raise ValueError(f"length={length} is smaller than w_size={w_size}")

    start = (length - w_size) // 2
    end = start + w_size

    return x[start:end]


def extract_central_window(x, w_size, lengths):

    n_trials, n_channels, _ = x.shape
    out = np.empty((n_trials, n_channels, w_size), dtype=x.dtype)

    for i in range(n_trials):
        for j in range(n_channels):
            out[i, j] = extract_central_window_1d(
                x[i, j],
                w_size=w_size,
                length=lengths[i, j],
            )

    return out
