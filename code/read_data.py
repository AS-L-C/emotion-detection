import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
print(PROJECT_DIR)


def read(project_dir=PROJECT_DIR):
    # Define paths
    paths = {}
    paths["repo"] = project_dir
    paths["data"] = paths["repo"] / "assets/data.npz"
    paths["trial_info"] = paths["repo"] / "assets/trial_info_train.csv"
    paths["channels"] = paths["repo"] / "metadata" / "channels.json"
    paths["results"] = paths["repo"] / "results"
    paths["configs"] = paths["repo"] / "configs" / "models.yml"

    # Read eeg data
    data = np.load(paths["data"])
    data = {key: data[key] for key in data.files}
    splits = list(data.keys())
    sf = 200

    # Set np.inf to np.nan
    for split in splits:
        data[split][np.isinf(data[split])] = np.nan

    # Read trial info
    trial_info = pd.read_csv(paths["trial_info"])
    labels, labels_names = pd.factorize(trial_info["emotion_label"])
    labels = labels.tolist()
    label2id = {label: i for i, label in enumerate(labels_names)}
    id2label = {i: label for i, label in enumerate(labels_names)}

    # Read metadata
    with open(paths["channels"], "r") as f:
        channels = json.load(f)["channels"]

    # Read configuration files
    with open(paths["configs"], "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f) or {}

    return data, labels, label2id, id2label, trial_info, channels, sf, configs, paths
