import importlib
import os
import random
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from transformers import set_seed

PROJECT_DIR = Path(__file__).resolve().parent.parent


def import_modules(module_list):
    if isinstance(module_list, str):
        module_list = [module_list]
    imports = []
    for module_name in module_list:
        imports.append(importlib.reload(importlib.import_module(module_name)))
        print(f"Imported latest version of '{module_name}' module")
    return imports[0] if len(imports) == 1 else tuple(imports)


def set_seeds(seed=0, deterministic=True):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    return


def save_preds(
    model: str,
    ypred_test,
    id2label: dict,
    results_dir: str | Path = PROJECT_DIR / "results",
) -> Path:
    output_dir = Path(results_dir) / "test_predictions" / model
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "best_model_predictions.csv"

    if isinstance(ypred_test, torch.Tensor):
        ypred_test = ypred_test.detach().cpu().numpy()

    ypred_test = np.asarray(ypred_test)

    predicted_labels = [id2label[int(pred)] for pred in ypred_test]

    df = pd.DataFrame(
        {
            "predicted_label": predicted_labels,
        }
    )

    df.to_csv(output_path, index=False)

    return output_path


def _collect_labels_from_dataloader(dataloader):
    ys = []

    for batch in dataloader:
        _, y = batch

        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        ys.append(np.asarray(y))

    return np.concatenate(ys)


def _get_majority_class_from_train(y_train):
    values, counts = np.unique(y_train, return_counts=True)
    majority_class = values[np.argmax(counts)]
    return majority_class


def _compute_baseline_scores(y_true, majority_class, n_classes: int):
    y_majority = np.full_like(y_true, fill_value=majority_class)
    majority_accuracy = accuracy_score(y_true, y_majority)
    majority_balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_majority,
    )
    random_accuracy = 1.0 / n_classes
    random_balanced_accuracy = 1.0 / n_classes

    return {
        "majority_accuracy": majority_accuracy,
        "majority_balanced_accuracy": majority_balanced_accuracy,
        "random_accuracy": random_accuracy,
        "random_balanced_accuracy": random_balanced_accuracy,
    }


def _get_metric(metrics_for_model: Dict[str, Any], split: str, metric_name: str):
    return metrics_for_model[split][metric_name]


def summarize_model_metrics(
    metrics: Dict[str, Dict[str, Any]],
    dataloaders: Dict[str, Any],
    n_classes: int,
    output_dir: str | Path = PROJECT_DIR / "results" / "plots",
    save_name: str = "model_summary_plot.png",
    splits=("train", "valid"),
    metric_names=("accuracy", "balanced_accuracy"),
):
    ytrue = {
        split: _collect_labels_from_dataloader(dataloaders[split]) for split in splits
    }
    majority_class = _get_majority_class_from_train(ytrue["train"])

    baselines = {
        split: _compute_baseline_scores(
            y_true=ytrue[split],
            majority_class=majority_class,
            n_classes=n_classes,
        )
        for split in splits
    }

    rows = []

    for model_name, model_metrics in metrics.items():
        row = {"model": model_name}

        for split in splits:
            for metric_name in metric_names:
                row[f"{split}_{metric_name}"] = model_metrics[split][metric_name]

        rows.append(row)

    baseline_models = {
        "majority_classifier": "majority",
        "random_guess_classifier": "random",
    }

    for baseline_model_name, baseline_key in baseline_models.items():
        row = {"model": baseline_model_name}

        for split in splits:
            for metric_name in metric_names:
                row[f"{split}_{metric_name}"] = baselines[split][
                    f"{baseline_key}_{metric_name}"
                ]

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    print("\nModel performance summary")
    print("=" * 80)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nMajority class selected from training set: {majority_class}")

    plot_path = _save_summary_plot(
        summary_df=summary_df,
        output_dir=output_dir,
        save_name=save_name,
    )

    print(f"\nSaved plot to: {plot_path}")

    return summary_df, plot_path


def _save_summary_plot(
    summary_df: pd.DataFrame,
    output_dir: str | Path,
    save_name: str,
) -> Path:
    plot_df = summary_df.copy()
    model_names = plot_df["model"].tolist()

    panels = [
        ("train_accuracy", "Training accuracy"),
        ("valid_accuracy", "Validation accuracy"),
        ("train_balanced_accuracy", "Training balanced accuracy"),
        ("valid_balanced_accuracy", "Validation balanced accuracy"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    axes = axes.ravel()

    for ax, (column, title) in zip(axes, panels):
        values = plot_df[column].values

        ax.bar(model_names, values)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=45)

        for i, value in enumerate(values):
            if value is not None and not pd.isna(value):
                ax.text(
                    i,
                    value,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    fig.tight_layout()

    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_path = save_dir / save_name
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return plot_path
