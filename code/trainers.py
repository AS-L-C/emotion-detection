from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import models as mods
import optuna
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent

# ============================================================
# Config utilities
# ============================================================


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def ensure_output_dir(home_folder: str | Path, model_name: str) -> Path:
    output_dir = Path(home_folder) / "models" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# Optuna parameter sampling
# ============================================================


def suggest_from_config(
    trial: optuna.Trial,
    name: str,
    spec: Dict[str, Any],
) -> Any:
    param_type = spec["type"]

    if param_type == "loguniform":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=True,
        )

    if param_type == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
        )

    if param_type == "int":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
        )

    if param_type == "categorical":
        return trial.suggest_categorical(
            name,
            spec["choices"],
        )

    raise ValueError(f"Unsupported hyperparameter type: {param_type}")


def resolve_component_params(
    trial: optuna.Trial,
    component_name: str,
    component_config: Dict[str, Any],
    HP_OPT: bool = True,
) -> Dict[str, Any]:
    params = {}

    fixed_params = component_config.get("fixed", {}) or {}
    search_params = component_config.get("search", {}) or {}

    params.update(fixed_params)

    for param_name, search_spec in search_params.items():
        optuna_name = f"{component_name}.{param_name}"

        if HP_OPT:
            params[param_name] = suggest_from_config(
                trial=trial,
                name=optuna_name,
                spec=search_spec,
            )
        else:
            params[param_name] = search_spec["default"]

    return params


def sample_hyperparameters(
    trial: optuna.Trial,
    config: Dict[str, Any],
    HP_OPT: bool = True,
) -> Dict[str, Any]:
    optimizer_params = resolve_component_params(
        trial=trial,
        component_name="optimizer",
        component_config=config["optimizer"],
        HP_OPT=HP_OPT,
    )

    scheduler_params = resolve_component_params(
        trial=trial,
        component_name="scheduler",
        component_config=config.get("scheduler", {"name": "none"}),
        HP_OPT=HP_OPT,
    )

    return {
        "optimizer": optimizer_params,
        "scheduler": scheduler_params,
    }


# ============================================================
# Optuna pruner
# ============================================================


def build_pruner(config: Dict[str, Any]) -> optuna.pruners.BasePruner:
    pruner_config = config.get("pruner", {})
    pruner_name = pruner_config.get("name", "none")

    if pruner_name is None or pruner_name == "none":
        return optuna.pruners.NopPruner()

    if pruner_name == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=int(pruner_config.get("n_startup_trials", 3)),
            n_warmup_steps=int(pruner_config.get("n_warmup_steps", 2)),
            interval_steps=int(pruner_config.get("interval_steps", 1)),
        )

    if pruner_name == "successive_halving":
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=int(pruner_config.get("min_resource", 2)),
            reduction_factor=int(pruner_config.get("reduction_factor", 3)),
            min_early_stopping_rate=int(
                pruner_config.get("min_early_stopping_rate", 0)
            ),
        )

    raise ValueError(f"Unsupported pruner: {pruner_name}")


# ============================================================
# Model, optimizer, scheduler, criterion
# ============================================================


def build_model(
    model_name: str,
    channels: int,
    data_spec: Any,
    n_classes: int,
    device: torch.device | str,
) -> nn.Module:
    model = mods.get_model(
        model=model_name,
        channels=channels,
        data_spec=data_spec,
        n_classes=n_classes,
        device=device,
    )

    return model.to(device)


def get_trainable_parameters(model: nn.Module):
    return filter(lambda p: p.requires_grad, model.parameters())


def build_optimizer(
    model: nn.Module,
    optimizer_config: Dict[str, Any],
    optimizer_params: Dict[str, Any],
) -> torch.optim.Optimizer:
    optimizer_name = optimizer_config.get("name", "adamw").lower()

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            get_trainable_parameters(model),
            **optimizer_params,
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            get_trainable_parameters(model),
            **optimizer_params,
        )

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            get_trainable_parameters(model),
            **optimizer_params,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: Dict[str, Any],
    scheduler_params: Dict[str, Any],
):
    scheduler_name = scheduler_config.get("name", "none")
    if scheduler_name is None or str(scheduler_name).lower() == "none":
        return None

    if scheduler_name == "reduce_lr_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **scheduler_params,
        )

    if scheduler_name == "cosine_annealing_lr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            **scheduler_params,
        )

    if scheduler_name == "step_lr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            **scheduler_params,
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def build_criterion(config: Dict[str, Any]) -> nn.Module:
    criterion_name = config.get("criterion", "cross_entropy")

    if criterion_name == "cross_entropy":
        return nn.CrossEntropyLoss()

    raise ValueError(f"Unsupported criterion: {criterion_name}")


# ============================================================
# Batch utilities
# ============================================================


def is_improvement(
    current_score: float,
    best_score: Optional[float],
    direction: str,
) -> bool:
    if best_score is None:
        return True

    if direction == "maximize":
        return current_score > best_score

    if direction == "minimize":
        return current_score < best_score

    raise ValueError(f"Unsupported direction: {direction}")


def step_scheduler(
    scheduler,
    scheduler_config: Dict[str, Any],
    current_score: float,
) -> None:
    """
    ReduceLROnPlateau needs a metric.
    Most other schedulers do not.
    """
    if scheduler is None:
        return

    scheduler_name = scheduler_config.get("name", "none")

    if scheduler_name is None:
        return

    if scheduler_name == "reduce_lr_on_plateau":
        scheduler.step(current_score)
    else:
        scheduler.step()


# ============================================================
# Training and evaluation
# ============================================================


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device | str,
    gradient_clip_norm: Optional[float] = None,
) -> float:
    model.train()

    total_loss = 0.0
    n_samples = 0

    for batch in dataloader:
        x, y = batch
        optimizer.zero_grad(set_to_none=True)

        logits = model(x.to(device))
        loss = criterion(logits, y.to(device))

        loss.backward()

        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(gradient_clip_norm),
            )

        optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        n_samples += batch_size

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
    criterion: nn.Module = None,
    dummy_y: bool = False,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    n_samples = 0

    all_preds = []
    all_targets = []

    for batch in dataloader:
        x, y = batch
        batch_size = x.shape[0]

        logits = model(x.to(device))

        if criterion and not dummy_y:
            loss = criterion(logits, y.to(device))
            total_loss += loss.item() * batch_size

        preds = torch.argmax(logits, dim=-1)
        n_samples += batch_size

        all_preds.append(preds.detach().cpu())
        all_targets.append(y.detach().cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    return {
        "loss": total_loss / n_samples if criterion and not dummy_y else None,
        "accuracy": accuracy_score(y_true, y_pred) if not dummy_y else None,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)
        if not dummy_y
        else None,
        "ypred": y_pred,
    }


def eval_model_on_splits(  # collect_train_valid_metrics(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    criterion: nn.Module,
    device: torch.device | str,
    splits=("train", "valid"),
) -> Dict[str, Dict[str, float]]:
    splits = [splits] if isinstance(splits, str) else splits
    metrics = {
        split: evaluate_model(
            model=model,
            dataloader=dataloaders[split],
            criterion=criterion,
            device=device,
        )
        for split in splits
    }
    return metrics


def train_model_for_epochs(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    config: Dict[str, Any],
    hyperparameters: Dict[str, Any],
    device: torch.device | str,
    n_epochs: int,
    trial: Optional[optuna.Trial] = None,
    use_tqdm: bool = False,
) -> Tuple[nn.Module, Dict[str, Any], torch.optim.Optimizer, Any]:
    criterion = build_criterion(config)

    optimizer = build_optimizer(
        model=model,
        optimizer_config=config["optimizer"],
        optimizer_params=hyperparameters["optimizer"],
    )

    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_config=config.get("scheduler", {"name": None}),
        scheduler_params=hyperparameters.get("scheduler", {}),
    )

    training_config = config.get("training", {})
    gradient_clip_norm = training_config.get("gradient_clip_norm")

    monitor_metric = config.get("monitor_metric", "valid_balanced_accuracy")
    direction = config.get("direction", "maximize")

    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_accuracy": [],
        "valid_accuracy": [],
        "train_balanced_accuracy": [],
        "valid_balanced_accuracy": [],
    }

    best_score = None
    best_state_dict = None
    best_epoch = None

    epoch_iter = range(n_epochs)

    if use_tqdm:
        epoch_iter = tqdm(
            epoch_iter,
            total=n_epochs,
            desc="Training",
            leave=True,
        )

    for epoch in epoch_iter:
        train_loss = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            gradient_clip_norm=gradient_clip_norm,
        )

        current_metrics = eval_model_on_splits(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            device=device,
            splits=["train", "valid"],
        )

        metrics_flat = {
            "train_loss": float(train_loss),
            "valid_loss": float(current_metrics["valid"]["loss"]),
            "train_accuracy": float(current_metrics["train"]["accuracy"]),
            "valid_accuracy": float(current_metrics["valid"]["accuracy"]),
            "train_balanced_accuracy": float(
                current_metrics["train"]["balanced_accuracy"]
            ),
            "valid_balanced_accuracy": float(
                current_metrics["valid"]["balanced_accuracy"]
            ),
        }

        for key, value in metrics_flat.items():
            history[key].append(value)

        current_score = metrics_flat[monitor_metric]

        step_scheduler(
            scheduler=scheduler,
            scheduler_config=config.get("scheduler", {"name": "none"}),
            current_score=current_score,
        )

        if is_improvement(
            current_score=current_score,
            best_score=best_score,
            direction=direction,
        ):
            best_score = current_score
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        if use_tqdm:
            epoch_iter.set_postfix(
                {
                    "epoch": f"{epoch + 1}/{n_epochs}",
                    monitor_metric: f"{current_score:.4f}",
                    "best": f"{best_score:.4f}" if best_score is not None else "None",
                    "best_epoch": best_epoch + 1 if best_epoch is not None else None,
                }
            )

        if trial is not None:
            trial.report(current_score, step=epoch)

            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Trial pruned at epoch {epoch}. "
                    f"{monitor_metric}={current_score:.6f}"
                )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    final_metrics = eval_model_on_splits(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        device=device,
        splits=["train", "valid"],
    )

    result = {
        "history": history,
        "final_metrics": final_metrics,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "monitor_metric": monitor_metric,
    }

    return model, result, optimizer, scheduler


# ============================================================
# Checkpointing
# ============================================================


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    hyperparameters: Dict[str, Any],
    metrics: Dict[str, Any],
    model_name: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_name": model_name,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "hyperparameters": hyperparameters,
        "metrics": metrics,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, path)


def load_model_state(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device | str,
) -> nn.Module:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return model


# ============================================================
# Main Optuna optimization function
# ============================================================
def optimize_model(
    *,
    model: str,
    dataloaders: Dict[str, DataLoader],
    channels: int,
    data_spec: Any,
    n_classes: int,
    HP_OPT: bool = True,
    device: torch.device | str,
    home_folder: str | Path = PROJECT_DIR / "results",
    config_path: str | Path = PROJECT_DIR / "configs/train_optuna.yml",
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    assert model in mods.get_avail_models(), f"Unrecognized model '{model}'"

    config = load_config(config_path)
    output_dir = ensure_output_dir(home_folder, model)

    hpo_epochs = int(config["hpo_epochs"])
    n_trials = int(config["n_trials"])
    final_epochs = int(config.get("final_epochs", hpo_epochs))

    direction = config.get("direction", "maximize")
    optuna_direction = "maximize" if direction == "maximize" else "minimize"

    best_checkpoint_path = output_dir / "best_model.pt"

    study = None
    best_trial = None
    completed_trials = []

    if HP_OPT:
        pruner = build_pruner(config)

        study = optuna.create_study(
            direction=optuna_direction,
            pruner=pruner,
        )

        def objective(trial: optuna.Trial) -> float:
            hyperparameters = sample_hyperparameters(
                trial=trial,
                config=config,
            )

            trial_model = build_model(
                model_name=model,
                channels=channels,
                data_spec=data_spec,
                n_classes=n_classes,
                device=device,
            )

            try:
                trial_model, result, optimizer, scheduler = train_model_for_epochs(
                    model=trial_model,
                    dataloaders=dataloaders,
                    config=config,
                    hyperparameters=hyperparameters,
                    device=device,
                    n_epochs=hpo_epochs,
                    trial=trial,
                )

            except optuna.TrialPruned:
                raise

            final_metrics = result["final_metrics"]
            objective_value = float(result["best_score"])

            trial_metrics = {
                "train": final_metrics["train"],
                "valid": final_metrics["valid"],
                "objective_value": objective_value,
                "history": result["history"],
                "best_epoch": result["best_epoch"],
                "best_score": result["best_score"],
            }

            save_every_trial = config.get("training", {}).get(
                "save_every_trial",
                False,
            )

            trial_checkpoint_path = output_dir / f"trial_{trial.number:04d}.pt"

            if save_every_trial:
                save_checkpoint(
                    path=trial_checkpoint_path,
                    model=trial_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=hpo_epochs,
                    hyperparameters=hyperparameters,
                    metrics=trial_metrics,
                    model_name=model,
                    extra={
                        "trial_number": trial.number,
                        "channels": channels,
                        "data_spec": data_spec,
                        "n_classes": n_classes,
                    },
                )

            trial.set_user_attr("hyperparameters", hyperparameters)
            trial.set_user_attr("metrics", trial_metrics)

            if save_every_trial:
                trial.set_user_attr("checkpoint_path", str(trial_checkpoint_path))

            return objective_value

        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=1,
        )

        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if len(completed_trials) == 0:
            raise RuntimeError(
                "No Optuna trials completed successfully. "
                "All trials may have been pruned or failed. "
                "Try increasing pruner.n_warmup_steps or pruner.n_startup_trials."
            )

        best_trial = study.best_trial
        best_hyperparameters = best_trial.user_attrs["hyperparameters"]

    else:
        print("Skipping hyperparameter optimization since HP_OPT=False")
        best_hyperparameters = sample_hyperparameters(
            trial=None,
            config=config,
            HP_OPT=HP_OPT,
        )
    best_model = build_model(
        model_name=model,
        channels=channels,
        data_spec=data_spec,
        n_classes=n_classes,
        device=device,
    )

    if HP_OPT:
        print("Training model with best hyperparameters set...")
    else:
        print("Training model with default hyperparameters...")

    best_model, result, optimizer, scheduler = train_model_for_epochs(
        model=best_model,
        dataloaders=dataloaders,
        config=config,
        hyperparameters=best_hyperparameters,
        device=device,
        n_epochs=final_epochs,
        trial=None,
        use_tqdm=True,
    )

    criterion = build_criterion(config)

    metrics = eval_model_on_splits(
        model=best_model,
        dataloaders=dataloaders,
        criterion=criterion,
        device=device,
        splits=["train", "valid"],
    )

    metrics["best_epoch"] = result["best_epoch"]
    metrics["best_score"] = result["best_score"]

    save_checkpoint(
        path=best_checkpoint_path,
        model=best_model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=final_epochs,
        hyperparameters=best_hyperparameters,
        metrics=metrics,
        model_name=model,
        extra={
            "best_trial_number": best_trial.number if best_trial is not None else None,
            "best_trial_value": best_trial.value if best_trial is not None else None,
            "channels": channels,
            "data_spec": data_spec,
            "n_classes": n_classes,
            "study_direction": optuna_direction if HP_OPT else None,
            "pruner": config.get("pruner", {"name": "none"}) if HP_OPT else None,
            "config": config,
            "hp_opt": HP_OPT,
        },
    )

    if HP_OPT:
        summary = {
            "best_trial_number": best_trial.number,
            "best_trial_value": best_trial.value,
            "best_hyperparameters": best_hyperparameters,
            "metrics": metrics,
            "best_checkpoint_path": str(best_checkpoint_path),
            "n_trials_requested": n_trials,
            "n_trials_completed": len(completed_trials),
            "n_trials_pruned": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            ),
            "n_trials_failed": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
            ),
        }

    else:
        summary = {
            "best_trial_number": None,
            "best_trial_value": None,
            "best_hyperparameters": best_hyperparameters,
            "metrics": metrics,
            "best_checkpoint_path": str(best_checkpoint_path),
            "n_trials_requested": None,
            "n_trials_completed": None,
            "n_trials_pruned": None,
            "n_trials_failed": None,
        }

    save_json(output_dir / "optuna_summary.json", summary)

    return best_model, metrics, best_hyperparameters


def retrieve_best_model(
    model,
    channels,
    data_spec,
    n_classes,
    device,
    home_fld=PROJECT_DIR / "results/models",
):
    model_path = home_fld / model
    model_checkpoint = model_path / "best_model.pt"
    optuna_summary = read_json(model_path / "optuna_summary.json")
    metrics = optuna_summary["metrics"]
    best_hyperparameters = optuna_summary["best_hyperparameters"]
    best_model = build_model(
        model_name=model,
        channels=channels,
        data_spec=data_spec,
        n_classes=n_classes,
        device=device,
    )
    state_dict = torch.load(model_checkpoint, map_location=device, weights_only=False,)
    best_model.load_state_dict(state_dict["model_state_dict"])
    best_model.to(device)
    best_model.eval()
    return best_model, metrics, best_hyperparameters

def save_small_checkpoint(checkpoint_path: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, )

    small_checkpoint = {
        "model_state_dict": checkpoint["model_state_dict"]
    }

    small_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}_small{checkpoint_path.suffix}"
    )

    torch.save(small_checkpoint, small_path)

    return small_path
