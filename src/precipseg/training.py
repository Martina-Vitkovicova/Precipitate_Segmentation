import gc
import itertools
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil
import torch
from segmentation_models_pytorch import Unet
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.precipseg import predictions
from src.precipseg.data import save_image_results
from src.precipseg.losses import make_loss
from src.precipseg.metrics import compute_metric_curves, save_metric_curves

REQUIRED_KEYS = ("dice", "iou", "precision", "recall", "f1")

Batch = dict[str, torch.Tensor]
Loss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class ImgData:
    train_dl: DataLoader[Batch]
    val_dl: DataLoader[Batch]
    test_imgs: list[np.ndarray]
    test_labels: list[np.ndarray]
    real_imgs: list[Path]
    synth_imgs: list[Path]
    synth_root: str | Path


class ResourceGuard(RuntimeError):
    pass


def _check_resources(*, disk_path=".", min_disk_gb=10.0, max_ram_frac=0.90, max_gpu_mem_frac=0.95) -> None:
    du = shutil.disk_usage(disk_path)
    free_gb = du.free / (1024**3)
    if free_gb < min_disk_gb:
        raise ResourceGuard(f"Low disk: {free_gb:.2f} GB free (< {min_disk_gb} GB) at {disk_path}")

    vm = psutil.virtual_memory()
    if (vm.percent / 100.0) > max_ram_frac:
        raise ResourceGuard(f"High RAM usage: {vm.percent:.1f}% (> {max_ram_frac*100:.1f}%)")

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used_frac = 1.0 - (free / total)
        if used_frac > max_gpu_mem_frac:
            raise ResourceGuard(f"High GPU mem usage: {used_frac*100:.1f}% (> {max_gpu_mem_frac*100:.1f}%)")


class EarlyStopper:
    def __init__(self, patience: int = 1, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_loss = np.inf

    def update(self, val_loss: float) -> bool:
        if val_loss < (self.best_val_loss - self.min_delta):
            self.best_val_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class MetricCheckpointer:
    def __init__(self, model: torch.nn.Module, model_path: Path, min_delta: float = 0.0):
        self.model = model
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(exist_ok=True, parents=True)
        self.best_val_loss = np.inf
        self.min_delta = min_delta

    def update(self, val_loss: float, *, epoch: int, iou: float | None) -> bool:
        if val_loss > (self.best_val_loss - self.min_delta):
            return False
        before = self.best_val_loss
        self.best_val_loss = val_loss

        msg = f"Epoch {epoch}: before_val_loss={before:.5f} -> {val_loss:.5f}."
        if iou is not None:
            msg += f" Val IoU={iou:.3f}."
        tqdm.write(msg)

        torch.save(self.model, self.model_path)
        return True


class EpochLogger:
    def __init__(self, log_path: str | Path):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", encoding="utf-8")
        self._epoch_lines: list[str] = []

    def write(self, line: str) -> None:
        self._f.write(line.rstrip("\n") + "\n")

    def write_epoch(self, *, epoch: int, train_loss: float, val_loss: float, val_iou: float, lr: float,
                    checkpoint: bool) -> None:
        self._epoch_lines.append(
            f"EPOCH epoch={epoch} "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} "
            f"val_iou={val_iou:.3f} "
            f"lr={lr:.1e} "
            f"checkpoint={1 if checkpoint else 0}."
        )

    def flush_epochs(self) -> None:
        for line in self._epoch_lines:
            self._f.write(line + "\n")

    def close(self) -> None:
        self._f.close()


def _get_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _step(model: torch.nn.Module, batch: Batch, loss_fn: Loss, device: str | torch.device = "cpu") -> torch.Tensor:
    batch = {k: v.to(device) for k, v in batch.items()}
    pred = model(batch["x"])
    return loss_fn(pred, batch["y"])


def _train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader[Batch],
    optimizer: torch.optim.Optimizer,
    step_fn: Callable[[Batch], torch.Tensor],
) -> float:
    model.train()

    def train_step(batch: Batch) -> float:
        optimizer.zero_grad(set_to_none=True)
        loss = step_fn(batch)
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu().item())

    losses = [train_step(batch) for batch in dataloader]
    return float(np.mean(losses)) if losses else float("nan")


def _validate_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader[Batch],
    step_fn: Callable[[Batch], torch.Tensor],
    *,
    threshold: float = 0.5,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    ious: list[float] = []

    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in dataloader:
            loss = float(step_fn(batch).detach().cpu().item())
            losses.append(loss)

            x = batch["x"].to(device)
            y_true = batch["y"].to(device)

            y_pred = torch.sigmoid(model(x)) > threshold
            y_true_bin = y_true > 0.5

            intersection = (y_pred & y_true_bin).float().sum((1, 2, 3))
            union = (y_pred | y_true_bin).float().sum((1, 2, 3))
            iou = (intersection / (union + 1e-6)).mean().item()
            ious.append(float(iou))

    return float(np.mean(losses)) if losses else float("nan"), float(np.mean(ious)) if ious else float("nan")


def _setup_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_patience: int,
) -> Callable[[float], None]:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=scheduler_patience, factor=0.3
    )

    def scheduler_fn(val_loss: float) -> None:
        scheduler.step(val_loss)

    return scheduler_fn


@dataclass
class TrainResult:
    train_losses: list[float]
    val_losses: list[float]
    val_ious: list[float]
    last_epoch: int
    last_train_loss: float
    last_val_loss: float
    last_val_iou: float
    start_time: datetime
    end_time: datetime


def _run_training(
    model: torch.nn.Module,
    train_dataloader: DataLoader[Batch],
    val_dataloader: DataLoader[Batch],
    loss_fn: Loss,
    *,
    epochs: int | None,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    epoch_logger: EpochLogger,
    disk_path: str | Path = ".",
    iou_threshold: float = 0.5,
    scheduler: Callable[[float], None] | None = None,
    early_stopper: EarlyStopper | None = None,
    checkpointer: MetricCheckpointer | None = None,
) -> TrainResult:
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_ious: list[float] = []

    last_epoch = -1
    last_train = float("nan")
    last_val = float("nan")
    last_iou = float("nan")

    def step_fn(batch: Batch) -> torch.Tensor:
        return _step(model, batch, loss_fn, device=device)

    start_time = datetime.now()
    epochs_iter = range(epochs) if epochs is not None else itertools.count()

    with logging_redirect_tqdm(), tqdm(epochs_iter, desc="Training epochs") as t:
        for epoch in t:
            last_epoch = int(epoch)

            _check_resources(disk_path=disk_path)

            last_train = _train_epoch(model, train_dataloader, optimizer, step_fn)
            last_val, last_iou = _validate_epoch(model, val_dataloader, step_fn, threshold=iou_threshold)

            train_losses.append(float(last_train))
            val_losses.append(float(last_val))
            val_ious.append(float(last_iou))

            if scheduler is not None:
                scheduler(last_val)

            saved = False
            if checkpointer is not None:
                saved = checkpointer.update(last_val, epoch=last_epoch, iou=last_iou)

            epoch_logger.write_epoch(
                epoch=last_epoch,
                train_loss=last_train,
                val_loss=last_val,
                val_iou=last_iou,
                lr=_get_lr(optimizer),
                checkpoint=saved,
            )

            t.set_postfix(
                epoch=last_epoch,
                train_loss=f"{last_train:.5f}",
                val_loss=f"{last_val:.5f}",
                val_IoU=f"{last_iou:.3f}",
                lr=f"{_get_lr(optimizer):.2e}",
            )

            if early_stopper is not None and early_stopper.update(last_val):
                break

    end_time = datetime.now()
    return TrainResult(
        train_losses=train_losses,
        val_losses=val_losses,
        val_ious=val_ious,
        last_epoch=last_epoch,
        last_train_loss=last_train,
        last_val_loss=last_val,
        last_val_iou=last_iou,
        start_time=start_time,
        end_time=end_time,
    )


def _write_training_summary(
    epoch_logger: EpochLogger,
    *,
    result: TrainResult,
    loss_fn: Callable[..., Any],
    synth_path: str | Path,
    real_imgs_num: int,
    synth_imgs_num: int,
) -> None:
    duration = result.end_time - result.start_time

    loss_desc_fn = getattr(loss_fn, "description", None)
    loss_desc = loss_desc_fn() if callable(loss_desc_fn) else type(loss_fn).__name__

    epoch_logger.write(f"Training summary ({result.end_time.strftime('%Y-%m-%d %H:%M:%S')})")
    epoch_logger.write("-------------------------------------------------------")
    epoch_logger.write(f"Total epochs run: {len(result.train_losses)}")
    epoch_logger.write(f"Training duration: {duration}")
    epoch_logger.write("")
    epoch_logger.write(f"Training data: real {real_imgs_num} + synth {synth_imgs_num}")

    if synth_imgs_num > 0 and synth_path:
        epoch_logger.write(f"Synthetic data path: {synth_path}")

    epoch_logger.write("")
    epoch_logger.write(f"Loss: {loss_desc}")
    epoch_logger.write("")
    epoch_logger.write("Epoch log:")
    epoch_logger.write("")

    epoch_logger.flush_epochs()


def train(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss_fn: Loss,
    epochs: int | None = None,
    patience: int | None = None,
    scheduler_patience: int | None = None,
    checkpoint_path: str | Path | None = None,
    lr: float = 0.001,
    device: str | torch.device = "cpu",
    log_path: str | Path = "training_log.txt",
    synth_path: str | Path = "",
    real_imgs_num: int = 45,
    synth_imgs_num: int = 450,
) -> tuple[list[float], list[float]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = _setup_scheduler(optimizer, scheduler_patience) if scheduler_patience is not None else None
    early_stopper = EarlyStopper(patience=patience, min_delta=0.0) if patience is not None else None

    checkpointer = None
    disk_path: str | Path = "."
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpointer = MetricCheckpointer(model, checkpoint_path)
        disk_path = checkpoint_path.parent

    epoch_logger = EpochLogger(log_path)
    result: TrainResult | None = None

    try:
        result = _run_training(
            model,
            train_dataloader,
            val_dataloader,
            loss_fn,
            epochs=epochs,
            optimizer=optimizer,
            device=device,
            epoch_logger=epoch_logger,
            disk_path=disk_path,
            scheduler=scheduler,
            early_stopper=early_stopper,
            checkpointer=checkpointer,
        )
        _write_training_summary(
            epoch_logger,
            result=result,
            loss_fn=loss_fn,
            synth_path=synth_path,
            real_imgs_num=real_imgs_num,
            synth_imgs_num=synth_imgs_num,
        )
    finally:
        epoch_logger.close()

    if result is not None:
        tqdm.write(
            f"Training finished at epoch {result.last_epoch}: "
            f"train_loss={result.last_train_loss:.5f}, "
            f"val_loss={result.last_val_loss:.5f}, val_IoU={result.last_val_iou:.3f}"
        )
        return result.train_losses, result.val_losses

    raise RuntimeError("Training did not complete; no result produced.")


def create_model(x_channels: int) -> Unet:
    return Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=x_channels,
        classes=1,
        activation=None,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_run_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def has_required_metrics(d: dict[str, Any]) -> bool:
    return all(k in d for k in REQUIRED_KEYS)


def run_one(
    loss_name: str,
    loss_params: dict[str, float],
    run_seed: int,
    runs_dir: Path,
    x_channels: int,
    device: str | torch.device,
    thresholds: np.ndarray,
    max_images: int | None,
    data: ImgData,
) -> dict[str, Any]:
    set_seed(run_seed)

    run_dir = runs_dir / f"seed_{run_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = create_model(x_channels).to(device)
        loss_fn = make_loss(loss_name, **loss_params)

        checkpoint_path = run_dir / "checkpoint.pt"
        log_path = run_dir / "log.txt"

        train_loss, val_loss = train(
            model,
            data.train_dl,
            data.val_dl,
            loss_fn,
            checkpoint_path=checkpoint_path,
            patience=10,
            scheduler_patience=5,
            device=device,
            log_path=log_path,
            synth_path=data.synth_root,
            real_imgs_num=len(data.real_imgs),
            synth_imgs_num=len(data.synth_imgs),
        )

        with torch.inference_mode():
            test_preds = predictions.segment_many(model, data.test_imgs, device=device, x_channels=x_channels)

        test_lbls = [y.detach().cpu().numpy() if torch.is_tensor(y) else y for y in data.test_labels]
        test_inps = [x.detach().cpu().numpy() if torch.is_tensor(x) else x for x in data.test_imgs]

        dice_vals, iou_vals, mean_p, mean_r, mean_f1 = compute_metric_curves(
            test_lbls, test_preds, thresholds
        )
        save_metric_curves(run_dir / "metrics.txt", thresholds, dice_vals, iou_vals, mean_p, mean_r, mean_f1)
        save_image_results(test_inps, test_lbls, test_preds, run_dir, max_images=max_images)

        payload = {
            "loss": loss_name,
            "loss_params": loss_params,
            "seed": int(run_seed),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "dice": dice_vals,
            "iou": iou_vals,
            "precision": mean_p,
            "recall": mean_r,
            "f1": mean_f1,
        }

        (run_dir / "run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    finally:
        try:
            del test_preds
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        try:
            del loss_fn
        except Exception:
            pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
