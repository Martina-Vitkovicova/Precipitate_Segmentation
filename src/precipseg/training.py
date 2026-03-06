# Adapted from semseg/training.py in jaroslavknotek/semantic-segmentation.
# Source: https://github.com/jaroslavknotek/semantic-segmentation/blob/main/src/semseg
# Changes: validation IoU, structured per-epoch and summary logging, clearer training flow,
# explicit early stopping, and resource checks.

import itertools
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import shutil
import psutil


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
    return optimizer.param_groups[0]["lr"]


def _step(model, batch, loss_fn, device="cpu"):
    batch = {k: v.to(device) for k, v in batch.items()}
    pred = model(batch["x"])
    return loss_fn(pred, batch["y"])


def _train_epoch(model, dataloader, optimizer, step_fn) -> float:
    model.train()

    def train_step(batch):
        optimizer.zero_grad(set_to_none=True)
        loss = step_fn(batch)
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu().item())

    losses = [train_step(batch) for batch in dataloader]
    return float(np.mean(losses)) if losses else float("nan")


def _validate_epoch(model, dataloader, step_fn, *, threshold: float = 0.5) -> tuple[float, float]:
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


def _setup_scheduler(optimizer, scheduler_patience: int):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=scheduler_patience, factor=0.3
    )

    def scheduler_fn(val_loss: float):
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
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss_fn: Callable,
    *,
    epochs: int | None,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch_logger,
    disk_path: str = ".",
    iou_threshold: float = 0.5,
    scheduler=None,
    early_stopper=None,
    checkpointer=None,
) -> TrainResult:
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_ious: list[float] = []

    last_epoch = -1
    last_train = float("nan")
    last_val = float("nan")
    last_iou = float("nan")

    def step_fn(batch):
        return _step(model, batch, loss_fn, device=device)

    start_time = datetime.now()
    epochs_iter = range(epochs) if epochs is not None else itertools.count()

    with logging_redirect_tqdm(), tqdm(epochs_iter, desc="Training epochs") as t:
        for epoch in t:
            last_epoch = epoch

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
                saved = checkpointer.update(last_val, epoch=epoch, iou=last_iou)

            epoch_logger.write_epoch(
                epoch=epoch,
                train_loss=last_train,
                val_loss=last_val,
                val_iou=last_iou,
                lr=_get_lr(optimizer),
                checkpoint=saved,
            )

            t.set_postfix(
                epoch=epoch,
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
    loss_fn: Callable,
    synth_path: str,
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
        loss_fn: Callable,
        epochs: Optional[int] = None,
        patience: Optional[int] = None,
        scheduler_patience: Optional[int] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        lr: float = 0.001,
        device: Union[str, torch.device] = "cpu",
        log_path: Union[str, Path] = "training_log.txt",
        synth_path: str = "",
        real_imgs_num: int = 45,
        synth_imgs_num: int = 450,
) -> Tuple[list[float], list[float]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = _setup_scheduler(optimizer, scheduler_patience) if scheduler_patience is not None else None
    early_stopper = EarlyStopper(patience=patience, min_delta=0.0) if patience is not None else None

    checkpointer = None
    disk_path = "."
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpointer = MetricCheckpointer(model, checkpoint_path)
        disk_path = str(checkpoint_path.parent)

    epoch_logger = EpochLogger(log_path)

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
            f"train_loss={result.last_train_loss:.5f}, val_loss={result.last_val_loss:.5f}, val_IoU={result.last_val_iou:.3f}"
        )
        return result.train_losses, result.val_losses

    raise RuntimeError("Training did not complete; no result produced.")
