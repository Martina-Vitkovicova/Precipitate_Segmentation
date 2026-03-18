from typing import Sequence

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as funct
from tqdm import tqdm

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

Image = Mask = npt.NDArray[np.generic]
Padding = tuple[int, int, int, int]  # (left, right, top, bottom)


def _pad_to_stride(
    t: torch.Tensor,
    stride: int = 32,
) -> tuple[torch.Tensor, Padding]:
    _, _, h, w = t.shape
    pad_h = (stride - h % stride) % stride
    pad_w = (stride - w % stride) % stride
    pads = (0, pad_w, 0, pad_h)
    return funct.pad(t, pads), pads


def _unpad(t: torch.Tensor, pads: Padding) -> torch.Tensor:
    _, pad_w, _, pad_h = pads
    if pad_h:
        t = t[..., :-pad_h, :]
    if pad_w:
        t = t[..., :, :-pad_w]
    return t


def _prepare_input_chw(img: Image, x_channels: int) -> Image:
    if x_channels not in (1, 3):
        raise ValueError(f"{x_channels=} is not supported; use 1 or 3.")

    if img.ndim == 3 and img.shape[2] >= 3:
        # RGB/RGBA -> grayscale
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        # (H,W) or (H,W,1/2) -> take channel 0 if present
        gray = img[:, :, 0] if img.ndim == 3 else img

    gray = gray.astype(np.float32, copy=False)

    m = float(gray.max())
    if m <= 0:
        gray = np.zeros_like(gray, dtype=np.float32)
    elif m > 1.0:
        gray = gray / (255.0 if m <= 255.0 else m)

    gray = np.clip(gray, 0.0, 1.0)

    x = np.stack([gray] * x_channels, axis=0).astype(np.float32)  # (C,H,W)

    if x_channels == 3:
        x = (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]

    return x


def _segment_image(
    model: torch.nn.Module,
    img: Image,
    *,
    device: str | torch.device = "cpu",
    x_channels: int = 3,
    pad_stride: int = 32,
    return_logits: bool = False,
) -> Image | tuple[Image, Image]:
    model.eval()

    x_np = _prepare_input_chw(img, x_channels)
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)  # (1,C,H,W)

    with torch.no_grad():
        x_pad, pads = _pad_to_stride(x, stride=pad_stride)
        logits = model(x_pad)              # (1,1,H',W') expected
        logits = _unpad(logits, pads)      # (1,1,H,W)
        probs = torch.sigmoid(logits)

    fg = probs[0, 0].detach().cpu().numpy().astype(np.float32)  # (H,W)

    if return_logits:
        lg = logits[0, 0].detach().cpu().numpy().astype(np.float32)
        return fg, lg

    return fg


def segment_many(
    model: torch.nn.Module,
    imgs: Sequence[Image],
    *,
    device: str | torch.device = "cpu",
    x_channels: int = 3,
    pad_stride: int = 32,
) -> list[Image]:
    return [
        _segment_image(model, img, device=device, x_channels=x_channels, pad_stride=pad_stride)
        for img in tqdm(imgs, desc="Segmenting", total=len(imgs))
    ]
