#!/usr/bin/env python3
"""
Train the convolutional autoencoder on frame pairs under ``Train/<episode>/frameN.jpg``.

Uses the same 64×64 center-crop + resize as the reference notebook (PIL only; no torchvision).

Example:
  python scripts/train_autoencoder.py --data Train --epochs 5 --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from lltp.dataset import FrameSequenceDataset
from lltp.models import ConvAE
from lltp.training import ae_loss, ae_train_step


def _pil_to_chw_64(pil: Image.Image) -> torch.Tensor:
    pil = pil.convert("RGB")
    w, h = pil.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    pil = pil.crop((left, top, left + side, top + side))
    pil = pil.resize((64, 64), Image.Resampling.LANCZOS)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ConvAE reconstruction on lunar lander frames.")
    parser.add_argument("--data", type=Path, default=Path("Train"), help="Root with episode subfolders.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu", help="e.g. cpu or cuda")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/convae.pt"))
    args = parser.parse_args()

    if not (args.data / "0").is_dir():
        print(f"No episode folders under {args.data} (expected {args.data}/0/...).", file=sys.stderr)
        return 1

    ds = FrameSequenceDataset(args.data, transform=_pil_to_chw_64)
    if len(ds) == 0:
        print(f"No frames found under {args.data}.", file=sys.stderr)
        return 1

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    device = torch.device(args.device)
    net = ConvAE().to(device)
    opt = optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        losses = []
        for x, _t in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            x = x.float().to(device)
            loss = ae_train_step(x, net, opt, ae_loss)
            losses.append(float(loss.cpu()))
        print(f"epoch {epoch + 1}: mean train MSE = {np.mean(losses):.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.output)
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
