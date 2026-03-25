"""Tests use tiny synthetic JPEGs (no Gymnasium)."""

from pathlib import Path

import pytest
import torch
import numpy as np
from PIL import Image

from lltp.dataset import FrameSequenceDataset


def _pil_to_chw_float(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.resize((64, 64)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


@pytest.fixture
def fake_episode_root(tmp_path: Path) -> Path:
    """Two episodes: ep0 has 3 frames, ep1 has 2 frames."""
    for ep, n in [(0, 3), (1, 2)]:
        d = tmp_path / str(ep)
        d.mkdir()
        for i in range(n):
            img = Image.new("RGB", (32, 24), color=(ep * 40, i * 50, 10))
            img.save(d / f"frame{i}.jpg")
    return tmp_path


def test_frame_sequence_dataset_length_and_pairing(fake_episode_root: Path):
    ds = FrameSequenceDataset(fake_episode_root, transform=_pil_to_chw_float)
    assert len(ds) == 5
    x0, t0 = ds[0]
    assert x0.shape == (3, 64, 64)
    assert not torch.allclose(t0, x0)


def _pil_to_chw_float_any_size(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def test_episode_boundary_indices_target_self(fake_episode_root: Path):
    ds = FrameSequenceDataset(fake_episode_root, transform=_pil_to_chw_float_any_size)
    boundaries = [i for i in range(len(ds)) if i + 1 in ds.end_list]
    assert boundaries
    for idx in boundaries:
        x, t = ds[idx]
        assert torch.allclose(x, t), f"boundary idx={idx}"
