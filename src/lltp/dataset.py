"""Frame sequence dataset: (current frame, next frame) pairs across episode folders."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class FrameSequenceDataset(Dataset):
    """
    Loads JPEG frames from ``root/{episode}/frame{n}.jpg``.

    Episode boundaries are detected when ``frame0`` of the next folder is missing.
    At the last frame of an episode, the target is the current frame (no next frame).
    """

    def __init__(self, root: str | Path, transform=None, target_transform=None):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.data: list[Image.Image] = []
        self.end_list: list[int] = []
        endlist_count = 0
        folder = 0
        count = 0
        img: Image.Image | None = True  # type: ignore[assignment]

        while img is not None:
            while img is not None:
                path = self.root / str(folder) / f"frame{count}.jpg"
                if path.is_file():
                    img = Image.open(path)
                else:
                    img = None
                count += 1
                if img is not None:
                    self.data.append(img)
                    endlist_count += 1
            count = 0
            folder += 1
            next_path = self.root / str(folder) / f"frame{count}.jpg"
            if next_path.is_file():
                img = Image.open(next_path)
            else:
                img = None
            count += 1
            if img is not None:
                self.data.append(img)
                endlist_count += 1
            self.end_list.append(endlist_count)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        if idx + 1 in self.end_list:
            target = self.data[idx]
        else:
            target = self.data[idx + 1]
        if self.transform:
            sample = self.transform(sample)
            target = self.transform(target)
        return sample, target
