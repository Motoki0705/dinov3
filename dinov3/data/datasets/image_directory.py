# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from pathlib import Path
from typing import Any, Callable, Optional

from .extended import ExtendedVisionDataset


_IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})


class ImageDirectory(ExtendedVisionDataset):
    """Load unlabeled images recursively from a directory."""

    Labels = int

    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        root_path = Path(root).expanduser()
        if not root_path.is_dir():
            raise FileNotFoundError(f'Image directory does not exist: "{root_path}"')

        self._image_paths = tuple(
            sorted(
                path
                for path in root_path.rglob("*")
                if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
            )
        )
        if not self._image_paths:
            raise RuntimeError(f'No supported images found under "{root_path}"')

    def get_image_data(self, index: int) -> bytes:
        return self._image_paths[index].read_bytes()

    def get_target(self, index: int) -> Any:
        return 0

    def __len__(self) -> int:
        return len(self._image_paths)
