from pathlib import Path

import pytest
from PIL import Image

from dinov3.data.datasets import ImageDirectory
from dinov3.data.loaders import make_dataset


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), color).save(path)


def test_image_directory_loads_images_recursively(tmp_path: Path):
    _write_image(tmp_path / "first.jpg", (255, 0, 0))
    _write_image(tmp_path / "nested/second.PNG", (0, 255, 0))
    (tmp_path / "ignored.txt").write_text("not an image")

    dataset = make_dataset(dataset_str=f"ImageDirectory:root={tmp_path}")

    assert len(dataset) == 2
    image, target = dataset[0]
    assert image.mode == "RGB"
    assert target == 0


def test_image_directory_rejects_empty_directory(tmp_path: Path):
    with pytest.raises(RuntimeError, match="No supported images"):
        ImageDirectory(root=str(tmp_path))


def test_image_directory_rejects_imagenet_parameters(tmp_path: Path):
    _write_image(tmp_path / "image.jpg", (255, 0, 0))
    with pytest.raises(ValueError, match="only accepts the root"):
        make_dataset(dataset_str=f"ImageDirectory:root={tmp_path}:split=TRAIN")
