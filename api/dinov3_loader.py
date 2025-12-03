import torch


REPO_DIR = "third_party/dinov3"


def get_dinov3_vits16(checkpoint_path: str | None = None):
    """Load DINOv3 ViT-S/16 model.

    Args:
        checkpoint_path: Optional path to a checkpoint file. If provided,
            the checkpoint is loaded from this path. If ``None``, the model
            is initialized with random weights (``pretrained=False``).

    Returns:
        torch.nn.Module: DINOv3 ViT-S/16 model instance.
    """
    if checkpoint_path is not None:
        model = torch.hub.load(
            REPO_DIR,
            "dinov3_vits16",
            source="local",
            weights=checkpoint_path,
        )
    else:
        model = torch.hub.load(
            REPO_DIR,
            "dinov3_vits16",
            source="local",
            pretrained=False,
        )
    return model
