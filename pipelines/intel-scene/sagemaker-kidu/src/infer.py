from typing import List, Tuple

import hydra
import pyrootutils
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from omegaconf import DictConfig
from PIL import Image

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils

log = utils.get_pylogger(__name__)


def load_and_predict(cfg: DictConfig) -> Tuple[dict, dict]:
    """A function to load the scripted model and predict.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating scripted model <{cfg.ckpt_path}>")
    model = torch.jit.load(cfg.ckpt_path)

    log.info(f"Loaded Model: {model}")

    categories = [
        "buildings",
        "forest",
        "glacier",
        "mountain",
        "sea",
        "street",
    ]

    transforms = T.Compose(
        [
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(r"data\seg_pred\seg_pred\22.jpg")
    image = transforms(image).unsqueeze(0)
    logits = model(image)
    preds = F.softmax(logits, dim=1).squeeze(0).tolist()
    top_pred = {categories[i]: preds[i] for i in range(6)}
    print(f"{top_pred=}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="infer.yaml")
def main(cfg: DictConfig) -> None:
    load_and_predict(cfg)


if __name__ == "__main__":
    main()
