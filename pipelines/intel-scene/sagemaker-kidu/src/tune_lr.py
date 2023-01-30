import json
from typing import List, Optional, Tuple

import hydra
import pyrootutils
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def tune_learning_rate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Tune the model using LR finder and save the results.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        None
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    log.info("Starting Auto LR finder!")

    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(
        model=model,
        datamodule=datamodule,
        num_training=(datamodule.hparams.train_val_test_split[0] // datamodule.hparams.batch_size),
    )

    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.savefig(f"{cfg.paths.output_dir}/lr_plot.png")

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    log.info(f"New Learning rate is: {new_lr}")

    json_data = lr_finder.results
    json_data["new_lr"] = new_lr

    with open(f"{cfg.paths.output_dir}/lr_result.json", "w") as outfile:
        json.dump(json_data, outfile, indent=4)

    return None, None


@hydra.main(version_base="1.3", config_path="../configs", config_name="tune_lr.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    # tune the model
    tune_learning_rate(cfg)


if __name__ == "__main__":
    main()
