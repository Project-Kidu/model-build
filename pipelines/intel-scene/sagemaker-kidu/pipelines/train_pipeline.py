import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import json
import math
import os
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig
from torchvision.datasets import ImageFolder

from src import utils
from src.train import train as train_main

log = utils.get_pylogger(__name__)

sm_output_dir = os.environ.get("SM_OUTPUT_DIR")
sm_model_dir = os.environ.get("SM_MODEL_DIR")
num_cpus = int(os.environ.get("SM_NUM_CPUS"))

train_channel = os.environ.get("SM_CHANNEL_TRAIN")
test_channel = os.environ.get("SM_CHANNEL_TEST")
batch_size = int(os.environ.get("BATCH_SIZE"))
optimizer = os.environ.get("OPTIMIZER")
learning_rate = float(os.environ.get("LR"))
model_name = os.environ.get("MODEL")
use_augmentation_pipeline = int(os.environ.get("AUGMENTATION"))

ml_root = Path("/opt/ml")


def get_training_env():
    sm_training_env = os.environ.get("SM_TRAINING_ENV")
    sm_training_env = json.loads(sm_training_env)

    return sm_training_env


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    
    train_dataset = ImageFolder(train_channel)
    print(":: Class names: ", train_dataset.classes)

    sm_training_env = get_training_env()

    # print(cfg["model"].keys())
    cfg["tags"] = ["Intel Scene Classification Training"]
    cfg["seed"] = 12345
    cfg["script"] = True

    cfg["trainer"]["accelerator"] = "auto"
    cfg["trainer"]["min_epochs"] = 5
    cfg["trainer"]["max_epochs"] = 24

    cfg["data"]["batch_size"] = batch_size
    cfg["data"]["num_workers"] = num_cpus
    cfg["data"]["train_data_dir"] = train_channel
    cfg["data"]["test_data_dir"] = test_channel
    cfg["data"]["use_augmentation_pipeline"] = use_augmentation_pipeline

    cfg["model"]["learning_rate"] = learning_rate / 10.0
    cfg["model"]["net"]["model_name"] = model_name
    cfg["model"]["optimizer"]["_target_"] = optimizer
    cfg["model"]["scheduler"]["steps_per_epoch"] =  math.ceil(cfg["data"]["train_val_test_split"][0] / batch_size)
    cfg["model"]["scheduler"]["max_lr"] = learning_rate
    
    cfg["callbacks"]["model_checkpoint"]["dirpath"] = sm_model_dir
    cfg["logger"]["tensorboard"]["save_dir"] = ml_root / "output" / "tensorboard" / sm_training_env["job_name"]

    # train the model
    print(":: Training ...")
    metric_dict, _ = train_main(cfg)



    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
