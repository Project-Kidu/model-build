import json
import os
import subprocess
import tarfile
from pathlib import Path
from PIL import Image

import hydra
import pyrootutils
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms as T
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import numpy as np

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils
from src.eval import evaluate

log = utils.get_pylogger(__name__)

ml_root = Path("/opt/ml")
model_artifacts = ml_root / "processing" / "model"
test_data_dir = ml_root / "processing" / "test"
train_data_dir = ml_root / "processing" / "train"
prediction_data_dir = ml_root / "processing" / "prediction"
batch_size = int(os.environ.get("BATCH_SIZE"))
model_name = os.environ.get("MODEL")


def explain_model(eval_folder):
    log.info("Running Model explanation")
    model = torch.jit.load("model.scripted.pt")
    log.info(f"Loaded Model: {model}")

    categories = [
        "buildings",
        "forest",
        "glacier",
        "mountain",
        "sea",
        "street"
    ]

    transforms = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
        ]
    )
    transform_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for file in prediction_data_dir.glob("*.*"):
        log.info(file)
        image = Image.open(file)
        transformed_img = transforms(image)
        image_tensor = transform_normalize(transformed_img)
        image_tensor = image_tensor.unsqueeze(0)

        output = model(image_tensor)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)

        pred_label_idx.squeeze_()
        predicted_label = categories[pred_label_idx.item()]
        log.info(
            f"Predicted: {predicted_label} (confidence = {prediction_score.squeeze().item()}, index = {pred_label_idx.item()})"
        )

        log.info("Integrated Gradients :")
        integrated_gradients = IntegratedGradients(model)
        attributions_ig = integrated_gradients.attribute(image_tensor, target=pred_label_idx, n_steps=200)

        fig, ax = viz.visualize_image_attr(
                np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                method="blended_heat_map",
                cmap="inferno",
                show_colorbar=True,
                sign="positive",
                outlier_perc=1,
                use_pyplot=False
            )
        fig.savefig(f"{eval_folder}/model_explanation_{file.stem}.png")
        plt.close()


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:

    model_path = model_artifacts / "model.tar.gz"

    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    cfg["tags"] = ["Intel Scene Classification Evaluation"]
    cfg["ckpt_path"] = "last.ckpt"

    cfg["trainer"]["accelerator"] = "auto"
    cfg["data"]["batch_size"] = batch_size
    cfg["data"]["num_workers"] = os.cpu_count()
    cfg["data"]["train_data_dir"] = train_data_dir.absolute()
    cfg["data"]["test_data_dir"] = test_data_dir.absolute()
    cfg["model"]["net"]["model_name"] = model_name
 

    print(":: Evaluating Model")
    test_res, _ = evaluate(cfg)

    report_dict = {
        "multiclass_classification_metrics": {
            "accuracy": {
                "value": test_res["test/acc"],
                "standard_deviation": "0",
            },
            "confusion_matrix": {
                "0": {
                    "0": test_res["test/confusion_matrix/00"],
                    "1": test_res["test/confusion_matrix/01"],
                    "2": test_res["test/confusion_matrix/02"],
                    "3": test_res["test/confusion_matrix/03"],
                    "4": test_res["test/confusion_matrix/04"],
                    "5": test_res["test/confusion_matrix/05"],
                },
                "1": {
                    "0": test_res["test/confusion_matrix/10"],
                    "1": test_res["test/confusion_matrix/11"],
                    "2": test_res["test/confusion_matrix/12"],
                    "3": test_res["test/confusion_matrix/13"],
                    "4": test_res["test/confusion_matrix/14"],
                    "5": test_res["test/confusion_matrix/15"],
                },
                "2": {
                    "0": test_res["test/confusion_matrix/20"],
                    "1": test_res["test/confusion_matrix/21"],
                    "2": test_res["test/confusion_matrix/22"],
                    "3": test_res["test/confusion_matrix/23"],
                    "4": test_res["test/confusion_matrix/24"],
                    "5": test_res["test/confusion_matrix/25"],
                },
                "3": {
                    "0": test_res["test/confusion_matrix/30"],
                    "1": test_res["test/confusion_matrix/31"],
                    "2": test_res["test/confusion_matrix/32"],
                    "3": test_res["test/confusion_matrix/33"],
                    "4": test_res["test/confusion_matrix/34"],
                    "5": test_res["test/confusion_matrix/35"],
                },
                "4": {
                    "0": test_res["test/confusion_matrix/40"],
                    "1": test_res["test/confusion_matrix/41"],
                    "2": test_res["test/confusion_matrix/42"],
                    "3": test_res["test/confusion_matrix/43"],
                    "4": test_res["test/confusion_matrix/44"],
                    "5": test_res["test/confusion_matrix/45"],
                },
                "5": {
                    "0": test_res["test/confusion_matrix/50"],
                    "1": test_res["test/confusion_matrix/51"],
                    "2": test_res["test/confusion_matrix/52"],
                    "3": test_res["test/confusion_matrix/53"],
                    "4": test_res["test/confusion_matrix/54"],
                    "5": test_res["test/confusion_matrix/55"],
                },
            },
        }
    }

    eval_folder = ml_root / "processing" / "evaluation"
    eval_folder.mkdir(parents=True, exist_ok=True)
    explain_model(eval_folder)

    subprocess.check_call(
        "cp -r /opt/ml/processing/input/code/logs/* /opt/ml/processing/evaluation/",
        shell=True,
    )

    out_path = eval_folder / "evaluation.json"

    print(f":: Writing to {out_path.absolute()}")

    with out_path.open("w") as f:
        f.write(json.dumps(report_dict))


if __name__ == "__main__":
    main()