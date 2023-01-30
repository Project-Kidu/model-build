import argparse
import os
import subprocess
from pathlib import Path

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from git.repo.base import Repo
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from utils_pipeline import extract_archive

dvc_repo_url = os.environ.get("DVC_REPO_URL")
dvc_branch = os.environ.get("DVC_BRANCH")

git_user = os.environ.get("GIT_USER", "sagemaker")
git_email = os.environ.get("GIT_EMAIL", "sagemaker-processing@example.com")

ml_root = Path("/opt/ml/processing")

dataset_zip = ml_root / "input" / "intel.zip"
git_path = ml_root / "sagemaker-scene"


def configure_git():
    subprocess.check_call(["git", "config", "--global", "user.email", f'"{git_email}"'])
    subprocess.check_call(["git", "config", "--global", "user.name", f'"{git_user}"'])


def clone_dvc_git_repo():
    print(f"\t:: Cloning repo: {dvc_repo_url}")
    repo = Repo.clone_from(dvc_repo_url, git_path.absolute(), allow_unsafe_protocols=True)

    return repo


def sync_data_with_dvc(repo):
    os.chdir(git_path)
    print(f":: Create branch {dvc_branch}")
    try:
        repo.git.checkout("-b", dvc_branch)
        print(f"\t:: Create a new branch: {dvc_branch}")
    except:
        repo.git.checkout(dvc_branch)
        print(f"\t:: Checkout existing branch: {dvc_branch}")
    print(":: Add files to DVC")

    subprocess.check_call(["dvc", "add", "dataset"])

    repo.git.add(all=True)
    repo.git.commit("-m", f"'add data for {dvc_branch}'")

    print("\t:: Push data to DVC")
    subprocess.check_call(["dvc", "push"])

    print("\t:: Push dvc metadata to git")
    repo.remote(name="origin")
    repo.git.push("--set-upstream", repo.remote().name, dvc_branch, "--force")

    sha = repo.head.commit.hexsha

    print(f":: Commit Hash: {sha}")


class AlbumentationTransforms:
    """Helper class to create test and train transforms using Albumentations."""

    def __init__(self, transforms_list):
        self.transforms = A.Compose(transforms_list)

    def __call__(self, img):
        img = np.array(img)
        return self.transforms(image=img)["image"]


def write_dataset(image_paths, output_dir):
    for (data, _), (img_path, _) in zip(image_paths, image_paths.imgs):
        Path(output_dir / Path(img_path).parent.stem).mkdir(parents=True, exist_ok=True)
        save_image(data / 255, output_dir / Path(img_path).parent.stem / Path(img_path).name)


def generate_train_test_split():
    dataset_extracted = ml_root / "tmp"
    dataset_extracted.mkdir(parents=True, exist_ok=True)

    # split dataset and save to their directories
    print(f":: Extracting Zip {dataset_zip} to {dataset_extracted}")
    extract_archive(from_path=dataset_zip, to_path=dataset_extracted)

    transforms = AlbumentationTransforms(
        [
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose(),
            A.GaussNoise(p=0.2),
            A.OneOf(
                [
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ],
                p=0.2,
            ),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf(
                [
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.1),
                    A.PiecewiseAffine(p=0.3),
                ],
                p=0.2,
            ),
            A.OneOf(
                [
                    A.CLAHE(clip_limit=2),
                    A.Sharpen(),
                    A.Emboss(),
                    A.RandomBrightnessContrast(),
                ],
                p=0.3,
            ),
            A.HueSaturationValue(p=0.3),
            ToTensorV2(),
        ]
    )

    trainset = ImageFolder(dataset_extracted / "seg_train" / "seg_train", transform=transforms)
    testset = ImageFolder(dataset_extracted / "seg_test" / "seg_test", transform=transforms)

    for path in ["train", "test"]:
        output_dir = git_path / "dataset" / path
        print(f"\t:: Creating Directory {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    print(":: Saving Datasets")
    write_dataset(trainset, git_path / "dataset" / "train")
    write_dataset(testset, git_path / "dataset" / "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # setup git
    print(":: Configuring Git")
    configure_git()

    print(":: Cloning Git")
    repo = clone_dvc_git_repo()

    # extract the input zip file and split into train and test
    print(":: Generate Train Test Split")
    generate_train_test_split()

    print(":: copy data to train")
    subprocess.check_call(
        "cp -r /opt/ml/processing/sagemaker-scene/dataset/train/* /opt/ml/processing/dataset/train",
        shell=True,
    )
    subprocess.check_call(
        "cp -r /opt/ml/processing/sagemaker-scene/dataset/test/* /opt/ml/processing/dataset/test",
        shell=True,
    )

    print(":: Sync Processed Data to Git & DVC")
    sync_data_with_dvc(repo)
