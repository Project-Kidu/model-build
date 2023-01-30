from typing import Any, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.functional import confusion_matrix, f1_score, precision, recall


class Timm_LitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        num_class: int,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        learning_rate: float = 0.1,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_class = num_class

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_class)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_class)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_class)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log(
            "val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            "val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True, sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log(
            "test/loss",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log("hp_metric", self.test_acc, sync_dist=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        preds = torch.cat([tmp["preds"] for tmp in outputs])
        targets = torch.cat([tmp["targets"] for tmp in outputs])

        Precision = precision(preds, targets, task="multiclass", num_classes=self.num_class)
        self.log("test/Precision", Precision, on_step=False, on_epoch=True, sync_dist=True)

        Recall = recall(preds, targets, task="multiclass", num_classes=self.num_class)
        self.log("test/Recall", Recall, on_step=False, on_epoch=True, sync_dist=True)

        F1_Score = f1_score(preds, targets, task="multiclass", num_classes=self.num_class)
        self.log("test/F1_Score", F1_Score, on_step=False, on_epoch=True, sync_dist=True)

        # Confusion matrix calculation
        confusion_mat = confusion_matrix(
            preds, targets, task="multiclass", num_classes=self.num_class
        )

        df_cm = pd.DataFrame(confusion_mat.cpu().numpy(), index=range(6), columns=range(6))
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap="Spectral", fmt="d").get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

        confusion_mat = confusion_mat.type(torch.float32)
        self.log("test/confusion_matrix/00", confusion_mat[0][0], on_epoch=True)
        self.log("test/confusion_matrix/01", confusion_mat[0][1], on_epoch=True)
        self.log("test/confusion_matrix/02", confusion_mat[0][2], on_epoch=True)
        self.log("test/confusion_matrix/03", confusion_mat[0][3], on_epoch=True)
        self.log("test/confusion_matrix/04", confusion_mat[0][4], on_epoch=True)
        self.log("test/confusion_matrix/05", confusion_mat[0][5], on_epoch=True)

        self.log("test/confusion_matrix/10", confusion_mat[1][0], on_epoch=True)
        self.log("test/confusion_matrix/11", confusion_mat[1][1], on_epoch=True)
        self.log("test/confusion_matrix/12", confusion_mat[1][2], on_epoch=True)
        self.log("test/confusion_matrix/13", confusion_mat[1][3], on_epoch=True)
        self.log("test/confusion_matrix/14", confusion_mat[1][4], on_epoch=True)
        self.log("test/confusion_matrix/15", confusion_mat[1][5], on_epoch=True)

        self.log("test/confusion_matrix/20", confusion_mat[2][0], on_epoch=True)
        self.log("test/confusion_matrix/21", confusion_mat[2][1], on_epoch=True)
        self.log("test/confusion_matrix/22", confusion_mat[2][2], on_epoch=True)
        self.log("test/confusion_matrix/23", confusion_mat[2][3], on_epoch=True)
        self.log("test/confusion_matrix/24", confusion_mat[2][4], on_epoch=True)
        self.log("test/confusion_matrix/25", confusion_mat[2][5], on_epoch=True)

        self.log("test/confusion_matrix/30", confusion_mat[3][0], on_epoch=True)
        self.log("test/confusion_matrix/31", confusion_mat[3][1], on_epoch=True)
        self.log("test/confusion_matrix/32", confusion_mat[3][2], on_epoch=True)
        self.log("test/confusion_matrix/33", confusion_mat[3][3], on_epoch=True)
        self.log("test/confusion_matrix/34", confusion_mat[3][4], on_epoch=True)
        self.log("test/confusion_matrix/35", confusion_mat[3][5], on_epoch=True)

        self.log("test/confusion_matrix/40", confusion_mat[4][0], on_epoch=True)
        self.log("test/confusion_matrix/41", confusion_mat[4][1], on_epoch=True)
        self.log("test/confusion_matrix/42", confusion_mat[4][2], on_epoch=True)
        self.log("test/confusion_matrix/43", confusion_mat[4][3], on_epoch=True)
        self.log("test/confusion_matrix/44", confusion_mat[4][4], on_epoch=True)
        self.log("test/confusion_matrix/45", confusion_mat[4][5], on_epoch=True)

        self.log("test/confusion_matrix/50", confusion_mat[5][0], on_epoch=True)
        self.log("test/confusion_matrix/51", confusion_mat[5][1], on_epoch=True)
        self.log("test/confusion_matrix/52", confusion_mat[5][2], on_epoch=True)
        self.log("test/confusion_matrix/53", confusion_mat[5][3], on_epoch=True)
        self.log("test/confusion_matrix/54", confusion_mat[5][4], on_epoch=True)
        self.log("test/confusion_matrix/55", confusion_mat[5][5], on_epoch=True)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters(), lr=self.learning_rate)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = Timm_LitModule(None, None, None)
