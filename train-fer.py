import argparse
import os
import sys

# import pytorch_lightning as pl
import lightning as L

import torch
import torch.nn.functional as F
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import timm
import torchmetrics
from torchvision import datasets, transforms
from torchvision.transforms import v2 as v2_transforms
import optuna
# from  optuna.integration.wandb import WeightsAndBiasesCallback

from src.distil_loss import DistilKL, Similarity, KDLoss
from src.model import AdapterResnet1, AdapterResnet2 ,AdapterResnet3, SepConv, CustomHead, Block
from src.customblock import CBAM
from helper import Fer2013DataModule , Cifar100DataModule


class FerModel(nn.Module):
    def __init__(self, model_name, num_classes=7):
        super(FerModel, self).__init__()
        self.backbone = timm.create_model(model_name=model_name, pretrained=False, features_only=True)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.maxpool = nn.Identity()

        self.adapter1 = AdapterResnet1(Block, CBAM, num_classes = num_classes, pool_size=(1, 1),features=True)
        self.adapter2 = AdapterResnet2(Block, CBAM, num_classes = num_classes, pool_size=(1, 1),features=True)
        self.adapter3 = AdapterResnet3(Block, CBAM, num_classes=num_classes, pool_size=(1, 1),features=True)


        self.classifier = CustomHead(in_planes=512, num_classes=num_classes, pool_size=(1, 1),features=True)
    def forward(self, x):
        x = self.backbone(x)
        fea1 = x[1]#16x16
        fea2 = x[2]#8x8
        fea3 = x[3]#4x4
        logit1,f1 = self.adapter1(fea1)
        logit2,f2 = self.adapter2(fea2)
        logit3,f3 = self.adapter3(fea3)
        logit4,f4 = self.classifier(x[4])
        return [logit1, logit2, logit3, logit4],[f1, f2, f3, f4]



class LightningFerModel(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        optimizer: str,
        lr_scheduler: str,
        max_epoch: int,
        num_classes: int = 100,  # type: int
        loss_alpha: float = 0.3,  # type: float
        distil_temp: float = 3.0,  # type: float
        feature_weight: float = 0.03
    ) -> None:
        """
        Initialize a LightningFerModel object.

        Args:
            model (nn.Module): The model to use.
            learning_rate (float): The learning rate for the optimizer.
            optimizer (str): The optimizer to use.
            lr_scheduler (str): The learning rate scheduler to use.
            max_epoch (int): The maximum number of training epochs.
            num_classes (int, optional): The number of classes for the model. Defaults to 7.
            loss_alpha (float, optional): The alpha value for the loss function. Defaults to 0.3.
            distil_temp (float, optional): The temperature value for the distillation loss. Defaults to 3.0.
            feature_weight (float, optional): The weight value for the feature loss. Defaults to 0.03.
        Returns:
            None
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.lr_scheduler = lr_scheduler
        self.learning_rate = learning_rate
        self.loss_alpha = loss_alpha
        self.distil_temp = distil_temp
        self.feature_weight = feature_weight
        self.save_hyperparameters(ignore=["model"])


        for i in range(4):
            self.__setattr__(f"train_acc{i+1}", torchmetrics.Accuracy(task="multiclass", num_classes=num_classes))  # type: ignore
            self.__setattr__(f"val_acc{i+1}", torchmetrics.Accuracy(task="multiclass", num_classes=num_classes))  # type: ignore
            self.__setattr__(f"test_acc{i+1}", torchmetrics.Accuracy(task="multiclass", num_classes=num_classes))  # type: ignore
        
    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits, fea = self(features)
        loss = F.cross_entropy(logits[3], true_labels)
        predicted_labels = []
        for i in range(3):
            loss += F.cross_entropy(logits[i], true_labels)* (1-self.loss_alpha)
            _,kd_loss = KDLoss(alpha=self.loss_alpha, temp=self.distil_temp)(logits[i], logits[3].detach()) 
            loss += kd_loss
            feature_loss = torch.dist(fea[i], fea[3].detach(), 2)
            loss += feature_loss*self.feature_weight
            predicted_labels.append(torch.argmax(logits[i], dim=1))
        
        predicted_labels.append(torch.argmax(logits[3], dim=1))
        # loss = F.cross_entropy(logits, true_labels)
        # predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("train_loss", loss)
        for i in range(4):
            self.log(f"train_acc{i+1}", self.__getattr__(f"train_acc{i+1}")(predicted_labels[i], true_labels),on_epoch=True ,on_step=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True)
        for i in range(4):
            self.log(f"val_acc{i+1}", self.__getattr__(f"val_acc{i+1}")(predicted_labels[i], true_labels),on_epoch=True ,on_step=False, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        for i in range(4):
            self.log(f"test_acc{i+1}", self.__getattr__(f"test_acc{i+1}")(predicted_labels[i], true_labels),on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9 , weight_decay=5e-4)
        elif self.optimizer == "adam_wav2vec2.0":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.98), eps=1e-6) # wav2vec2,0's optimizer set up on Adam. (Need to verify)
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-6) # distilBert's optimzer setup on Adam
        elif self.optimizer == "adamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.98), eps=1e-4)
        else:
            raise NotImplementedError
        if self.lr_scheduler == "":
            return optimizer
        elif self.lr_scheduler == "linear_decay_with_warm_up":
            def lr_lambda(current_epoch): # Copied from https://github.com/huggingface/transformers/blob/master/src/transformers/optimization.py
                if current_epoch < self.num_lr_warm_up_epoch:
                    return float(current_epoch+1) / float(max(1, self.num_lr_warm_up_epoch)) # current_epoch+1 to prevent lr=0 in epoch 0
                return max(
                    0.0, float(self.max_epoch - current_epoch) / float(max(1, self.max_epoch - self.num_lr_warm_up_epoch)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif self.lr_scheduler == "cosine_warmup_anneal":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=9, T_mult=1, eta_min=1e-6)
        elif self.lr_scheduler == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(self.train_dataloader())//256, epochs=self.max_epoch)
        elif self.lr_scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
        elif self.lr_scheduler == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 90, 120], gamma=0.2)
        elif self.lr_scheduler == "cosine_annealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epoch, eta_min=1e-6)
        elif self.lr_scheduler == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.75, patience=5, verbose=True)
        else:
            raise NotImplementedError

        return [optimizer], [scheduler]

ImageNetMEAN = [0.485, 0.456, 0.406]
ImageNetSTD = [0.229, 0.224, 0.225]
CIFAR100MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100STD = [0.2675, 0.2565, 0.2761]
def train():
    train_transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100MEAN, CIFAR100STD),
            
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100MEAN, CIFAR100STD),
        ]
    )
    L.seed_everything(2024)
    # dm = Fer2013DataModule(
    #     data_path="/kaggle/input/fer2013/train",
    #     test_path="/kaggle/input/fer2013/test",
    #     height_width=(32,32),
    #     batch_size=256, 
    #     train_transform=train_transform, 
    #     test_transform=test_transform,
    #     num_workers=4
    # )
    dm = Cifar100DataModule(
        data_path="./",
        height_width=(32,32),
        batch_size=256, 
        train_transform=train_transform, 
        test_transform=test_transform,
        num_workers=4
    )
    pytorch_model = FerModel(model_name='resnet18',num_classes=100)
    lightning_model = LightningFerModel(model=pytorch_model, learning_rate=0.1, optimizer="sgd", lr_scheduler="cosine_annealingLR", max_epoch=300, num_classes=100, loss_alpha=0.3, distil_temp=3.0, feature_weight=0.03)
    callbacks = [ModelCheckpoint(save_top_k=1, mode="max", monitor="val_acc4"), LearningRateMonitor(logging_interval="epoch")]

    trainer = L.Trainer(
        max_epochs=300,
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        callbacks=callbacks,
        log_every_n_steps=10,
        logger=WandbLogger(project="BYOT"),
        # deterministic=True,
        # enable_model_summary=True
    )

    trainer.fit(model=lightning_model, datamodule=dm)
    trainer.test(lightning_model, datamodule=dm,ckpt_path='best')

def objective(trial):
    train_transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100MEAN, CIFAR100STD),
            
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100MEAN, CIFAR100STD),
        ]
    )
    L.seed_everything(2024)
    dm = Cifar100DataModule(
        data_path="./",
        height_width=(32,32),
        batch_size=512, 
        train_transform=train_transform, 
        test_transform=test_transform,
        num_workers=4
    )

    pytorch_model = FerModel(model_name='resnet18',num_classes=100)
    loss_alpha = trial.suggest_float("loss_alpha", 0.1, 0.9)
    distil_temp = trial.suggest_float("distil_temp", 1.0, 5.0)
    feature_weight = trial.suggest_float("feature_weight", 0.01, 0.1)
    lightningmodel = LightningFerModel(model = pytorch_model, learning_rate=0.1, optimizer="sgd", lr_scheduler="cosine_annealingLR", max_epoch=100, num_classes=100, loss_alpha=loss_alpha, distil_temp=distil_temp, feature_weight=feature_weight)
    Trainer = L.Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        callbacks=[ModelCheckpoint(save_top_k=1, mode="max", monitor="val_acc4"), LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=10,
        # logger=WandbLogger(**wandb_kwargs),
        # deterministic=True,
        # enable_model_summary=True
    )
    Trainer.fit(model=lightningmodel, datamodule=dm)
    return Trainer.callback_metrics["val_acc4"].item()

if __name__ == '__main__':
    # wandb_kwargs = {
    #     "project": "BYOT",
    # }
    # wandbc = WeightsAndBiasesCallback(metric_name="val_acc4",wandb_kwargs=wandb_kwargs)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)
    print("Number of finished trials: {}".format(len(study.trials)))
    print(study.best_trial)
    trial = study.best_trial

    print("Best trial: score {}, params {}".format(trial.value, trial.params))
