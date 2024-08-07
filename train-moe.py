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
from src.distil_loss import DistilKL, Similarity, KDLoss
from src.model import AdapterResnet1, AdapterResnet2 ,AdapterResnet3, SepConv, CustomHead, Block, MoE_ResNet18
from src.customblock import CBAM
from helper import Fer2013DataModule , Cifar100DataModule

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
        distil_temp: float = 4.0  # type: float
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
        self.save_hyperparameters(ignore=["model"])


        for i in range(5):
            self.__setattr__(f"train_acc{i+1}", torchmetrics.Accuracy(task="multiclass", num_classes=num_classes))  # type: ignore
            self.__setattr__(f"val_acc{i+1}", torchmetrics.Accuracy(task="multiclass", num_classes=num_classes))  # type: ignore
            self.__setattr__(f"test_acc{i+1}", torchmetrics.Accuracy(task="multiclass", num_classes=num_classes))  # type: ignore
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits[4], true_labels)
        predicted_labels = []
        for i in range(4):
            loss += F.cross_entropy(logits[i], true_labels)* (1-self.loss_alpha)
            kd_loss = DistilKL(T=self.distil_temp)(logits[i], logits[4])   
            loss += kd_loss

            predicted_labels.append(torch.argmax(logits[i], dim=1))
        
        predicted_labels.append(torch.argmax(logits[4], dim=1))
        # loss = F.cross_entropy(logits, true_labels)
        # predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss)
        for i in range(5):
            self.log(f"train_acc{i+1}", self.__getattr__(f"train_acc{i+1}")(predicted_labels[i], true_labels),on_epoch=True ,on_step=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True)
        for i in range(5):
            self.log(f"val_acc{i+1}", self.__getattr__(f"val_acc{i+1}")(predicted_labels[i], true_labels),on_epoch=True ,on_step=False, sync_dist=True)
            
        return loss

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        for i in range(5):
            self.log(f"test_acc{i+1}", self.__getattr__(f"test_acc{i+1}")(predicted_labels[i], true_labels),on_epoch=True, on_step=False, sync_dist=True)
        self.log("F1 score", self.f1(predicted_labels[4], true_labels),on_epoch=True, on_step=False, sync_dist=True)
        self.log("Recall", self.recall(predicted_labels[4], true_labels),on_epoch=True, on_step=False, sync_dist=True)
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

def train():
    CIFAR100MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR100STD = [0.2675, 0.2565, 0.2761]
    train_transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
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
        batch_size=256, 
        train_transform=train_transform, 
        test_transform=test_transform,
        num_workers=4
    )
    pytorch_model = MoE_ResNet18(num_classes=100)
    lightning_model = LightningFerModel(model=pytorch_model,learning_rate=0.1, optimizer="sgd",num_classes=100, lr_scheduler="cosine_annealingLR", max_epoch=250, loss_alpha=0.5, distil_temp=4.0)
    callbacks = [ModelCheckpoint(save_top_k=1, mode="max", monitor="val_acc4"), LearningRateMonitor(logging_interval="epoch")]

    trainer = L.Trainer(
        max_epochs=250,
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        callbacks=callbacks,
        log_every_n_steps=100,
        logger=WandbLogger(project="BYOT"),
        # deterministic=True,
        # enable_model_summary=True
    )

    trainer.fit(model=lightning_model, datamodule=dm)
    # trainer.test(lightning_model, datamodule=dm,ckpt_path='best')

if __name__ == '__main__':
    train()
