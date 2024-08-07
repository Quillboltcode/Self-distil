import lightning as L
import torch
import torch.nn.functional as F
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import timm
import torchmetrics
from torchvision import datasets, transforms
# from torchvision.transforms import v2 as v2_transforms
from src.model import FGW, FGWLinear
import argparse
# import os
from helper import Fer2013DataModule, FerPlusDataModule

class Resnet34Fer(nn.Module):
    def __init__(self, model_name,pretrained=False, num_classes=7):
        super().__init__()
        self.resnet = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet(x)
    

class LightningFer(L.LightningModule):
    def __init__(self, model, learning_rate=1e-3,num_classes=7,label_smoothing=0.1,other_hyperparameters={},):
        super().__init__()
        
        self.model = model
        self.learning_rate = learning_rate
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass",num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass",num_classes=num_classes)
        self.label_smoothing = label_smoothing
        self.other_hyperparameters = other_hyperparameters
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx):
        features, true_labels = batch
        logits = self.forward(features)
        loss = F.cross_entropy(logits, true_labels,label_smoothing=self.label_smoothing)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch, batch_idx)

        self.log("train_loss", loss)
        self.train_acc(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch, batch_idx)

        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch, batch_idx)

        self.log("test_loss", loss, prog_bar=True)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9 , weight_decay=1e-4, nesterov=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',factor=0.75,patience=5, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    
def train(model_name="resnet18",dataset_name="fer2013",batch_size=128,target_size=48,crop_size=32):
    # Training settings
    dropout = 0.5
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    GRAY_MEAN = 0
    GRAY_STD = 255
    target_size = 48
    crop_size = 32
    num_classes = 7
    ###########################################################
    # v1 agumentations from Hung
    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(target_size, scale=(0.8, 1.2)),
    #     transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    # ])
    # test_and_val_transform = transforms.Compose([
    #     transforms.Resize((target_size, target_size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    # ])
    #############################################################3
    # v2 agumentations
    # train_transform = transforms.Compose([
    #                 transforms.Grayscale(),
    #                 transforms.RandomResizedCrop(target_size, scale=(0.8, 1.2)),
    #                 transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
    #                 transforms.RandomHorizontalFlip(),
    #                 transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize(mean=GRAY_MEAN, std=GRAY_STD)
    #             ])

    # test_and_val_transform = transforms.Compose([
    #     transforms.Grayscale(),
    #     transforms.Resize((target_size, target_size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(GRAY_MEAN,), std=(GRAY_STD,)),
    # ])
    ####################################################################2
    # EmoNeXt agumentations
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Grayscale(),
            transforms.Resize(target_size),
            transforms.RandomRotation(degrees=20),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(target_size),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(target_size),
            transforms.TenCrop(crop_size),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops]
                )
            ),
            transforms.Lambda(
                lambda crops: torch.stack([crop.repeat(3, 1, 1) for crop in crops])
            ),
        ]
    )
    if dataset_name == "fer2013":
        
        dm = Fer2013DataModule(
        train_path="/kaggle/input/fer2013/org_fer2013/train",
        val_path="/kaggle/input/fer2013/org_fer2013/val",
        test_path="/kaggle/input/fer2013/org_fer2013/test",
        height_width=(target_size, target_size),
        batch_size=batch_size, 
        train_transform=train_transform, 
        test_transform=val_transform,
        num_workers=4
        )
    elif dataset_name == "ferplus":
        num_classes = 8
        dm = FerPlusDataModule(
        train_path="/kaggle/input/fer-plus/fer_plus/train",
        val_path="/kaggle/input/fer-plus/fer_plus/val",
        test_path="/kaggle/input/fer-plus/fer_plus/test",
        height_width=(target_size, target_size),
        batch_size=batch_size, 
        train_transform=train_transform, 
        test_transform=val_transform,
        num_workers=4
        )
    
    logger = WandbLogger(project="BYOT")
    hyper_params = {
        "learning_rate": 0.001,
        "batch_size": batch_size,
        "num_classes": num_classes,
        "target_size": target_size,
        "crop_size": crop_size,
        "model_name": model_name,
        "optimizer": "SGD",
        "lr_scheduler": "ReduceLROnPlateau",
    }
    # model = Resnet34Fer(model_name=model_name,pretrained=False, num_classes=num_classes)
    model = FGW(in_channels=3, num_classes=num_classes)
    lightning_model = LightningFer(model=model, learning_rate=0.001,num_classes=num_classes,other_hyperparameters=hyper_params)

    trainer = L.Trainer(
        max_epochs=100,
        devices=2,
        strategy="ddp",
        callbacks=[ModelCheckpoint(save_top_k=1, mode="min", monitor="val_loss"), 
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor="val_loss", patience=7, mode="min")],
        logger=logger,
    )

    trainer.fit(lightning_model, dm)
    
    trainer.test(lightning_model, dm, ckpt_path="best")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BYOT')
    parser.add_argument('--dataset', type=str, default="fer2013")
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--target_size', type=int, default=236)
    parser.add_argument('--crop_size', type=int, default=224)
    args = parser.parse_args()
    train(args.model,args.dataset,args.batch_size)