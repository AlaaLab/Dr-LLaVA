from argparse import ArgumentParser
from collections import Counter
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
import wandb
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from torchmetrics.functional.classification import auroc
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, csv_path, img_dir, label_column):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.label_column = label_column
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{self.df.iloc[idx]['study_id']}.jpeg")
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx][self.label_column]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

class ClassifierModel(L.LightningModule):
    def __init__(self, model: str, freeze_encoder: bool):
        super().__init__()
        if model == 'resnet':
            self.model = models.resnet50(pretrained=True)
        elif model == 'vit':
            self.model = models.vit_l_16(pretrained=True)
        else:
            raise ValueError('Model must be reset or vit.')

        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        if model == 'resnet':
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)
        if model == 'vit':
            self.model.heads.head = torch.nn.Linear(self.model.heads.head.in_features, 1)

        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        inputs, labels = batch
        outputs = model(inputs).squeeze(1)
        loss = self.loss_fn(outputs, labels)
        self.log("train_loss", loss)
        self.log("train_auroc", auroc(outputs, labels.int(), task='binary'))
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = model(inputs).squeeze(1)
        loss = self.loss_fn(outputs, labels)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_auroc", auroc(outputs, labels.int(), task='binary'), sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--label_column', type=str)
    parser.add_argument('--freeze-encoder', type=bool)
    parser.add_argument('--balance-training', type=bool)
    args = parser.parse_args()

    batch_size = 48

    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project='Dr-LLaVA')

    model = ClassifierModel(model=args.model, freeze_encoder=args.freeze_encoder)

    # Create Dataset and DataLoader for training and validation 
    train_dataset = ImageDataset('../data/train_downsampled_df.csv', '../data/image_folder_clean', label_column=args.label_column)
    if args.balance_training:
        class_cnts = Counter(train_dataset.df[train_dataset.label_column].tolist())
        weigths = [ class_cnts.total() / class_cnts[i] for i in train_dataset.df[train_dataset.label_column].tolist()]
        sampler = WeightedRandomSampler(weigths, len(train_dataset.df))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    test_dataset = ImageDataset('../data/test_downsampled_df.csv', '../data/image_folder_clean', label_column=args.label_column)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    trainer = L.Trainer(max_epochs=5, accelerator="auto", logger=wandb_logger)
    trainer.fit(model, train_dataloader, test_dataloader)

    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(vars(args))
