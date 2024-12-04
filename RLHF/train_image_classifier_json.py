from argparse import ArgumentParser
from collections import Counter
import os
import json
from sklearn.preprocessing import OrdinalEncoder
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
import wandb
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from torchmetrics.functional.classification import multiclass_accuracy
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, json_path, img_dir, test=False):
        if test:
            with open(json_path) as f:
                self.data = [ json.loads(line) for line in f.readlines()]
        else:
            with open(json_path) as f:
                self.data = json.load(f)
        
        if test:
            self.labels = OrdinalEncoder(dtype='int64').fit_transform([ [s['category']] for s in self.data ]).flatten()
        else:
            self.labels = OrdinalEncoder(dtype='int64').fit_transform([ [s['diagnosis']] for s in self.data ]).flatten()
        self.num_classes = int(self.labels.max() + 1)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data[idx]['image'])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class ClassifierModel(L.LightningModule):
    def __init__(self, model: str, freeze_encoder: bool, num_class: int):
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
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_class)
        if model == 'vit':
            self.model.heads.head = torch.nn.Linear(self.model.heads.head.in_features, num_class)

        self.num_class = num_class
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        inputs, labels = batch
        outputs = model(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log("train_loss", loss)
        self.log("train_accuracy", multiclass_accuracy(outputs, labels.int(), self.num_class))
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = model(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_accuracy", multiclass_accuracy(outputs, labels.int(), self.num_class), sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--image_folder', type=str, default='../data/image_folder')
    parser.add_argument('--train_json', type=str, default='../data/train_conversations_with_preds.json')
    parser.add_argument('--test_json', type=str, default='../data/test_conversations_with_preds.json')
    parser.add_argument('--freeze-encoder', action='store_true')
    parser.add_argument('--balance-training', action='store_true')
    args = parser.parse_args()

    batch_size = 48

    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project='Dr-LLaVA')

    # Create Dataset and DataLoader for training and validation 
    train_dataset = ImageDataset(args.train_json, args.image_folder)
    model = ClassifierModel(args.model, args.freeze_encoder, train_dataset.num_classes)

    if args.balance_training:
        class_cnts = Counter(train_dataset.labels)
        weigths = [ class_cnts.total() / class_cnts[i] for i in train_dataset.labels ]
        sampler = WeightedRandomSampler(weigths, len(train_dataset.labels))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    test_dataset = ImageDataset(args.test_json, args.image_folder)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    trainer = L.Trainer(max_epochs=10, accelerator="auto", logger=wandb_logger)
    trainer.fit(model, train_dataloader, test_dataloader)

    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(vars(args))
