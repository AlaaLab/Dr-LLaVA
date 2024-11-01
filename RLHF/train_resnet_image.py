from argparse import ArgumentParser
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
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

# Load ResNet model
class ResNet(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)  # Single label output for binary classification
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
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = model(inputs).squeeze(1)
        loss = self.loss_fn(outputs, labels)
        self.log("val_loss", loss)
        self.log("val_auroc", auroc(outputs, labels.int(), task='binary'))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--label_column', type=str, default='st_elevation')
    args = parser.parse_args()

    model = ResNet()

    # Create Dataset and DataLoader for training and validation 
    train_dataset = ImageDataset('../data/train_resnet_df.csv', '../data/image_folder', label_column=args.label_column)
    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True, pin_memory=True)

    test_dataset = ImageDataset('../data/test_resnet_df.csv', '../data/image_folder', label_column=args.label_column)
    test_dataloader = DataLoader(test_dataset, batch_size=64, num_workers=4, shuffle=False, pin_memory=True)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    trainer = L.Trainer(max_epochs=20, accelerator="auto", callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloader, test_dataloader)
