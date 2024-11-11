from argparse import ArgumentParser
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import lightning as L
from tqdm import tqdm
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
        self.log("train_auroc", auroc(outputs, labels.int(), task='binary'))
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = model(inputs).squeeze(1)
        loss = self.loss_fn(outputs, labels)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_auroc", auroc(outputs, labels.int(), task='binary'), sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--label_column', type=str)
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()

    batch_size = 64

    model = ResNet.load_from_checkpoint(args.checkpoint)

    # Create Dataset and DataLoader for training and validation 
    dataset = ImageDataset('../data/mimic-acute-mi_modelling.csv', '../data/image_folder', label_column=args.label_column)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    trainer = L.Trainer()
    trainer.validate(model, dataloader, verbose=True)

    model.eval().cuda()
    results = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            x, y = batch
            y_hat = model(x.cuda())
            results.extend(y_hat.flatten().tolist())

    with open(args.output_file, 'w') as f:
        f.writelines([ f"{y}\n" for y in results ])
