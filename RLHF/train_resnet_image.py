import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.io import read_image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageDataset(Dataset):
    def __init__(self, csv_path, img_dir, label_column, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.label_column = label_column
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{self.df.iloc[idx]['study_id']}.jpeg")
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx][self.label_column]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# Define image transformations without downsampling
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create Dataset and DataLoader for training and validation 
train_dataset = ImageDataset('../data/train_downsampled_df.csv', '../data/image_folder_3lead', label_column='st_elevation', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = ImageDataset('../data/test_downsampled_df.csv', '../data/image_folder_3lead', label_column='st_elevation', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Load ResNet model
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Single label output for binary classification

# model = models.convnext_base(pretrained=True)
# model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 1)  # Single label output for binary classification

# Enable multi-GPU training if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

model = model.to(device)

# Save the first image for reference as PNG
example_image, _ = train_dataset[0]
example_image = transforms.ToPILImage()(example_image)
example_image.save('example_resnet_image.png')

# Loss and optimizer
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

def train_one_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        total_samples += len(labels)
    return total_loss / total_samples

def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze(1)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
    avg_loss = total_loss / total_samples
    auc = roc_auc_score(all_labels, all_outputs)
    accuracy = ((torch.tensor(all_outputs) > 0.5).float() == torch.tensor(all_labels)).float().mean().item()
    positive_share = (torch.tensor(all_outputs) > 0.5).float().mean().item()
    return avg_loss, accuracy, auc, positive_share

# Training Loop
n_epochs = 20
for epoch in tqdm(range(n_epochs)):
    train_loss = train_one_epoch(model, train_dataloader, loss_fn, optimizer)
    val_loss, val_accuracy, val_auc, positive_share = evaluate_model(model, test_dataloader)
    
    # Print metrics
    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, "
          f"Val AUC: {val_auc:.4f}, Positive Share: {positive_share:.4f}")
