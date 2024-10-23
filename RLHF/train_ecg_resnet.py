import os
import glob
import numpy as np
import pandas as pd
import wfdb
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.optim import Adam
from tqdm import tqdm

from scipy import stats

# Define the 'find_all_rec_paths' function
def find_all_rec_paths(input_dir, study_id_set=None):
    """
    Recursively find all ECG record paths by locating .hea files.

    Args:
        input_dir (str): Path to the directory containing ECG files.
        study_id_set (set): Set of study IDs to filter the records.

    Returns:
        list: List of record paths without file extensions.
    """
    rec_paths = []
    if study_id_set is None:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.hea'):
                    rec_path = os.path.join(root, file[:-4])  # Remove '.hea' extension
                    rec_paths.append(rec_path)
    else:
        for root, dirs, files in tqdm(os.walk(input_dir)):
            for file in files:
                if file.endswith('.hea'):
                    try:
                        study_id = int(file[:-4])
                        if study_id in study_id_set:
                            rec_path = os.path.join(root, file[:-4])  # Remove '.hea' extension
                            rec_paths.append(rec_path)
                    except ValueError:
                        # Handle the case where file[:-4] is not an integer
                        pass
    return rec_paths

class ResidualBlock(nn.Module):
    expansion = 1  # Used for scaling the number of output channels in the block

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(ResidualBlock, self).__init__()
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride=1,
            padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Define the downsample layer if needed
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super(ResNet1D, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(
            in_channels=12, out_channels=self.in_channels,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Create residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256 * block.expansion, num_classes)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []

        # Add the first block with potential downsampling
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion

        # Add remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (batch_size, 12, n_samples)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Pass through residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)  # Shape: (batch_size, channels, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, channels)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x.squeeze()

# Define the Dataset class
class ECGDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        record_path = self.df.loc[idx, 'record_path']
        label = self.df.loc[idx, 'st_elevation']
        
        # Load ECG data
        try:
            rd_record = wfdb.rdrecord(record_path)
            signal = rd_record.p_signal  # Shape: (n_samples, n_leads)
            signal = signal.T  # Transpose to (n_leads, n_samples)
            # Zero-mean normalization
            #signal = signal - stats.mode(signal, axis=1, keepdims=True)
            signal = signal - stats.mode(signal, axis=1, keepdims=True).mode
            signal = torch.tensor(signal, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading record {record_path}: {e}")
            signal = torch.zeros((12, 5000), dtype=torch.float32)
        
        label = torch.tensor(label, dtype=torch.float32)
        return signal, label

# Load labels
train_df = pd.read_csv('../data/train_downsampled_df.csv')
test_df = pd.read_csv('../data/test_downsampled_df.csv')

# Get the sets of 'study_id's
train_study_ids = set(train_df['study_id'].astype(int))
test_study_ids = set(test_df['study_id'].astype(int))
all_study_ids = train_study_ids.union(test_study_ids)

# Get all ECG record paths using 'find_all_rec_paths'
ecg_data_dir = '../data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files'
all_record_paths = find_all_rec_paths(ecg_data_dir, study_id_set=all_study_ids)

# Build a mapping from 'study_id' to 'record_path'
record_id_to_path = {}
for rec_path in all_record_paths:
    filename = os.path.basename(rec_path)
    try:
        study_id = int(filename)
        record_id_to_path[study_id] = rec_path
    except ValueError:
        # Handle the case where filename is not an integer
        pass

# Map 'study_id' in the DataFrames to 'record_path'
train_df['record_path'] = train_df['study_id'].astype(int).map(record_id_to_path)
test_df['record_path'] = test_df['study_id'].astype(int).map(record_id_to_path)

# Drop rows with missing 'record_path'
train_df.dropna(subset=['record_path'], inplace=True)
test_df.dropna(subset=['record_path'], inplace=True)

# Create Datasets and DataLoaders
train_dataset = ECGDataset(train_df)
test_dataset = ECGDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


# Instantiate the model
#model = ResNet1D(ResidualBlock, [1, 1, 1])  # 3 residual blocks
model = ResNet1D(ResidualBlock, layers=[1, 1, 1], num_classes=1)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for signals, labels in tqdm(train_loader):
        signals = signals.to(device)
        labels = labels.to(device)
        
        outputs = model(signals)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
    
    # Validation
    model.eval()
    val_losses = []
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            
            outputs = model(signals)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())
    
    val_loss = np.mean(val_losses)
    val_accuracy = accuracy_score(all_labels, np.round(all_preds))
    val_auc = roc_auc_score(all_labels, all_preds)
    positive_share = np.mean(np.round(all_preds))
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {np.mean(train_losses):.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, '
          f'Val AUC: {val_auc:.4f}, Positive Share: {positive_share:.4f}')
