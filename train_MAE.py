import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from torchvision.models import resnet50, ResNet50_Weights


logging.basicConfig(filename='L1Loss_training_log.log', level=logging.INFO)


class CorrelationDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):

        self.corr_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.corr_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.corr_frame.iloc[idx, 0] + '.png')
        image = Image.open(img_name)
        correlation = self.corr_frame.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, correlation

transform = transforms.Compose([
    # transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = CorrelationDataset(csv_file='/home/ai2lab/Desktop/ML_intern/correlation_assignment/train/train_responses.csv', root_dir='/home/ai2lab/Desktop/ML_intern/correlation_assignment/train', transform=transform)
val_dataset = CorrelationDataset(csv_file='/home/ai2lab/Desktop/ML_intern/correlation_assignment/val/val_responses.csv', root_dir='/home/ai2lab/Desktop/ML_intern/correlation_assignment/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)  # Adjust the final layer to output one value

# Change the criterion to L1 Loss
criterion = torch.nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

num_epochs = 30
best_val_loss = float('inf')

train_losses = []
val_losses = []


print("start training")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} Training')
    for inputs, labels in train_progress_bar:
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        
        # 更新进度条的描述信息
        train_progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs} Train Loss: {loss.item():.8f}")
        
    train_loss = running_loss / len(train_loader.dataset)
    
    model.eval()
    val_loss = 0.0
    val_progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} Validation')
    with torch.no_grad():
        for inputs, labels in val_progress_bar:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            val_loss += loss.item() * inputs.size(0)
            # 更新进度条的描述信息
            val_progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs} Val Loss: {loss.item():.8f}")
            
    val_loss /= len(val_loader.dataset)
    
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}')
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}')
    torch.save(model.state_dict(), 'L1_last_checkpoint.pth')
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'L1_best_model_checkpoint.pth')
        print(f"Saved best model at epoch {epoch+1} with Val Loss: {val_loss:.8f}")
        
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('L1_Loss.png')
plt.show()



