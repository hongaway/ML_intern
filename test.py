import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50
import numpy as np
import argparse
import logging
import os

csv_file = '/home/ai2lab/Desktop/ML_intern/correlation_assignment/test/test_responses.csv'  # 
root_dir = '/home/ai2lab/Desktop/ML_intern/correlation_assignment/test'  # 

# model_path = '/home/ai2lab/Desktop/ML_intern/checkpoint/HuberLoss_best_model_checkpoint.pth'  # 
# model_path = '/home/ai2lab/Desktop/ML_intern/checkpoint/L1_best_model_checkpoint.pth'  # 
model_path = '/home/ai2lab/Desktop/ML_intern/checkpoint/resnet50_best_model_checkpoint.pth'  # 

threshold = 0.01  

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


# load data
test_dataset = CorrelationDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# load model
model = resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 1)  # Adjust the final layer
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# cal acc
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)

        outputs = model(inputs)
        prediction_error = torch.abs(outputs - labels)

        correct_predictions += (prediction_error < threshold).sum().item()
        total_predictions += labels.size(0)

accuracy = correct_predictions / total_predictions * 100
print(f'Accuracy: {accuracy:.2f}%')

logging.basicConfig(filename='test_log.log', level=logging.INFO)
logging.info(f'Model: {model_path}, Accuracy: {accuracy:.2f}%')

