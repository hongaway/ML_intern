import torch
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd

# Define your dataset path and checkpoint path
dataset_path = '/home/ai2lab/Desktop/ML_intern/correlation_assignment/test'
checkpoint_path = '256best_model_checkpoint.pth'
csv_path = '/home/ai2lab/Desktop/ML_intern/correlation_assignment/test/test_responses.csv'  # CSV 文件路徑

# Load your model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 1)  # Adjusting the final layer based on your training script
model.load_state_dict(torch.load(checkpoint_path))

# Prepare your model for testing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Define your transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ground truth from CSV file into a dictionary
ground_truth_df = pd.read_csv(csv_path)
ground_truth_dict = dict(zip(ground_truth_df['id'], ground_truth_df['corr']))

# Function to predict single image
def predict_image(image_path, ground_truth):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
    prediction = output.item()
    print(f'Image: {image_file}, Prediction: {prediction}, Ground Truth: {ground_truth}')

# Iterate over your test dataset and make predictions
for image_file in os.listdir(dataset_path):
    image_path = os.path.join(dataset_path, image_file)
    ground_truth = ground_truth_dict.get(image_file.split('.')[0])
    if ground_truth is not None:
        predict_image(image_path, ground_truth)
    else:
        print(f"No ground truth found for {image_file}")
