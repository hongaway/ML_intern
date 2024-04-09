import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

csv_file = '/home/ai2lab/Desktop/ML_intern/correlation_assignment/responses.csv'
root_dir = '/home/ai2lab/Desktop/ML_intern/correlation_assignment/images'
train_dir = '/home/ai2lab/Desktop/ML_intern/correlation_assignment/train'
val_dir = '/home/ai2lab/Desktop/ML_intern/correlation_assignment/val'
test_dir = '/home/ai2lab/Desktop/ML_intern/correlation_assignment/test'

df = pd.read_csv(csv_file)

# Split the data set - first divide the data set into 90% training + validation set and 10% test set
train_val, test_data = train_test_split(df, test_size=0.1, random_state=42)

# Then divide the training + validation set into 88.89% training set and 11.11% validation set to get a ratio of 8:1:1
train_data, val_data = train_test_split(train_val, test_size=0.1111, random_state=42) # 约等于0.1 / 0.9

def process_data(data, images_dir, csv_path):
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    data[['id', 'corr']].to_csv(csv_path, index=False)  
    for _, row in data.iterrows():
        file_name = row['id'] + '.png'
        source_path = os.path.join(root_dir, file_name)
        target_path = os.path.join(images_dir, file_name)
        shutil.copy(source_path, target_path)
    print(f'Processed {len(data)} images into {images_dir} and saved CSV to {csv_path}')

process_data(train_data, train_dir, os.path.join(train_dir, 'train_responses.csv'))
process_data(val_data, val_dir, os.path.join(val_dir, 'val_responses.csv'))
process_data(test_data, test_dir, os.path.join(test_dir, 'test_responses.csv'))

