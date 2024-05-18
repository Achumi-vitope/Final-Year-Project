import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

from model import Net

import random

from torchvision.models import ViT_B_16_Weights, vit_b_16
import os
from tqdm import tqdm

from PIL import Image

import torch.nn as nn


class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, path

def Q_pre_processing(dir_):
    transform =  transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = CustomImageFolder(dir_, transform=transform)
    return dataset

def D_pre_processing(dir_):
    pretrained_vit_weights = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    transform = pretrained_vit_weights.transforms()  # This should handle the correct transformation
    dataset = CustomImageFolder(dir_, transform=transform)
    return dataset

def show_images(query_image_path, retrieved_image_paths, dataset_labels):
    plt.figure(figsize=(12, 12))  # Adjust figure size as needed
        # Load and display the query image
    query_image = Image.open(query_image_path)
    plt.subplot(5, 5, 1)
    plt.imshow(query_image)
    plt.title(f"Query Image\n{os.path.basename(query_image_path)}")
    plt.axis('off')
    
    # Load and display the retrieved images
    for i, image_path in enumerate(retrieved_image_paths, start=2):
        retrieved_image = Image.open(image_path)
        plt.subplot(5, 5, i)
        plt.imshow(retrieved_image)
        # Get the label of the retrieved image
        label = dataset_labels[retrieved_image_paths.index(image_path)]
        # Get the original image name
        original_image_name = os.path.basename(image_path)
        plt.title(original_image_name)
        plt.axis('off')
        
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
    

def extract_features(dataloader, save_path, model_path, num_classes=2,):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(num_classes)  # Initialize the model with the required number of classes
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    features_list = []
    labels_list = []
    file_path_list = []

    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, total=len(dataloader)):
            images = images.to(device)
            outputs = model(images)
            features = outputs.detach().cpu().numpy()  # Detach and move to CPU for numpy conversion
            features_list.append(features)
            labels_list.append(labels.numpy())
            file_path_list.extend(paths)

    # Optionally, save the extracted features, labels, and paths to files
    features_array = np.concatenate(features_list)
    labels_array = np.concatenate(labels_list)
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "features.npy"), features_array)
    np.save(os.path.join(save_path, "labels.npy"), labels_array)
    with open(os.path.join(save_path, "paths.txt"), "w") as f:
        for path in file_path_list:
            f.write("%s\n" % path)

    print("Features, labels, and paths have been saved successfully to:", save_path)
    
def save_feature_to_dir(query_path, database_path, model_path):
    q_save_dirs = ["query_features/40x/", "query_features/100x/", "query_features/200x/", "query_features/400x/"]
    d_save_dirs = ["database_features/40x/", "database_features/100x/", "database_features/200x/", "database_features/400x/"]
    for i in range(len(query_path)):
        pre_pro = Q_pre_processing(query_path[i])
        pre_pro_loader = DataLoader(pre_pro, batch_size=3, shuffle=False)
        extract_features(pre_pro_loader, q_save_dirs[i], model_path[i])
        
    for i in range(len(database_path)):
        pre_pro = D_pre_processing(database_path[i])
        pre_pro_loader = DataLoader(pre_pro, batch_size=30, shuffle=False)
        extract_features(pre_pro_loader, d_save_dirs[i], model_path[i])


def euclidean_similarity(query_features, dataset_features):
    # Compute Euclidean distance between query image features and dataset features
    distances = np.linalg.norm(dataset_features - query_features, axis=1)
    return distances


def loss_function(feature_distances, dataset_features, query_features, margin):
    losses = []
    for distance in feature_distances:
        if distance < margin:
            loss = (margin - distance) ** 2
        else:
            loss = 0
        losses.append(loss)
    return losses
# def loss_function(feature_distances, dataset_features, query_features, margin):
#     similarities = []
#     for distance in feature_distances:
#         # Calculate similarity as an inverse of the distance, adjusted by the margin
#         similarity = max(0, margin - distance)
#         similarities.append(similarity)
#     return similarities


# def loss_function(feature_distances, labels, q_label, margin):
#     losses = []
#     for i, distance in enumerate(feature_distances):
#         y_ij = 1 if labels[i] == q_label else 0 
#         if y_ij == 1:
#             loss = distance ** 2
#         else:
#             loss = max(0, margin - distance) ** 2
#         losses.append(loss)
#     return losses

def calculate_metrics(query_label, top_k_indices, dataset_labels):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    query_label = int(query_label)  # Ensure query_label is integer
    
    for i in top_k_indices:
        if dataset_labels[i] == query_label:
            true_positives += 1
        else:
            false_positives += 1
    
    false_negatives = len(top_k_indices) - true_positives
    
    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    accuracy = true_positives / len(top_k_indices) if len(top_k_indices) > 0 else 0
    
    return precision, recall, accuracy

def main():
    
    query_path = ["query/40x", "query/100x", "query/200x", "query/400x"]
    database_path = ["database/40x", "database/100x", "database/200x", "database/400x"]
    model_path = ["model_/1_40X_model.pth", "model_/1_100X_model.pth", "model_/1_200X_model.pth", "model_/1_400X_model.pth"]
    if not os.path.exists('database_features') or not os.listdir('query_features'):
        save_feature_to_dir(query_path, database_path, model_path)

    running = True
    choice = None
    feature_dataset_path = None
    label_dataset_path = None
    dataset_path_path = None
    magnification = None
    
    while running:
        choice = int(input("CBMIR on - 1. Client 1 | 2. Client 2 | 3. Client 3 | 4. Client 4: "))
        if 1 <= choice <= 4:
            magnification = ['40x', '100x', '200x', '400x'][choice - 1]
            feature_dataset_path = f'database_features/{magnification}/features.npy'
            label_dataset_path = f'database_features/{magnification}/labels.npy'
            dataset_path_path = f'database_features/{magnification}/paths.txt'
            magnification = magnification
            running = False
        else:
            print("Undefined input.")
            
    
    
    with open(dataset_path_path, "r") as f:
        dataset_paths = f.read().splitlines()

    query_datas_path = f'query_features/{magnification}/features.npy'
    query_label_path = f'query_features/{magnification}/labels.npy'
    query_path_path = f'query_features/{magnification}/paths.txt'
    with open(query_path_path, "r") as f:
        query_paths = f.read().splitlines()

    query_feature = np.load(query_datas_path)
    query_label = np.load(query_label_path)

    dataset_features = np.load(feature_dataset_path)
    dataset_labels = np.load(label_dataset_path)
    
    print(len(dataset_features))

    dataset_unique_labels = np.unique(dataset_labels)
    query_unique_labels = np.unique(query_label)

    print("Unique Dataset labels:", dataset_unique_labels)
    print("Number of unique Dataset labels:", len(dataset_unique_labels)) 

    print("Unique Query labels:", query_unique_labels)
    print("Number of unique Query labels:", len(query_unique_labels))

    total_query = len(query_feature)
    print(f"Total number of query images: {total_query}")
    acc_ = []
    pr = []
    f1 = []
    # for  i in range(1, total_query):
    query_index = 10
    query_image_features = query_feature[query_index].reshape(1, -1)
    query_image_label = query_label[query_index]

    top_k = 5
    feature_distances = euclidean_similarity(query_image_features, dataset_features)

    margin = 9.0
    losses = loss_function(feature_distances, dataset_features, query_image_features, margin)

    similarities = -np.array(losses) 
    
    top_k_indices = np.argsort(similarities)[:top_k]

    num_pred = 0
    top_k_labels = [dataset_labels[i] for i in top_k_indices]
    for label in top_k_labels:
        if label == query_image_label:
            num_pred += 1
            
    print(f"Query Label: {query_image_label}")
    # for i in top_k_indices:
    #     print(f"Index: {i}, Label: {dataset_labels[i]}")
    precision, recall, accuracy = calculate_metrics(query_image_label, top_k_indices, dataset_labels)
    acc = num_pred / top_k
    print("Accuracy: ",acc)
    print("Precision:", precision)
    print("Recall:", recall)
    #     acc_.append(acc)
    #     pr.append(precision)
    #     f1.append(f1)
    # acc_avg = 0
    # for i in range(len(acc_)):
    #     print(acc_[i])
    #     acc_avg += acc_[i]
    

    # print(acc_avg)
    

if  __name__ == "__main__":
    main()
