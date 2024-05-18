import torch
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ViT_B_16_Weights
from sklearn.utils import resample
import numpy as np
import torch.nn.functional as F
import os
from tqdm import tqdm 
import matplotlib.pyplot as plt

def plot_class_distribution(benign_count, malignant_count, save_path, filename, magnification):
    # Labels for the sections of our bar chart
    labels = ['Benign', 'Malignant']
    
    # The values for each section of the bar chart
    sizes = [benign_count, malignant_count]
    
    # The colors for each section of the bar chart
    colors = ['#ff9999','#66b3ff']
    
    fig, ax = plt.subplots()
    bars = ax.bar(labels, sizes, color=colors)
    
    # Adding the text on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval}', va='bottom')  # va: vertical alignment
    
    # Adding magnification to the title
    ax.set_title(f'Class Distribution at {magnification} Magnification')
    
    # Adding labels to the axes
    ax.set_xlabel('Class Type')
    ax.set_ylabel('Number of Samples')
    
    # Save the plot to a file
    plt.savefig(f"{save_path}/{filename}")
    plt.close(fig)  # Close the figure to free up memory





def check_class_sizes(trainloader):
    # Initialize counters for each class label
    count_0 = 0
    count_1 = 0
    for _, batch_labels in trainloader:
        # Convert batch_labels to tensor if not already
        batch_labels = torch.tensor(batch_labels) if not isinstance(batch_labels, torch.Tensor) else batch_labels
        count_0 += (batch_labels == 0).sum().item()
        count_1 += (batch_labels == 1).sum().item()
    
    # print(f"Class 0 size: {count_0}")
    # print(f"Class 1 size: {count_1}")
    return count_0, count_1


def totens(path_):
    pretrained_vit_weights = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    transform = pretrained_vit_weights.transforms()
    transformed = ImageFolder(root=path_, transform=transform)
    return transformed

def upsample_minority_class(dataset):
    # Separate samples and labels
    images, labels = zip(*dataset.samples)

    # Count samples for each class
    class_counts = np.bincount(labels)

    # Determine the minority class label
    minority_class_label = np.argmin(class_counts)

    # Separate minority and majority class samples
    minority_indices = np.where(np.array(labels) == minority_class_label)[0]
    majority_indices = np.where(np.array(labels) != minority_class_label)[0]

    minority_samples = np.array(images)[minority_indices]
    minority_labels = np.array(labels)[minority_indices]

    majority_samples = np.array(images)[majority_indices]
    majority_labels = np.array(labels)[majority_indices]

    # Upsample minority class to match majority class
    minority_samples_upsampled, minority_labels_upsampled = resample(minority_samples, minority_labels,
                                                                    replace=True, n_samples=len(majority_samples),
                                                                    random_state=42)

    # Concatenate upsampled minority class samples with majority class samples
    images_upsampled = np.concatenate((majority_samples, minority_samples_upsampled))
    labels_upsampled = np.concatenate((majority_labels, minority_labels_upsampled))

    # Shuffle the upsampled data
    indices = np.arange(len(images_upsampled))
    np.random.shuffle(indices)
    images_upsampled = images_upsampled[indices]
    labels_upsampled = labels_upsampled[indices]

    # Update dataset samples
    dataset.samples = list(zip(images_upsampled, labels_upsampled))

    return dataset


def prepare_dataset():
    train_paths = [
        'dataset/train/40x',
        'dataset/train/100x',
        'dataset/train/200x',
        'dataset/train/400x'
    ]
    os.makedirs("Plots/", exist_ok=True)
    trainloaders = []
    valloaders = []
    testloader = []
    # Count the number of benign and malignant images in each train_path
    benign_count = []
    malignant_count = []

    for train_path in tqdm(train_paths, desc="Processing datasets"):
        dataset = totens(train_path)
        class_0, class_1 = check_class_sizes(dataset)
        benign_count.append(class_0)
        malignant_count.append(class_1)
    
    mags = ['40X', '100X', '200X', '400X']
    names = ['Normal_40X', 'Normal_100X', 'Normal_200X', 'Normal_400X']
    for i in tqdm(range(len(benign_count)), desc="Plotting Normal Distributions"):
        plot_class_distribution(benign_count[i], malignant_count[i], "Plots", f"{names[i]}.png", mags[i])

    names = ['Upsampled_40X', 'Upsampled_100X', 'Upsampled_200X', 'Upsampled_400X']
    benign_count.clear()
    malignant_count.clear()
    
    for train_path in tqdm(train_paths, desc="Processing upsampled datasets"):
        dataset = totens(train_path)
        upsampled_trainset = upsample_minority_class(dataset)
        class_0, class_1 = check_class_sizes(upsampled_trainset)
        benign_count.append(class_0)
        malignant_count.append(class_1)
        
    for i in tqdm(range(len(benign_count)), desc="Plotting Upsampled Distributions"):
        plot_class_distribution(benign_count[i], malignant_count[i], "Plots", f"{names[i]}.png", mags[i])
        
prepare_dataset()
        
        


