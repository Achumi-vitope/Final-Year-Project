import torch
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ViT_B_16_Weights
from sklearn.utils import resample
import numpy as np
import torch.nn.functional as F

def check_class_sizes(trainloader):
    # Initialize counters for each class label
    count_0 = 0
    count_1 = 0
    for _, batch_labels in trainloader:
        count_0 += (batch_labels == 0).sum().item()
        count_1 += (batch_labels == 1).sum().item()
    
    print(f"Class 0 size: {count_0}")
    print(f"Class 1 size: {count_1}")
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


def prepare_dataset(batch_size: int, val_ratio: float = 0.1):
    train_paths = [
        'dataset/train/40x',
        'dataset/train/100x',
        'dataset/train/200x',
        'dataset/train/400x'
    ]

    test_paths = [
        'test_set/40x',
        'test_set/100x',
        'test_set/200x',
        'test_set/400x'
    ]
    trainloaders = []
    valloaders = []
    testloader = []

    for train_path in train_paths:
        # print("Train_Path Lengths per mag:",len(train_path))
        dataset = totens(train_path)
        upsampled_trainset = upsample_minority_class(dataset) 
        num_samples = len(upsampled_trainset)
        num_val = int(val_ratio * num_samples)
        num_train = num_samples - num_val

        train_subset, val_subset = random_split(dataset, [num_train, num_val])

        trainloaders.append(
            DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        )
        valloaders.append(
            DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
        )

    for test_path in test_paths:
        # print("Test_Path Lengths per mag:",len(test_path))
        testset = totens(test_path)
        testloader.append(DataLoader(testset, batch_size=30, shuffle=False))
        
    print(f"Total length of trainloader: {len(trainloaders)}")
    print(f"Total length of testloader: {len(testloader)}")

    return trainloaders, valloaders, testloader


