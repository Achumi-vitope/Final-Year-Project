import torch
import torch.nn as nn

import os
from sklearn.metrics import precision_score, confusion_matrix, f1_score

import pickle

from tqdm import tqdm
from torchvision.models import vit_b_16, ViT_B_16_Weights

from warnings import  filterwarnings
import torch.nn.functional as F

# Filter out warnings
filterwarnings("ignore", category=UserWarning)

def Net(num_classes):
    pretrained_vit_weights = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    pretrained_vit = vit_b_16(weights=pretrained_vit_weights)

    for params in pretrained_vit.parameters():
        params.requires_grad = False

    # Specify the layers to unfreeze
    layers_to_unfreeze = [
        "encoder.layers.encoder_layer_11.ln_1",
        "encoder.layers.encoder_layer_11.self_attention.out_proj",
        "encoder.layers.encoder_layer_11.mlp.3",  # Unfreezing the last linear layer in the MLP block of layer 11
        "heads.head"
    ]

    # Unfreeze the specified layers
    for name, param in pretrained_vit.named_parameters():
        if any(layer_name in name for layer_name in layers_to_unfreeze):
            param.requires_grad = True

    # Replace the head with a new one appropriate for the number of classes
    pretrained_vit.heads = nn.Sequential(nn.Linear(
                                    in_features=768, 
                                    out_features=num_classes))


    return pretrained_vit



def train(net, trainloader, optimizer, epochs, device: str):

    criterion = nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    for epoch in tqdm(range(epochs)):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
        

def test(net, testloader, device: str, text: str):
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels.long()).item()
            predicted = outputs.data.max(1, keepdim=True)[1]
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (predicted == labels.view_as(predicted)).sum().item()
    accuracy = correct / len(testloader.dataset)
    precision = precision_score(all_labels, all_predictions, average='binary')  # Changed to 'binary'
    cm = confusion_matrix(all_labels, all_predictions)
    os.makedirs("confusion_mat", exist_ok=True)
    with open(f"confusion_mat/{text}confusion_matrix.pkl", "wb") as f:
        pickle.dump(cm, f)
    f1 = f1_score(all_labels, all_predictions, average='binary')  # Changed to 'binary'
    return loss, {"accuracy": accuracy, "precision": precision, "f1_score": f1}
