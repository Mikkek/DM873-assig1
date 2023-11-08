import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.nn.functional import pad
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from scipy.optimize import curve_fit

class CatDogClassifier(nn.Module):
    def __init__(self):
        super(CatDogClassifier, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128*3*3, out_features=2)
        )
        
    def forward(self, x): 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer3(x)
        x = self.layer3(x)
        x = self.layer3(x)
        x = self.classifier(x)
        return x
    
def train_step(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss, correct, total = 0, 0, 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)  # move data to GPU
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)  # compute loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)  # accumulate the loss
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    avg_loss = train_loss / len(train_loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

def validate_step(model, validate_loader, criterion, device):
    model.eval()
    validate_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for data, target in validate_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            validate_loss += loss.item() * data.size(0)  # accumulate the loss
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = validate_loss / len(validate_loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

def train_and_validate(model, train_loader, validate_loader, optimizer, criterion, device, epochs):
    results = {
        "train_loss": [],
        "train_acc": [],
        "validate_loss": [],
        "validate_acc": []
    }

    scheduler = ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, train_loader, criterion, optimizer, device)
        validate_loss, validate_acc = validate_step(model, validate_loader, criterion, device)

        scheduler.step(validate_loss)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["validate_loss"].append(validate_loss)
        results["validate_acc"].append(validate_acc)

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Validate Loss: {validate_loss:.4f}, Validate Acc: {validate_acc:.4f}')

    return results

def logaritmic(x, a, b, c, d):
    return a * np.log(b * x + c) + d

def plot_results(results):
    epochs = range(1, len(results["train_loss"])+1)
    plt.figure(figsize=(15, 5))


    popt, _ = curve_fit(logaritmic, epochs, results["validate_acc"], maxfev=10000)
    accuracy_approximation = logaritmic(epochs, *popt)

    # plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results["train_loss"], label="Training Loss")
    plt.plot(epochs, results["validate_loss"], label="Validation Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results["validate_acc"], label="Validation Accuracy")
    plt.plot(epochs, accuracy_approximation, label="Validation Accuracy Approximation", linestyle='--')
    plt.title("Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()




if __name__ == '__main__':

    train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        ])

    validate_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])

    train_dataset = datasets.ImageFolder('./data/catdog_data/train', train_transforms)
    test_dataset = datasets.ImageFolder('./data/catdog_data/test', validate_transforms)
    validate_dataset = datasets.ImageFolder('./data/catdog_data/validation', validate_transforms)

    print(f'Define data loaders')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=64)
    print(f'done')

    model = CatDogClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda")
    model = model.to(device)
    print(device)

    results = train_and_validate(model, train_loader, validate_loader, optimizer, criterion, device, epochs=25)
    torch.save(model, './catdog_classifier.pth')

    plot_results(results)
