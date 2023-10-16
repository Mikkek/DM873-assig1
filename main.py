import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.nn.functional import pad

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

def custom_pad(batch):
    images, labels = zip(*batch)
    
    max_width = max([image.shape[1] for image in images])
    max_height = max([image.shape[2] for image in images])

    padded_images = [pad(img, (max_width - img.size(2), 0, max_height - img.size(1), 0)) for img in images]

    images_stacked = torch.stack(padded_images)
    labels_stacked = torch.tensor(labels)

    return images_stacked, labels_stacked

train_transforms = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

validate_transforms = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

train_dataset = datasets.ImageFolder('./data/catdog_data/train', train_transforms)
test_dataset = datasets.ImageFolder('./data/catdog_data/test')
validate_dataset = datasets.ImageFolder('./data/catdog_data/validation', validate_transforms)

print(f'Define data loaders')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=32)
print(f'done')

class CatDogClassifier(nn.Module):
    def __init__(self):
        super(CatDogClassifier, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(64*3*3, 10)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(10, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

model = CatDogClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda")
model = model.to(device)
print(device)

def trainModuleOnGpu(model, train_loader, validate_loader, criterion, optimizer, num_epoch=10):
    train_losses = []
    validate_losses = []

    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0
        for data, label in train_loader:
            # move data to GPU if CUDA is available
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, label)
            loss.backward()
            # print(model.conv1.weight.grad)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # validate the loss
        model.eval()
        val_loss = 0
        total = 0
        correct = 0

        with torch.no_grad():
            for images, labels in validate_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                _, predicted = torch.max(outputs.detach(), dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(validate_loader.dataset)
        validate_losses.append(val_loss)
        accuracy = 100 * correct / total
        print(f'Epoch: {epoch + 1}/{num_epoch}\n\tTraining Loss: {train_loss:.4f}\n\tValidation Loss: {val_loss:.4}\n\tValidation Accuracy: {accuracy:.2f}%\n')

    return train_losses, validate_losses

train_losses, validate_losses = trainModuleOnGpu(model, train_loader, validate_loader, criterion, optimizer, num_epoch=10)
