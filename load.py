import torch
from vision import RegClassSpecificImgGen
from train import CatDogClassifier
from torchvision import models

if __name__ == '__main__':
    model = torch.load('catdog_classifier.pth')
    RegClassSpecificImgGen(model, 0).generate()
    RegClassSpecificImgGen(model, 1).generate()
