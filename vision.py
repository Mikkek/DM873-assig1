import os
import copy
import numpy as np
from PIL import Image, ImageFilter

import torch
from torch.optim import SGD
from torch.autograd import Variable
from torchvision import models

use_cuda = torch.cuda.is_available()

class RegClassSpecificImgGen():

    def __init__(self, model, target_class):
        self.mean = [-0.485, -0.456, -0.406]
        self.std = [1/0.229, 1/0.224, 1/0.225]
        self.model = model.cuda() if use_cuda else model
        self.model.eval()
        self.target_class = target_class

        self.created_img = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))

        if not os.path.exists(f'./generated/class_{self.target_class}'):
            os.makedirs(f'./generated/class_{self.target_class}')

    def generate(self, iterations=150, blur_freq=4, blur_rad=1, wd=0.0001, clipping_value=0.1):
        init_learning_rate = 6
        for i in range(1, iterations):
            if i % blur_freq == 0:
                self.processed_img = preprocess_and_blur_img(
                    self.created_img, blur_rad)
                
            else:
                self.processed_img = preprocess_and_blur_img(
                    self.created_img)
                
            if use_cuda:
                self.processed_img = self.processed_img.cuda()

            optimizer = SGD([self.processed_img],
                            lr=init_learning_rate, weight_decay=wd)
            
            output = self.model(self.processed_img)
            class_loss = -output[0, self.target_class]

            if i in np.linspace(0, iterations, 10, dtype=int):
                print('Iteration', str(i), 'Loss',
                      "{0:.2f}".format(class_loss.data.cpu().numpy()))
                
            self.model.zero_grad()
            class_loss.backward()

            if clipping_value:
                torch.nn.utils.clip_grad_norm(
                    self.model.parameters(), clipping_value)
                
            optimizer.step()

            self.created_img = recreate_img(self.processed_img.cpu())

            if i in np.linspace(0, iterations, 10, dtype=int):
                # Save image
                img_path = f'./generated/class_{self.target_class}/c_{self.target_class}_iter_{i}_loss_{class_loss.data.cpu().numpy()}.jpg'
                save_img(self.created_img, img_path)

        #save final image
        im_path = f'./generated/class_{self.target_class}/c_{self.target_class}_iter_{i}_loss_{class_loss.data.cpu().numpy()}.jpg'
        save_img(self.created_img, im_path)

        return self.processed_img
        
def preprocess_and_blur_img(pil_img, blur_rad=None):
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if type(pil_img) != Image.Image:
        try:
            pil_img = Image.fromarray(pil_img)
        except Exception as e:
            print(
                "Could not transform PIL_img to a PIL image object. Check input")
            
    if blur_rad:
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(blur_rad))

    img_as_arr = np.float32(pil_img)
    img_as_arr = img_as_arr.transpose(2, 0, 1) # Convert arr to D,W,H

    # Normalize channels
    for channel, _ in enumerate(img_as_arr):
        img_as_arr[channel] /= 255
        img_as_arr[channel] -= mean[channel]
        img_as_arr[channel] /= std[channel]

    # Convert to float tensor
    img_as_tensor = torch.from_numpy(img_as_arr).float()
    img_as_tensor = img_as_tensor.unsqueeze(0) # Add one more channel. Tensor shape = 1, 3, 224, 224

    # Convert to pytorch variable
    if use_cuda:
        img_as_var = Variable(img_as_tensor.cuda(), requires_grad=True)
    else:
        img_as_var = Variable(img_as_tensor, requires_grad=True)

    return img_as_var

def recreate_img(img_as_var):

    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]

    recreated_img = copy.copy(img_as_var.data.numpy()[0])
    for c in range(3):
        recreated_img[c] /= reverse_std[c]
        recreated_img[c] -= reverse_mean[c]

    recreated_img[recreated_img > 1] = 1
    recreated_img[recreated_img < 0] = 0
    recreated_img = np.round(recreated_img * 255)

    recreated_img = np.uint8(recreated_img).transpose(1, 2, 0)
    return recreated_img

def save_img(img, path):
    if isinstance(img, (np.ndarray, np.generic)):
        img = format_np_output(img)
        img = Image.fromarray(img)

    img.save(path)

def format_np_output(np_arr):
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr
