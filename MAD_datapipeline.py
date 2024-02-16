# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:33:35 2024

@author: harip
"""

# import dependencies
import torch
import random
from PIL import Image
from pathlib import Path
from torch import nn, save, load
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import numpy as np
import json
import requests
import sys
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import os

d = "\n\n------------------------------------------------------------------------------------\n"

# making sure cuda support is set up and working correctly
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(d)
print(f'Using {device} for inference')
print(torch.cuda.get_device_name(0))
print(d)


# load in the input images + organise into training and validation datasets

# Set up directory for dataset and walk through its folders to check
directory = Path("C:/Users/harip/desktop/facetest/sample-dataset/")
''' pipeline '''

def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

walk_through_dir(directory)
print(d)

# Setup train and testing paths
train_dir = directory / "train"
test_dir = directory / "test"
print("\n",train_dir,"\n",test_dir)
print(d)

#plots out a random image with image metadata
# Set seed
random.seed(9) # <- try changing this and see what happens

# 1. Get all image paths (* means "any combination")
image_path_list = list(directory.glob("*/*/*.jpg"))

# 2. Get random image path
random_image_path = random.choice(image_path_list)

# 3. Get image class from path name (the image class is the name of the directory where the image is stored)
image_class = random_image_path.parent.stem

# 4. Open image
img = Image.open(random_image_path)

# 5. Print metadata
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}") 
print(f"Image width: {img.width}")
img

# 6. Turn the image into an array
img_as_array = np.asarray(img)

# 7. Plot the image with matplotlib
plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
plt.axis(False);


# Write transform for image
data_transform = transforms.Compose([
    # Resize the images to 300x300
    transforms.Resize(size=(300, 300)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])

#JUST HERE TO TEST IF THE DATA TRANSFORM IS WORKING CORRECTLY
def plot_transformed_images(image_path, transform, n=3, seed=819):
    #Plots a series of random images from image_paths.

    #Will open n image paths from image_paths, transform them
    #with transform and plot them side by side.

    #Args:
        #image_paths (list): List of target image paths. 
        #transform (PyTorch Transforms): Transforms to apply to images.
        #n (int, optional): Number of images to plot. Defaults to 3.
        #seed (int, optional): Random seed for the random generator. Defaults to 42.
    
    random.seed(seed)
    random_image_paths = random.sample(image_path, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

plot_transformed_images(image_path_list,
                        transform=data_transform,
                        n=3)


# Use ImageFolder function to create dataset(s)
train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root=test_dir, 
                                 transform=data_transform)

print(f"Train data:\n{train_data}\nTest data:\n{test_data}")
print(d)

# little check the dataset classes and length is correct
# Get class names as a list
class_names = train_data.classes
print(class_names)
# Check the lengths
print("traindata:",(len(train_data)), "\ntestdata:",(len(test_data)))

# index our train and test datasets to find samples and their target labels
img, label = train_data[0][0], train_data[0][1]
print(f"Image tensor:\n{img}")
print(f"Image shape: {img.shape}")
print(f"Image datatype: {img.dtype}")
print(f"Image label: {label}")
print(f"Label datatype: {type(label)}")


# Turn train and test Datasets into DataLoaders
BATCH_SIZE = 1  # how many samples per batch?
NUM_WORKERS = 1  # how many subprocesses to use for data loading? (higher = more)
print(f"\nCreating DataLoader's with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")
print(d)


train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True) # shuffle the data?

test_dataloader = DataLoader(dataset=test_data, 
                             batch_size=BATCH_SIZE, 
                             num_workers=NUM_WORKERS, 
                             shuffle=False) # don't usually need to shuffle testing data

print("\n",train_dataloader,"\n",test_dataloader)
img, label = next(iter(train_dataloader))
print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")
print(d)

#  shape: 3, 300, 300 | classes: "morph" or "not morph"

# create / configure the image classifier neural network model

class MorphDetector(nn.Module):
    def __init__(self):
        super().__init__() #instantiate our nn.Module
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, (30,30), 1), # input layer
            nn.ReLU(),
            nn.Conv2d(32, 64, (30,30), 1), # first hidden layer
            nn.ReLU(),
            nn.Conv2d(64, 64, (30,30), 1), # second hidden layer
            nn.ReLU(),
            nn.Flatten(),
            ### THE SHAPE COMING OUT OF FLATTEN DOES NOT ALIGN WITH THE SIZE OF SHAPE EXPECTED IN LINEAR
            # NVM ITS SOLVED :)
            nn.Linear(64*(273-60)*(273-60), 1)  # output layer "morph" or "not morph"
        )

    #function to move data between layers
    def forward(self, x):
        return self.model(x)

# instantiate our neural network, optimizer, and loss function
md = MorphDetector().to(device)
opt = Adam(md.parameters(), lr=1e-3)  # adam gets trained very quickly do like 10 epochs not 10000
loss_fn = nn.MSELoss()

# training loop 
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    print("this epoch has started (#1b)")
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_dataloader):
        print("this is the part that does not work (#1b)")
        # Every data instance is an input + label pair
        inputs, labels = data
        labels = (labels.to(device).float())  # fixed some cuda stuff
        print(1)
        # Zero your gradients for every batch!
        opt.zero_grad()
        print(2)
        # Make predictions for this batch
        output = md(inputs.to(device).float())
        outputs = torch.squeeze(output)
        print(3)
        print(f"labels: {labels}")
        print(f"inputs: {inputs.shape} -- outputs: {outputs.shape} -- labels: {labels.shape}")
        # issue is demonstrated in the above print statements
        # input (labels) size is 1; output size is 1,1 array
        # the output of the network is multidimensional
        # whereas the input (labels) are single dimensional
        # need to reshape output so that the loss can be calculated properly
        # maybe clamp values between 0 and 1 might work might not
        # depends on output layer
        # might be worth adding a softmax layer to normalise between layers
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        print(4)
        # Adjust learning weights
        opt.step()
        print(5)
        # Gather data and report
        print(6)
        # training loop crashes here or if i try to print loss after 1 loop
        # this suggests there is something wrong with the loss value
        # or how its being calculated, since this
        print(type(loss))
        running_loss += loss.to(torch.device('cpu')).item()
        print(7)
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
        print(8)
    return last_loss

# PyTorch TensorBoard support
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
# Initializing in a separate cell, so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5 # runs the training for 5 epochs

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    md.train(True)
    print("this part works (#1a)")
    avg_loss = train_one_epoch(epoch_number, writer)
    print("this part doesn't even get to run (#2b)")

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    print(md.eval())

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(test_dataloader):
            vinputs, vlabels = vdata
            voutputs = md(vinputs.to(device))
            vloss = loss_fn(voutputs, vlabels.to(device))
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track the best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(md.state_dict(), model_path)

    epoch_number += 1

'''old epoch thingy
if __name__ == "__main__":
    for epoch in range(3): # train for 3 epochs
        for batch in dataset:
            X,y = batch  # unpack the data
            X, y = X.to('cuda'), y.to('cuda') # send x,y values to GPU
            yhat = md(X) # generate a prediction
            loss = loss_fn(yhat, y) # calculates our loss
            
            #Apply backprop
            opt.zero_grad() # zero out any existing gradients
            loss.backward() # calculate gradients
            opt.step() # apply gradient descent
        
        # print out loss values at every epoch
        print(f"Epoch:{epoch} loss is {loss.item()}") 
'''

'''save our model to our environment
    with open('model_state.pt', 'wb') as f:
        save(md.state_dict(), f)
        
    # load our model
    with open('model_state.pt', 'rb') as f:     
        md.load_state_dict(load(f))
'''
# test the model on the validation dataset ????where is this


'''proof of concept test
img = Image.open('face1.jpg') #import in test image
img_tensor = ToTensor()(img).unsqueeze(0).to('cuda') #convert that to a tensor
print(torch.argmax(md(img_tensor))) # print out final output tensor
'''

# create data visualisations to evaluate the performance (cherry on top step)


#%%

#%%
