# %%
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
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json
import requests
import sys
import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

import warnings

warnings.filterwarnings('ignore')

import os

d = "\n\n------------------------------------------------------------------------------------\n"

# making sure cuda support is set up and working correctly
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(d)
print(f'Using {device} for inference')
print(torch.cuda.get_device_name(0), d)

####################################################
#             Load in the input images +           #
#  Organise into training and validation datasets  #
####################################################

# Set up directory for dataset and walk through its folders to check
# directory = Path("C:/Users/harip/desktop/facetest/dataset/sample-dataset/")
# directory = Path("C:/Users/harip/desktop/facetest/dataset/finalised-dataset/")
directory = Path("C:/Users/harip/desktop/facetest/dataset/balanced-dataset/")


def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


walk_through_dir(directory)
print(d)

# Setup train and testing paths
train_dir = directory / "train"
test_dir = directory / "test"
print(train_dir, "\n", test_dir, d)

# 1. Get all image paths (* means "any combination")
# image_path_list = list(directory.glob("*/*/*.jpg"))
image_path_list = list(directory.glob("*/*/*.png"))

# 2. Get random image path
random_image_path = random.choice(image_path_list)

# 3. Get image class from path name (the image class is the name of the directory where the image is stored)
image_class = random_image_path.parent.stem

# 4. Open image
img = Image.open(random_image_path)

# 5. Turn the image into an array
img_as_array = np.asarray(img)

# 6. Write transform for image
data_transform = transforms.Compose([
    # Resize the images to 300x300
    # transforms.Resize(size=(300, 300)),
    transforms.Resize(size=(512, 512)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5),  # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor()  # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
])

# 7. Use ImageFolder function to create dataset(s)
train_data = datasets.ImageFolder(root=train_dir,  # target folder of images
                                  transform=data_transform,  # transforms to perform on data (images)
                                  target_transform=None)  # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

print(f"Train data:\n{train_data}\nTest data:\n{test_data}", d)

# little check the dataset classes and length is correct
# Get class names as a list
class_names = train_data.classes
print(class_names)
# Check the lengths
print("traindata:", (len(train_data)), "\ntestdata:", (len(test_data)), d)

# 8. index our train and test datasets to find samples and their target labels
img, label = train_data[0][0], train_data[0][1]

# 9. Turn train and test Datasets into DataLoaders
#BATCH_SIZE = 32  # how many samples per batch?
BATCH_SIZE = 4
NUM_WORKERS = 6  # how many subprocesses to use for data loading? (higher = more)
print(f"Creating DataLoader's with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.", d)

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True)  # shuffle the data?

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS,
                             shuffle=False)  # don't usually need to shuffle testing data


# shape: 3, 300, 300 | classes: "morph" or "not morph"

# create / configure the image classifier neural network model
'''
class MorphDetector(nn.Module):
    def __init__(self):
        super().__init__()  # instantiate our nn.Module
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, (30, 30), 1),  # input layer
            nn.ReLU(),
            nn.Conv2d(32, 64, (30, 30), 1),  # first hidden layer
            nn.ReLU(),
            nn.Conv2d(64, 64, (30, 30), 1),  # second hidden layer
            nn.ReLU(),
            nn.Flatten(),  # Shape coming out from flatten, aligns with shape expected in Linear Layer
            # parameters of linear layer is (in_neurons * (image height or width * 2), out_neurons)
            nn.Linear(64 * (213 ** 2), 1),  # output layer "morph" or "not morph"
            nn.Sigmoid()  # Sigmoid activation function to get an output between 0 and 1
        )

    # function to move data between layers
    def forward(self, x):
        return self.model(x)
'''


class MorphDetector(nn.Module):
    def __init__(self):
        super().__init__()  # instantiate our nn.Module
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, (32, 32), 1),  # input layer
            nn.ReLU(),
            nn.Conv2d(32, 64, (32, 32), 1),  # first hidden layer
            nn.ReLU(),
            nn.Conv2d(64, 64, (32, 32), 1),  # second hidden layer
            nn.ReLU(),
            nn.Flatten(),  # Shape coming out from flatten, aligns with shape expected in Linear Layer
            # parameters of linear layer is (in_neurons * (image height or width * 2), out_neurons)
            nn.Linear(64 * (419**2), 1),  # output layer "morph" or "not morph"
            nn.Sigmoid()  # Sigmoid activation function to get an output between 0 and 1
        )

    # function to move data between layers
    def forward(self, x):
        return self.model(x)


# instantiate our neural network, optimizer, and loss function
md = MorphDetector().to(device)
opt = optim.SGD(md.parameters(), lr=0.001)  # adam gets trained too quickly so we use SGD
loss_fn = nn.BCELoss()

# print out the shape of our neural network
print(md.eval(), d)

from tqdm import tqdm

# this stores the data for predicted and actual labels at every epoch
# used to calculate confusion matrices
x_pred, x_true, y_pred, y_true = [], [], [], []

# this stores the TP, FP, FN, TN data from the confusion matrix
# used later to calculate APCER, BPCER and other evaluation metrics
TP, FP, FN, TN = [], [], [], []
APCER, BPCER, ACER = [], [], []


# visual confusion matrix function to reduce code repetition
def conf_matrix(pred, true, num, suffix):
    # creates a confusion matrix using skikit learn
    cm = confusion_matrix(true, pred)

    # creates a pandas dataframe of the confusion matrix
    df_cm = pd.DataFrame(cm, index=[i for i in class_names], columns=[i for i in class_names])

    # creates a plot for visualisations + uses seaborn to create heatmap on plot
    plt.figure(figsize=(12, 7))
    cm_plot = plt.subplot()
    sn.heatmap(df_cm, annot=True, cmap='Greens', fmt='d', ax=cm_plot)

    # set label axis
    cm_plot.set_xlabel('Predicted labels')
    cm_plot.set_ylabel('True labels')

    # creates name for visualisation and assigns it as plot title
    cm_title = ('Confusion Matrix - ' + suffix + ' ' + str(num))
    cm_plot.set_title(cm_title)

    # display the heatmap matrix in console and save it as an image in our project files
    plt.show()
    plt.savefig(cm_title+'.png')

    # Get the current TP, FP, FN, TN values from the confusion matrix
    # Then add them all into the relevant global list objects,
    TP_temp = (df_cm.iat[0, 0]); TP.append(TP_temp)  # morphs detected as such
    FP_temp = (df_cm.iat[0, 1]); FP.append(FP_temp)  # bonafide detected as morphs
    FN_temp = (df_cm.iat[1, 0]); FN.append(FN_temp)  # morphs detected as bonafide
    TN_temp = (df_cm.iat[1, 1]); TN.append(TN_temp)  # bonafide detected as such


# training loop
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of iter(training_loader)
    # so that we can track the batch index and do some intra-epoch reporting
    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

        # Every data instance is an input + label pair
        inputs, labels = data
        labels = (labels.to(device).float())  # fixed some cuda stuff

        # Zero gradients for every batch
        opt.zero_grad()

        # Make predictions for this batch (forward pass)
        output = md(inputs.to(device).float())
        outputs = output[..., 0]  # flatten outputs
        loss = loss_fn(outputs, labels)

        # (backwards pass)
        loss.backward()

        # Adjust learning weights (optimise)
        opt.step()

        # Gather data and report
        running_loss += loss.to(torch.device('cpu')).item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
        '''
        # stores the predicted and true class values in an array for the epoch
        outputs = outputs.data.cpu().numpy()
        x_pred.extend((outputs > 0.5).astype("int32"))  # sorts the outputs into binary values
        x_true.extend(labels.data.cpu().numpy())
        print("batch ", i, ":", outputs) # used to track real-time prediction values in console during training

    # call the confusion matrix creation function
    # then clear the variables before next epoch
    conf_matrix(x_pred, x_true)
    del x_pred[:]; del x_true[:]
    '''
    return last_loss


# PyTorch TensorBoard support
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Initializing in a separate cell, so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/morph_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5  # runs the training for 5 epochs

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    md.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout
    # and using population statistics for batch normalization.

    # Disable gradient computation and reduce memory consumption.
    # Test the model on the validation dataset
    with torch.no_grad():
        for i, vdata in enumerate(test_dataloader):
            vinputs, vlabels = vdata
            vlabels = (vlabels.to(device).float())
            voutput = md(vinputs.to(device).float())
            voutputs = voutput[..., 0]  # flatten voutputs
            vloss = loss_fn(voutputs, vlabels.to(device))
            running_vloss += vloss

            # stores the predicted and true class values in an array for the epoch
            voutputs = voutputs.data.cpu().numpy()
            y_pred.extend((voutputs >= 0.5).astype("int32"))  # sorts the outputs into binary values
            y_true.extend(vlabels.data.cpu().numpy())
            # print("batch ", i, ":", voutputs) # used to track real-time prediction values in console during validation

        # call the confusion matrix creation function
        conf_matrix(y_pred, y_true, epoch_number, "Validation")
        # clear the list variables for next epoch
        del y_pred[:]; del y_true[:]

    # print out the avg loss for the training and test dataset in this epoch
    avg_vloss = running_vloss / (i + 1)
    print('LOSS : train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': avg_loss, 'Validation': avg_vloss},
                       epoch_number + 1)
    writer.flush()

    # Track the best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(md.state_dict(), model_path)

    epoch_number += 1

# Lets us know our model is done training
print(d, "\nThe model has finished training!\n")
print("The model state was saved at the lowest loss rate:", best_vloss.data.cpu().numpy(), d)

# %%
# here to check we have the correct values from our conf matrix
print("Here are the conf matrix values:\n", TP, FP, FN, TN, d)


# function to calculate PAD evaluation metrics
def eval_metrics():
    for i in range(len(TP)):
        APCER.append(FP[i] / (TN[i] + FP[i]))
        BPCER.append(FN[i] / (TP[i] + FN[i]))
        # patch to replace values are being divided by 0, with 0 instead of null
        # to make sure metrics still show up on the graph if value is null
        if (TN[i] + FP[i]) == 0: APCER[i] = 0
        if (TP[i] + FN[i]) == 0: BPCER[i] = 0
        # calculate ACER by averaging APCER and BPCER
        ACER.append((APCER[i] + BPCER[i]) / 2)


# create data visualisations to evaluate the model performance / show effect of model training
def eval_metrics_graph():

    # plotting evaluation metrics over time (epochs)
    plt.figure(figsize=(12, 7))
    eval_plot = plt.subplot()

    # Assigning the first subplot to graph training loss and validation loss
    eval_plot.plot(APCER, color='b', label='APCER')
    eval_plot.plot(BPCER, color='r', label='BPCER')
    eval_plot.plot(ACER, color='g', label='ACER')
    eval_plot.legend(loc="upper right")

    # set label axis
    eval_plot.set_xlabel('Epochs', fontsize=15)
    eval_plot.set_ylabel('Classification Error Rate', fontsize=15)

    # giving the graph a title
    eval_plot.set_title('Visualisation of model error rate, over time',
                    fontdict={'fontsize': 24}, pad=12)

    # display the heatmap matrix in console and save it as an image in our project files
    plt.show()
    # eval_plot.savefig("model_performance.png")


# %%

# call functions to calculate PAD evaluation metrics & create the relevant graphs
eval_metrics(); eval_metrics_graph()
# print out values (temporary, just for debugging)
print(d, APCER, d, BPCER, d, ACER, d)
# clear the lists, so they aren't storing previous/outdated values
del APCER[:]; del BPCER[:]; del ACER[:]


# %%
'''save our model to our environment
    with open('model_state.pt', 'wb') as f:
        save(md.state_dict(), f)
        
    # load our model
    with open('model_state.pt', 'rb') as f:     
        md.load_state_dict(load(f))
'''
