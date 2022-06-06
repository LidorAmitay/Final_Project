# Check out the README file before

import os

# model
import torch
import torch.nn as nn
import torchvision.models as models

# image
import numpy as np
import cv2 as cv
import albumentations as A
from albumentations.pytorch import ToTensorV2


NUM_OF_CLASSES = 7
model = models.resnext50_32x4d(pretrained=True)
dim_in = model.fc.in_features
model.fc = nn.Linear(dim_in, NUM_OF_CLASSES)
optimizer = torch.optim.Adam(model.parameters(), lr = 5e-5)

def save_model(model, path):
  torch.save({
            'model_state_dict': model.state_dict(),
            }, path)

def load_model(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    # return loss

# path to the model
# CHANGE TO YOUR PATH
MODEL_PATH = "D:/Development/Final_Project/model/model"
#"D:/Development/Final_Project/model/best_model_cpu"

load_model(MODEL_PATH, model, optimizer)
# save_model(model, "D:/Development/Final_Project/model/just_model")
# Loading image and predict

emotions = ['disgust', 'happy', 'fear', 'angry', 'sad', 'suprise', 'neutral']

IMG_SIZE = 256
test_transform = A.Compose([
        A.Normalize(
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
])

# CHANGE TO YOUR PATH
# img_path = "D:/Development/Final_Project/images/Lidor_Discust.jpg"
# read img
# img = cv.imread(img_path)
# resize
# img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
# required transformations
# img = test_transform(image = img)['image']

# predicting
# adding new axis, because the model waiting for 4 dim input
# output = model(img[np.newaxis, :, :])
# applying softmax to get the probabilities

# f = os.path.join(directory, filename)
for img_tag in os.listdir('images'):
    img_path = "D:/Development/Final_Project/images/" + img_tag
    # read img
    img = cv.imread(img_path)
    # resize
    img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
    # required transformations
    img = test_transform(image = img)['image']

    # predicting
    # adding new axis, because the model waiting for 4 dim input
    output = model(img[np.newaxis, :, :])
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    emotion = emotions[np.argmax(probabilities.detach().numpy())]
    print(probabilities)
    print(img_tag)
    print(emotion)

