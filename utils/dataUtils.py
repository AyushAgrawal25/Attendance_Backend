from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import torch.nn as nn
import torch
from torchvision import transforms
from core.model import ShuffleFaceNet, load_shuffleFaceNet
import numpy as np
import os
import time

def load_image(img_path):
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = img.resize((96, 112), Image.BILINEAR)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
    img = img.unsqueeze(0)
    return img

def extract_embeddings(imgPath, model):
    img=load_image(imgPath)
    output=model(img)
    output=output.detach().numpy()
    return output.flatten()

# path is path to the upload folder.
def extract_data(dir_path, model):
    labels=[]
    embeddings=[]

    for label in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, label)):
            for file in os.listdir(os.path.join(dir_path, label)):
                if file.endswith('.jpg'):
                    embedding=extract_embeddings(os.path.join(dir_path, label, file), model)
                    embeddings.append(embedding)
                    labels.append(label)

    return labels, embeddings