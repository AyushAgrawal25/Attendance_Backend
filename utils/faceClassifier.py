# Creating a face classifier
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import torch.nn as nn
import torch
from torchvision import transforms
from core.model import ShuffleFaceNet
import numpy as np
import os
# from

def load_shuffleFaceNet():
    model=ShuffleFaceNet()
    ckpt=torch.load('./model/060.ckpt', map_location='cpu')
    model.load_state_dict(ckpt['net_state_dict'])
    model.eval()
    return model

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

# TODO: Learn the Logistic Regression classifier once again.
# TODO: Try creating a threshold for the classifier.
def train():
    softmax=LogisticRegression(
        solver='lbfgs',
        multi_class='multinomial',
        max_iter=100,
        C=0.1,
    )

    clf=GridSearchCV(
        estimator=softmax,
        param_grid={
            'C': [0.1, 1, 10, 100, 1000]
        },
        cv=2,
        verbose=1,
        n_jobs=-1
    )

    model=load_shuffleFaceNet()

    lables, emeddings=extract_data('./static/uploads/', model)
    print(lables)

    # clf.fit(emeddings, lables)

    # Save the model
    torch.save(clf, './model/faceClassifier.ckpt')
    return clf.best_estimator_

def getClassifier():
    clf=torch.load('./model/faceClassifier.ckpt')
    return clf