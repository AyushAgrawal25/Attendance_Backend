# Creating a face classifier
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
from utils.dataUtils import extract_data

# TODO: Learn the Logistic Regression classifier once again.
# TODO: Try creating a threshold for the classifier.
def train():
    print('Classification Model Training:')
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

    startTime=time.time()
    print('Extracting Data...')
    lables, emeddings=extract_data('./static/uploads/', model)
    endTime=time.time()
    print('Data Extraction Time: ', endTime-startTime)
    print('{} Labels and {} Embeddings'.format(len(lables), len(emeddings)))

    print('Training Classification Model using Transfer Learning...')
    startTime=time.time()
    clf.fit(emeddings, lables)
    endTime=time.time()
    print('Training Time: {} seconds'.format(endTime-startTime))
    
    # Save the model
    torch.save(clf, './model/faceClassifier.ckpt')
    return clf.best_estimator_

def getClassifier():
    clf=torch.load('./model/faceClassifier.ckpt')
    return clf