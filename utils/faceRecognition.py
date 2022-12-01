from PIL import Image
import torch.nn as nn
import torch
from torchvision import transforms
from core.model import ShuffleFaceNet, load_shuffleFaceNet
import numpy as np
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import time
from utils.dataUtils import extract_data
from sklearn.model_selection import train_test_split
    
def load_image(img_path):
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = img.resize((96, 112), Image.BILINEAR)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
    img = img.unsqueeze(0)
    return img

def getThreshold(score, threshold):
    return np.mean(score) + threshold * np.std(score)

def predict(img_path1, img_path2):
    model=ShuffleFaceNet()
    ckpt=torch.load('./model/060.ckpt', map_location='cpu')
    model.load_state_dict(ckpt['net_state_dict'])
    model.eval()

    img1=load_image(img_path1)
    img2=load_image(img_path2)

    output1=model(img1)
    output2=model(img2)

    output1=output1.detach().numpy()
    output2=output2.detach().numpy()
    
    # # Similarity using mean
    # score=np.sum(np.square(output1-output2), axis=1)
    # return score[0]
    
    # Similarity using cosine distance
    score=np.sum(output1*output2, axis=1)/(np.linalg.norm(output1)*np.linalg.norm(output2))
    return score[0]

def face_recognition_model():
    model=Sequential(
        name='face_recognition_model'
    )

    # Create MobileNet Bottleneck Layer
    # Expansion layer with 1x1 convolution and 32 filters for input of shape (1, 128, 1)
    model.add(Conv1D(32, 1, activation='relu', input_shape=(128, 1)))

    # Depthwise convolution with 3x3 kernel and 32 filters
    model.add(Conv1D(32, 3, activation='relu', padding='same'))

    # Pointwise convolution with 1x1 kernel and 16 filters
    model.add(Conv1D(32, 1, activation='relu'))

    # Softmax layer
    model.add(Flatten())
    model.add(Dense(units=4, activation='softmax')) 

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train():
    # Loading ShuffleFaceNet model
    shuffle_facenet=load_shuffleFaceNet()
    
    # Extracting embeddings from the dataset
    startTime=time.time()
    print('Extracting Data...')
    labels, embeddings=extract_data('./static/uploads/', shuffle_facenet)
    endTime=time.time()
    print('Data Extraction Time: ', endTime-startTime)
    print('{} Labels and {} Embeddings'.format(len(labels), len(embeddings)))
    
    # Convert the labels into numbers
    label_to_num=[]
    for label in labels:
        label_to_num.append(int(label))
    labels=np.array(label_to_num)

    # Splitting the dataset into training and testing set
    X_train, X_test, y_train, y_test=train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    print(len(X_train), len(X_test), len(y_train), len(y_test))
    # Training the model
    print('Training the Face Recognition model...')
    model=face_recognition_model()
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    model.save('./model/face_recognition_model.h5')

if __name__ == '__main__':
    output=predict('./static/temps/22e99bcea1db2efd9f25f63c3227a5a6/faces/group.image.01.jpg_face.1.jpg', './static/temps/22e99bcea1db2efd9f25f63c3227a5a6/faces/group.image.01.jpg_face.2.jpg')
    print('Working... {}'.format(output))