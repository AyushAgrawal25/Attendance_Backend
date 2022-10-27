from PIL import Image
import torch.nn as nn
import torch
from torchvision import transforms
from core.model import ShuffleFaceNet
import numpy as np

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

if __name__ == '__main__':
    output=predict('./static/temps/22e99bcea1db2efd9f25f63c3227a5a6/faces/group.image.01.jpg_face.1.jpg', './static/temps/22e99bcea1db2efd9f25f63c3227a5a6/faces/group.image.01.jpg_face.2.jpg')
    print('Working... {}'.format(output))