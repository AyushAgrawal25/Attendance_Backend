from PIL import Image
import torch.nn as nn
import torch
from torchvision import transforms
from core.model import ShuffleFaceNet

def load_image(img_path):
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = img.resize((96, 112), Image.BILINEAR)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
    img = img.unsqueeze(0)
    return img

def predict(img_path1, img_path2):
    model=ShuffleFaceNet()
    ckpt=torch.load('./model/060.ckpt', map_location='cpu')
    model.load_state_dict(ckpt['net_state_dict'])
    model.eval()

    img1=load_image(img_path1)
    img2=load_image(img_path2)

    output1=model(img1)
    output2=model(img2)

    cos=nn.CosineSimilarity(dim=1, eps=1e-6)
    sim=cos(output1, output2)

    return sim.item()

if __name__ == '__main__':
    output=predict('./static/temps/22e99bcea1db2efd9f25f63c3227a5a6/faces/group.image.01.jpg_face.1.jpg', './static/temps/22e99bcea1db2efd9f25f63c3227a5a6/faces/group.image.01.jpg_face.2.jpg')
    print('Working... {}'.format(output))