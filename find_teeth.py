from model import BiSeNet
import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np

net = BiSeNet(n_classes=19)

net.load_state_dict(torch.load("models/face.pth", map_location=torch.device('cpu')))
net.eval()

def imread(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((512, 512), Image.BILINEAR)
    return img

def get_teeth_mask(img):
    with torch.no_grad():
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        img = to_tensor(img)
        img = torch.unsqueeze(img, 0)
        out = net(img)[0]
        face_mask = out.squeeze(0).cpu().numpy().argmax(0)
        teeth_mask = np.zeros_like(face_mask)
        teeth_mask[face_mask == 11] = 255
        return teeth_mask

