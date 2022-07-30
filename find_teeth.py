from model import BiSeNet
import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np

net = BiSeNet(n_classes=19)

net.load_state_dict(torch.load("models/face.pth", map_location=torch.device('cpu')))
net.eval()

with torch.no_grad():
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = Image.open("input_imgs/Capture.PNG").convert('RGB')
    image = img.resize((512, 512), Image.BILINEAR)
    # img = cv2.imread("input_imgs/Capture.PNG", 4)
    # image = cv2.resize(img, (512, 512))
    # cv2.imwrite('original.png', image)
    img = to_tensor(image)
    img = torch.unsqueeze(img, 0)
    # img = img.cuda()
    print(img.shape)
    out = net(img)[0]
    parsing = out.squeeze(0).cpu().numpy().argmax(0)
    img = np.array(image)
    img[parsing == 11, :] = 0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('without_teeth.png', img)