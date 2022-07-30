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
        teeth_mask = np.zeros_like(face_mask).astype('uint8')
        teeth_mask[face_mask == 11] = 1
        return teeth_mask

def get_bbox_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.argmax(rows), mask.shape[0] - 1 - np.argmax(np.flipud(rows))
    cmin, cmax = np.argmax(cols), mask.shape[1] - 1 - np.argmax(np.flipud(cols))
    return rmin, rmax, cmin, cmax

def write_mask(output_path, mask):
    cv2.imwrite(output_path, np.expand_dims(mask, -1))

def compare_shape(input_mask, ref_mask):
    ref_mask_resized = cv2.resize(ref_mask, (input_mask.shape[1], input_mask.shape[0]))
    return np.sum(input_mask * ref_mask_resized)

def change_teeth(input_image_path, ref_image_path, output_path):
    input_img = imread(input_image_path)
    input_mask = get_teeth_mask(input_img)
    input_img = np.array(input_img)[:,:,::-1]

    # write_mask("input_mask.png", input_mask)
    input_rmin, input_rmax, input_cmin, input_cmax = get_bbox_from_mask(input_mask)

    input_mask_teeth = input_mask[input_rmin:input_rmax, input_cmin:input_cmax]
    write_mask("input_mask.png", input_mask_teeth)

    ref_img = imread(ref_image_path)
    ref_mask = get_teeth_mask(ref_img)
    # write_mask("ref_mask.png", ref_mask)
    ref_rmin, ref_rmax, ref_cmin, ref_cmax = get_bbox_from_mask(ref_mask)

    ref_mask_teeth = ref_mask[ref_rmin:ref_rmax, ref_cmin:ref_cmax]

    print(compare_shape(input_mask_teeth, ref_mask_teeth))

    ref_img = np.array(ref_img)[:,:,::-1]
    ref_img[ref_mask == 0,:] = 0
    ref_img = ref_img[ref_rmin:ref_rmax, ref_cmin:ref_cmax, :]
    ref_img = cv2.resize(ref_img, (input_cmax - input_cmin, input_rmax - input_rmin))

    input_teeth_part = input_img[input_rmin:input_rmax, input_cmin:input_cmax, :]
    np.putmask(input_teeth_part, ref_img != 0, ref_img)
    # input_teeth_part[(ref_img != 0).all(axis=-1)] = ref_img
    cv2.imwrite(output_path, input_img)

change_teeth('input_imgs/Capture.PNG',  "input_imgs/1.png", "1.png")
change_teeth('input_imgs/Capture.PNG',  "input_imgs/2.png", "2.png")
change_teeth('input_imgs/Capture.PNG',  "input_imgs/3.png", "3.png")
change_teeth('input_imgs/Capture.PNG',  "input_imgs/4.png", "4.png")
change_teeth('input_imgs/Capture.PNG',  "input_imgs/5.png", "5.png")
change_teeth('input_imgs/Capture.PNG',  "input_imgs/6.jpg", "6.png")