from model import BiSeNet
import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

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
    cv2.imwrite(output_path, np.expand_dims(mask, -1) * 250)

def compare_shape(input_mask, ref_mask):
    input_mask_resized = cv2.resize(ref_mask, (250, 50))
    return np.sum(input_mask_resized * ref_mask)

def save_ref_imgs(dir):
    masks = []
    imgs = []
    for image_path in os.listdir(dir):
        ref_img = imread(os.path.join(dir, image_path))
        ref_mask = get_teeth_mask(ref_img)
        ref_rmin, ref_rmax, ref_cmin, ref_cmax = get_bbox_from_mask(ref_mask)

        ref_mask_teeth = ref_mask[ref_rmin:ref_rmax, ref_cmin:ref_cmax]
        ref_mask_resized = cv2.resize(ref_mask_teeth, (250, 50))
        masks.append(ref_mask_resized)

        ref_img = np.array(ref_img)[:,:,::-1]
        # ref_img[ref_mask == 0,:] = 0
        ref_img = ref_img[ref_rmin:ref_rmax, ref_cmin:ref_cmax, :]
        ref_img = cv2.resize(ref_img, (250, 50))
        imgs.append(ref_img)
    
    masks = np.array(masks)
    imgs = np.array(imgs)
    np.save('masks.npy', masks)
    np.save('imgs.npy', imgs)

def extract_teeth_and_save(input_path, output_path):
    input_img = imread(input_path)
    input_mask = get_teeth_mask(input_img)
    input_img = np.array(input_img)
    input_img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    input_img_gray[input_mask == 0] = 0
    input_rmin, input_rmax, input_cmin, input_cmax = get_bbox_from_mask(input_mask)
    teeth_part = input_img[input_rmin:input_rmax, input_cmin:input_cmax]
    cv2.imwrite(output_path, teeth_part[:,:,::-1])

def extract_teeth_parts_in_dir(input_dir, output_dir):
    for image_path in os.listdir(input_dir):
        input_path = os.path.join(input_dir, image_path)
        output_path = os.path.join(output_dir, image_path)
        extract_teeth_and_save(input_path, output_path)

def change_teeth(input_image_path, output_path):
    input_img = imread(input_image_path)
    input_mask = get_teeth_mask(input_img)
    input_img = np.array(input_img)
    input_img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    input_img_gray[input_mask == 0] = 0

    input_rmin, input_rmax, input_cmin, input_cmax = get_bbox_from_mask(input_mask)
    input_img_gray_teeth = input_img_gray[input_rmin:input_rmax, input_cmin:input_cmax]
    # input_img_gray_teeth = cv2.equalizeHist(input_img_gray_teeth)
    edged = cv2.Canny(input_img_gray_teeth, 30, 200)
    fig = plt.figure(figsize=(1, 3))
    fig.add_subplot(1, 3, 1)
    plt.imshow(input_img)
    fig.add_subplot(1, 3, 2)
    plt.imshow(input_img_gray_teeth, cmap='gray')
    fig.add_subplot(1, 3, 3)
    plt.imshow(edged)
    plt.show()



    # # ref_img = imread(ref_image_path)
    # # ref_mask = get_teeth_mask(ref_img)
    # # # write_mask("ref_mask.png", ref_mask)
    # # ref_rmin, ref_rmax, ref_cmin, ref_cmax = get_bbox_from_mask(ref_mask)

    # # ref_mask_teeth = ref_mask[ref_rmin:ref_rmax, ref_cmin:ref_cmax]

    # ref_masks = np.load("masks.npy")
    # ref_imgs = np.load("imgs.npy")

    # max_sim = 0
    # max_idx = -1

    # for i in range(ref_masks.shape[0]):
    #     ref_mask = ref_masks[i]
    #     sim = compare_shape(input_mask_teeth, ref_mask)
    #     if sim > max_sim:
    #         max_sim = sim
    #         max_idx = i

    # # cv2.imwrite("style.png", ref_imgs[max_idx])
    # # ref_img = np.array(ref_img)[:,:,::-1]
    # # ref_img[ref_mask == 0,:] = 0
    # # ref_img = ref_img[ref_rmin:ref_rmax, ref_cmin:ref_cmax, :]
    # # ref_img = cv2.resize(ref_img, (input_cmax - input_cmin, input_rmax - input_rmin))
    # input_teeth_part = input_img[input_rmin:input_rmax, input_cmin:input_cmax, :]
    # # cv2.imwrite("content.png", cv2.resize(input_teeth_part, (250, 50)))

    # ref_img = ref_imgs[max_idx]
    # ref_img = cv2.resize(ref_img, (input_cmax - input_cmin, input_rmax - input_rmin))
    # # np.putmask(input_teeth_part, ref_img != 0, ref_img)
    # print(input_teeth_part.shape, input_mask_teeth.shape, ref_img.shape)
    # input_teeth_part[input_mask_teeth != 0,:] = ref_img[input_mask_teeth != 0,:]
    # # input_teeth_part[(ref_img != 0).all(axis=-1)] = ref_img

    # # new_teeth = style_transfer("content.png", "style.png")
    # # new_teeth = cv2.resize(new_teeth, (input_cmax - input_cmin, input_rmax - input_rmin))
    # # ref_mask = cv2.resize(ref_mask, (input_cmax - input_cmin, input_rmax - input_rmin))
    # # new_teeth[ref_mask == 0, :] = 0
    # # np.putmask(input_teeth_part, new_teeth != 0, new_teeth)
    # cv2.imwrite(output_path, input_img)


# save_ref_imgs("ref_imgs")
# change_teeth('input_imgs/Capture.PNG', "1.png")
# change_teeth('input_imgs/1.PNG', "2.png")
# change_teeth('input_imgs/4.PNG', "2.png")
# change_teeth('input_imgs/2.PNG', "3.png")
# change_teeth('input_imgs/5.PNG', "3.png")
# change_teeth('input_imgs/Capture.PNG',  "input_imgs/2.png", "2.png")
# change_teeth('input_imgs/Capture.PNG',  "input_imgs/3.png", "3.png")
# change_teeth('input_imgs/Capture.PNG',  "input_imgs/4.png", "4.png")
# change_teeth('input_imgs/Capture.PNG',  "input_imgs/5.png", "5.png")
# change_teeth('input_imgs/Capture.PNG',  "input_imgs/6.jpg", "6.png")

extract_teeth_parts_in_dir('input_imgs', "output")