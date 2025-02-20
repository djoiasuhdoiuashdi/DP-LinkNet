# Prepare dataset for training

import os

import cv2
import argparse
from utils import get_patches

parser = argparse.ArgumentParser(description='Prepare dataset for training.')
parser.add_argument('--input', type=str, required=True)
args=parser.parse_args()

TILE_SIZE = 256
print("Image patch size:", TILE_SIZE, "x", TILE_SIZE)

data_root = os.path.join("./dataset/", args.input)
img_list = os.listdir(os.path.join(data_root, "images"))
img_list.sort()

data_train_dir = os.path.join("./dataset/", "train_" + args.input)
if not os.path.exists(data_train_dir):
    os.makedirs(data_train_dir)

# img_patches, msk_patches = [], []  # patches for each image or ground truth
total_img_patches, total_msk_patches = [], []  # patches for all the images or ground truths

for idx in range(len(img_list)):
    if os.path.isdir(os.path.join(data_root, img_list[idx])):
        continue

    print("Now processing image:", os.path.join(data_root, img_list[idx]))
    (fname, fext) = os.path.splitext(img_list[idx])
    img = cv2.imread(os.path.join(data_root,"images", img_list[idx]))
    msk = cv2.imread(os.path.join(data_root, "images_gt", fname + "_GT.tiff"))

    # extract the patches from the original document images and the corresponding ground truths
    img_patch_locations, img_patches, padded, pad_bottom, pad_right, original_height, original_width = get_patches(img,
                                                                                                                   TILE_SIZE,
                                                                                                                   TILE_SIZE)
    msk_patch_locations, msk_patches, padded, pad_bottom, pad_right, original_height, original_width = get_patches(msk,
                                                                                                                   TILE_SIZE,
                                                                                                                   TILE_SIZE)

    print("Patches extracted:", len(img_patches))
    for idy in range(len(img_patches)):
        total_img_patches.append(img_patches[idy])
        total_msk_patches.append(msk_patches[idy])

print("Total number of train patches:", len(total_img_patches))
for idz in range(len(total_img_patches)):
    cv2.imwrite(os.path.join(data_train_dir, str(idz) + "_img.png"), total_img_patches[idz])
    cv2.imwrite(os.path.join(data_train_dir, str(idz) + "_mask.png"), total_msk_patches[idz])

print("Done")