import numpy as np
from pathlib import Path
import os
import cv2

TILE_SIZE = 128
PADDING_SIZE = 21  # round(TILE_SIZE / 4)

LEFT_EDGE = -2
TOP_EDGE = -1
MIDDLE = 0
RIGHT_EDGE = 1
BOTTOM_EDGE = 2


def get_patches(img, patch_h=TILE_SIZE, patch_w=TILE_SIZE):
    y_stride, x_stride, = patch_h - (2 * PADDING_SIZE), patch_w - (2 * PADDING_SIZE)
    if (patch_h > img.shape[0]) or (patch_w > img.shape[1]):
        print("Invalid cropping: Cropping dimensions larger than image shapes (%r x %r with %r)" % (
        patch_h, patch_w, img.shape))
        exit(1)

    locations, patches = [], []
    y = 0
    y_done = False
    while y <= img.shape[0] and not y_done:
        x = 0
        if y + patch_h > img.shape[0]:
            y = img.shape[0] - patch_h
            y_done = True
        x_done = False
        while x <= img.shape[1] and not x_done:
            if x + patch_w > img.shape[1]:
                x = img.shape[1] - patch_w
                x_done = True
            locations.append(((y, x, y + patch_h, x + patch_w),
                              (y + PADDING_SIZE, x + PADDING_SIZE, y + y_stride, x + x_stride),
                              TOP_EDGE if y == 0 else (BOTTOM_EDGE if y == (img.shape[0] - patch_h) else MIDDLE),
                              LEFT_EDGE if x == 0 else (RIGHT_EDGE if x == (img.shape[1] - patch_w) else MIDDLE)))
            patches.append(img[y:y + patch_h, x:x + patch_w, :])
            x += x_stride
        y += y_stride

    return locations, patches


def stitch_together(locations, patches, size, patch_h=TILE_SIZE, patch_w=TILE_SIZE):
    output = np.zeros(size, dtype=np.float32)

    for location, patch in zip(locations, patches):
        outer_bounding_box, inner_bounding_box, y_type, x_type = location
        y_paste, x_paste, y_cut, x_cut, height_paste, width_paste = -1, -1, -1, -1, -1, -1
        # print outer_bounding_box, inner_bounding_box, y_type, x_type

        if y_type == TOP_EDGE:
            y_cut = 0
            y_paste = 0
            height_paste = patch_h - PADDING_SIZE
        elif y_type == MIDDLE:
            y_cut = PADDING_SIZE
            y_paste = inner_bounding_box[0]
            height_paste = patch_h - 2 * PADDING_SIZE
        elif y_type == BOTTOM_EDGE:
            y_cut = PADDING_SIZE
            y_paste = inner_bounding_box[0]
            height_paste = patch_h - PADDING_SIZE

        if x_type == LEFT_EDGE:
            x_cut = 0
            x_paste = 0
            width_paste = patch_w - PADDING_SIZE
        elif x_type == MIDDLE:
            x_cut = PADDING_SIZE
            x_paste = inner_bounding_box[1]
            width_paste = patch_w - 2 * PADDING_SIZE
        elif x_type == RIGHT_EDGE:
            x_cut = PADDING_SIZE
            x_paste = inner_bounding_box[1]
            width_paste = patch_w - PADDING_SIZE

        # print (y_paste, x_paste), (height_paste, width_paste), (y_cut, x_cut)
        output[y_paste:y_paste + height_paste, x_paste:x_paste + width_paste] = patch[y_cut:y_cut + height_paste,
                                                                                x_cut:x_cut + width_paste]

    return output


def load_image_as_binary(image_path):
    """Load a black and white image and convert it to a binary array."""
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Convert the image to binary (0 and 1)
    binary_array = (image > 0).astype(np.uint8)
    return binary_array


def calculate_psnr(im, im_gt):
    height, width = im.shape
    FP_mask = (im == 0) & (im_gt == 1)
    FN_mask = (im == 1) & (im_gt == 0)
    FP = FP_mask.sum()
    FN = FN_mask.sum()
    mse = (FP + FN) / (height * width)
    return 10.0 * np.log10(1.0 / mse) if mse > 0 else float('inf')


def get_image_files(input_dir):
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
    image_files = []

    # Get the list of files in the specified directory
    for file in os.listdir(input_dir):
        file_path = Path(input_dir) / file  # Create full path to the file
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)

    return image_files


def get_filename_and_extension(file_path):
    filename, file_extension = os.path.splitext(os.path.basename(file_path))
    return filename, file_extension