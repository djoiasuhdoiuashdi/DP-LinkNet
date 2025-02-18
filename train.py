import os
from time import time
import numpy as np
import torch
import cv2
import torch.utils.data as data
import wandb
from torch.autograd import Variable as V
from data import ImageFolder
from framework import MyFrame
from loss import dice_bce_loss
from metrics import calculate_metrics

from networks.dplinknet import LinkNet34, DLinkNet34, DPLinkNet34
from utils import load_image_as_binary, get_filename_and_extension, get_image_files, get_patches, \
    stitch_together

TILE_SIZE = 256


def calculate_val_metrics(metric_solver):
    validation_images_path = "./dataset/validation"
    validation_images_ground_truth_path = "./dataset/validation/GT"

    validation_images = get_image_files(validation_images_path)

    total_fmeasure = 0.0
    total_pfmeasure = 0.0
    total_psnr = 0.0
    total_drd = 0.0

    length = len(validation_images)

    for image in validation_images:

        file_name, file_ext = get_filename_and_extension(image)

        input_image = cv2.imread(image)
        locations, patches, padded, pad_bottom, pad_right, original_height, original_width = get_patches(input_image,
                                                                                                         TILE_SIZE,
                                                                                                         TILE_SIZE)
        masks = []
        for idy in range(len(patches)):
            msk = metric_solver.test_one_img_from_path_1(patches[idy])
            masks.append(msk)

        prediction = stitch_together(locations, masks, [original_height + pad_bottom, original_width + pad_right],
                                     TILE_SIZE, TILE_SIZE)
        prediction[prediction < 5.0] = 0
        prediction[prediction >= 5.0] = 1

        image_gt = load_image_as_binary(os.path.join(validation_images_ground_truth_path, f"{file_name}_GT.tiff"))


        r_weight = np.loadtxt(os.path.join("./dataset/validation/r_weights", file_name + "_GT_RWeights.dat"),
                              dtype=np.float64).flatten()[:original_height * original_width].reshape(
            (original_height, original_width))
        p_weight = np.loadtxt(os.path.join("./dataset/validation/p_weights", file_name + "_GT_PWeights.dat"),
                              dtype=np.float64).flatten()[:original_height * original_width].reshape(
            (original_height, original_width))
        fmeasure, pfmeasure, psnr, drd = calculate_metrics(prediction, image_gt, r_weight, p_weight)
        total_psnr += psnr
        total_fmeasure += fmeasure
        total_pfmeasure += pfmeasure
        total_drd += drd

    total_fmeasure = total_fmeasure / length
    total_pfmeasure = total_pfmeasure / length
    total_psnr = total_psnr / length
    total_drd = total_drd / length

    return total_fmeasure, total_pfmeasure, total_psnr, total_drd


SHAPE = (TILE_SIZE, TILE_SIZE)
train_root = "./dataset/train/"
imagelist = list(filter(lambda x: x.find("img") != -1, os.listdir(train_root)))
trainlist = list(map(lambda x: x[:-8], imagelist))

BATCHSIZE_PER_CARD = 46  # 160
LEARNING_RATE = 2e-4 * BATCHSIZE_PER_CARD / 32  # 0.001
NUM_OF_WORKERS = 32  # 80

solver = MyFrame(DPLinkNet34, dice_bce_loss, LEARNING_RATE)

batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

dataset = ImageFolder(trainlist, train_root)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=NUM_OF_WORKERS)

loss_no_optim = 0
total_epoch = 500
train_epoch_best_loss = 100.
psnr_no_optim = 0
best_PSNR = 0.
best_fmeasure = 0.
best_pfmeasure = 0.

wandb.init(
    project="DP-LinkNet",
    config={
        "batch_size_per_card": BATCHSIZE_PER_CARD,
        "batch_size": torch.cuda.device_count() * BATCHSIZE_PER_CARD,
        "epochs": 200,
        "learning_rate": LEARNING_RATE,
        "dataset": train_root,
        "num_of_workers": NUM_OF_WORKERS,

    }
)

tic = time()
for epoch in range(1, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    for img, mask in data_loader_iter:
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss

    train_epoch_loss /= len(data_loader_iter)
    solver.net.eval()
    current_fmeasure, current_pfmeasure, current_psnr, current_drd = calculate_val_metrics(solver)
    solver.net.train()

    print("********")
    print("epoch:", epoch, "    time:", int(time() - tic))
    print("train_loss:", train_epoch_loss)
    print("Fmeasure:", current_fmeasure)
    print("pFmeasure:", current_pfmeasure)
    print("PSNR:", current_psnr)
    print("DRD:", current_drd)
    print("SHAPE:", SHAPE)

    wandb.log({
        "train_loss": train_epoch_loss,
        "learning_rate": solver.old_lr,
        "fmeasure": current_fmeasure,
        "pfmeasure": current_pfmeasure,
        "psnr": current_psnr,
        "drd": current_drd
    })

    if train_epoch_loss >= train_epoch_best_loss:
        loss_no_optim += 1
    else:
        loss_no_optim = 0
        train_epoch_best_loss = train_epoch_loss

    if current_psnr > best_PSNR:
        solver.save("weights/best_psnr.th")
        best_PSNR = current_psnr

    if current_fmeasure > best_fmeasure:
        solver.save("weights/best_fmeasure.th")
        best_fmeasure = current_fmeasure

    if current_pfmeasure > best_pfmeasure:
        solver.save("weights/best_pfmeasure.th")
        best_pfmeasure = current_pfmeasure

    if epoch >= 20:
        if current_psnr > best_PSNR:
            psnr_no_optim = 0
        else:
            psnr_no_optim += 1

    if loss_no_optim > 20 or psnr_no_optim > 20:
        print("early stop at %d epoch" % epoch)
        break

    if loss_no_optim > 10:
        if solver.old_lr < 1e-7:
            break
        solver.load("weights/best_psnr.th")
        solver.update_lr(5.0, factor=True)

print("Finish!")
