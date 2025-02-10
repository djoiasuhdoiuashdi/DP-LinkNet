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

from networks.dplinknet import LinkNet34, DLinkNet34, DPLinkNet34
from utils import load_image_as_binary, calculate_psnr, get_filename_and_extension, get_image_files, get_patches, \
    stitch_together

TILE_SIZE = 256


def calculate_total_psnr(solver):
    validation_images_path = "./dataset/validation"
    validation_images_ground_truth_path = "./dataset/validation/GT"

    validation_images = get_image_files(validation_images_path)

    psnr = 0.0

    for image in validation_images:

        file_name, file_ext = get_filename_and_extension(image)

        input_image = cv2.imread(image)
        locations, patches, padded, pad_bottom, pad_right, original_height, original_width = get_patches(input_image,
                                                                                                         TILE_SIZE,
                                                                                                         TILE_SIZE)
        masks = []
        for idy in range(len(patches)):
            msk = solver.test_one_img_from_path_1(patches[idy])
            masks.append(msk)

        prediction = stitch_together(locations, masks, [original_height + pad_bottom, original_width + pad_right],
                                     TILE_SIZE, TILE_SIZE)
        prediction[prediction < 5.0] = 0
        prediction[prediction >= 5.0] = 1

        image_gt = load_image_as_binary(os.path.join(validation_images_ground_truth_path, f"{file_name}_GT.tiff"))

        psnr += calculate_psnr(prediction, image_gt)

    return psnr


SHAPE = (TILE_SIZE, TILE_SIZE)
DATA_NAME = "v4"  # BickleyDiary, DIBCO, PLM
DEEP_NETWORK_NAME = "DPLinkNet34"
print("Now training dataset: {}, using network model: {}".format(DATA_NAME, DEEP_NETWORK_NAME))

train_root = "./dataset/train/"
imagelist = list(filter(lambda x: x.find("img") != -1, os.listdir(train_root)))
trainlist = list(map(lambda x: x[:-8], imagelist))
log_name = DATA_NAME.lower() + "_" + DEEP_NETWORK_NAME.lower()

BATCHSIZE_PER_CARD = 46  # 160
LEARNING_RATE = 2e-4 * BATCHSIZE_PER_CARD / 32  # 0.001
NUM_OF_WORKERS = 14  # 80

solver = MyFrame(DPLinkNet34, dice_bce_loss, LEARNING_RATE)

batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

dataset = ImageFolder(trainlist, train_root)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=NUM_OF_WORKERS)

mylog = open("logs/" + log_name + ".log", "w")
loss_no_optim = 0
total_epoch = 500
train_epoch_best_loss = 100.
psnr_no_optim = 0
best_PSNR = 0.

wandb.init(
    project="DP-LinkNet",
    config={
        "batch_size": torch.cuda.device_count() * BATCHSIZE_PER_CARD,
        "epochs": 200,
        "learning_rate": LEARNING_RATE
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
    current_PSNR = calculate_total_psnr(solver)
    solver.net.train()

    print("********", file=mylog)
    print("epoch:", epoch, "    time:", int(time() - tic), file=mylog)
    print("train_loss:", train_epoch_loss, file=mylog)
    print("PSNR:", current_PSNR, file=mylog)
    print("SHAPE:", SHAPE, file=mylog)
    print("********")
    print("epoch:", epoch, "    time:", int(time() - tic))
    print("train_loss:", train_epoch_loss)
    print("PSNR:", current_PSNR)
    print("SHAPE:", SHAPE)

    wandb.log({
        "train_loss": train_epoch_loss,
        "learning_rate": solver.old_lr,
        "PSNR": current_PSNR
    })

    if train_epoch_loss >= train_epoch_best_loss:
        loss_no_optim += 1
    else:
        loss_no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save("weights/" + log_name + ".th")

    if epoch >= 20:
        if current_PSNR > best_PSNR:
            solver.save("weights/" + log_name + "_" + str(epoch) + ".th")
            best_PSNR = current_PSNR
            psnr_no_optim = 0
        else:
            psnr_no_optim += 1

    if loss_no_optim > 20 or psnr_no_optim > 20:
        print("early stop at %d epoch" % epoch, file=mylog)
        print("early stop at %d epoch" % epoch)
        break

    if loss_no_optim > 10:
        if solver.old_lr < 1e-7:
            break
        solver.load("weights/" + log_name + ".th")
        solver.update_lr(5.0, factor=True, mylog=mylog)
    mylog.flush()

print("Finish!", file=mylog)
print("Finish!")
mylog.close()