# Algorithm framework

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable as V


class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode=False):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        # self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def test_one_img(self, img):
        pred = self.net.forward(img)

        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()

        return mask

    def test_batch(self):
        self.forward(volatile=True)
        mask = self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0

        return mask, self.img_id

    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())

        mask = self.net.forward(img).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0

        return mask

    def test_one_img_from_path_1(self, path):
        img = np.array(path)  # .transpose(2,0,1)[None]
        # img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())

        mask = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask1 = mask[:4] + mask[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()

        return loss.data

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        print("update learning rate: %f -> %f" % (self.old_lr, new_lr))
        self.old_lr = new_lr