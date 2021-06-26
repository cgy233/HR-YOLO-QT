import os
import cv2
import json
import torch
import random
import numpy as np
import torch.utils.data as data
from utils.transform import get_transform, transform_keypoint, generate_target_unbiased_encoding



class HandDataset(data.Dataset):
    def __init__(self,
                 data_dir,
                 min_area=2500,
                 training=True,
                 image_size=(224, 224)):
        self.data_dir = data_dir
        self.min_area = min_area
        self.training = True
        self.image_size = image_size
        self.sigma = 1.5
        self.imgnames = []
        self.boxs = []
        self.kpts = []
        self.perfixs = []
        self.vaild_idxs = []

    def add_anno_file(self, filename):
        with open(filename, "rb") as f:
            data = json.load(f)
        # self.imgnames.extend(data["imgname"])
        # self.boxs.extend(data["box"])
        # self.kpts.extend(data["kpt"])
        # self.perfixs.extend(data["perfix"])

        self.imgnames.extend(data["file_name"])
        self.boxs.extend(data["bbox"])
        self.kpts.extend(data["keypoints"])
        self.perfixs.extend(data["id"])
        for idx in range(len(self.boxs)):
            _, _, w, h = self.boxs[idx]
            if w * h >= self.min_area:
                self.vaild_idxs.append(idx)

    def __len__(self):
        return len(self.vaild_idxs)

    def __getitem__(self, i):
        idx = self.vaild_idxs[i]
        img = cv2.imread(
            os.path.join(self.data_dir, self.perfixs[idx],
                         self.imgnames[idx]))[:, :, ::-1]
        box = np.array(self.boxs[idx])
        kpt = np.array(self.kpts[idx]).reshape(-1, 3)
        if self.training and random.random() > 0.5:
            img = img[:, ::-1].copy()
            kpt[:, 0] = img.shape[1] - kpt[:, 0] - 1
            box[0] = img.shape[1] - box[0] - box[2] - 1
        x, y, w, h = box
        scale = max(w, h) / 200. * 1.25
        center = (x + w / 2., y + h / 2.)
        rot = 0
        if self.training:
            center += (random.uniform(-0.05, 0.05) * w,
                       random.uniform(-0.05, 0.05) * h)
            scale *= random.uniform(0.9, 1.1)
            rot = random.randint(-45, 45)
        meta = get_transform(center, scale, self.image_size, rot)
        warp_img = cv2.warpAffine(img,
                                  meta[:2],
                                  self.image_size,
                                  flags=cv2.INTER_LINEAR)
        warp_kpt = transform_keypoint(kpt, meta)
        # filename = str(idx) + ".jpg"
        # show_image(warp_img, warp_kpt, filename)
        target, target_weight = generate_target_unbiased_encoding(
            warp_kpt, sigma=self.sigma)
        input_tensor = np.transpose(warp_img / 255., (2, 0, 1))

        info = {
            "image": input_tensor.astype(np.float32),
            "target": target.astype(np.float32),
            "kpt": kpt.astype(np.float32),
            "warp_kpt": warp_kpt.astype(np.float32),
            "meta": meta.astype(np.float32),
            "weight": target_weight.astype(np.float32),
            "normalize": min(w, h)
        }
        return info


if __name__ == "__main__":
    from glob import glob
    from tqdm import tqdm
