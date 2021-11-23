import torch.utils.data as data
import torchvision.transforms as transforms
import torch

import os
from PIL import Image
import pandas as pd

IMG_EXTENSIONS = ['.png', '.jpg']


class LeNormalize(object):
    """Normalize to -1..1 in Google Inception style
    """
    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor


def default_inception_transform(img_size):
    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        LeNormalize(),
    ])
    return tf


def find_inputs(folder, filename_to_true=None, filename_to_target=None, types=IMG_EXTENSIONS):
    inputs = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            base, ext = os.path.splitext(rel_filename)
            if ext.lower() in types:
                abs_filename = os.path.join(root, rel_filename)
                True_label = filename_to_true[rel_filename.split('.')[0]] if filename_to_true else 0
                # Traget_label = filename_to_target[rel_filename.split('.')[0]] if filename_to_target else 0
                # inputs.append((abs_filename, True_label, Traget_label))
                inputs.append((abs_filename, True_label))
    return inputs


class Dataset(data.Dataset):
    
    def __init__(self, root, target_file='images.csv', transform=None):
        
        if target_file:
            target_file_path = os.path.join(root, target_file)
            target_df = pd.read_csv(target_file_path)#, header=None)
            target_df["TrueLabel"] = target_df["TrueLabel"].apply(int)
            # print(target_df["ImageId"], target_df["TrueLabel"])
            True_label = dict(zip(target_df["ImageId"], target_df["TrueLabel"] - 1))  # -1 for 0-999 class ids
            # Target_label = dict(zip(target_df["ImageId"], target_df["TargetClass"] - 1))
        else:
            True_label = dict()
        imgs = find_inputs(root, filename_to_true=True_label)
        # imgs = find_inputs(root, filename_to_true=True_label, filename_to_target=Target_label)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        # path, true, target = self.imgs[index]
        path, true = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if true is None:
            true = torch.zeros(1).long()
        # if target is None:
        #     target = torch.zeros(1).long()
        # return img, true, target
        return img, true

    def __len__(self):
        return len(self.imgs)

    def set_transform(self, transform):
        self.transform = transform

    def filenames(self, indices=[], basename=False):
        if indices:
            if basename:
                return [os.path.basename(self.imgs[i][0]) for i in indices]
            else:
                return [self.imgs[i][0] for i in indices]
        else:
            if basename:
                return [os.path.basename(x[0]) for x in self.imgs]
            else:
                return [x[0] for x in self.imgs]

