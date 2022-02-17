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


def find_inputs(folder, types=IMG_EXTENSIONS):
    inputs = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            base, ext = os.path.splitext(rel_filename)
            if ext.lower() in types:
                abs_filename = os.path.join(root, rel_filename)
                True_label = rel_filename.split('.')[0]
                True_label = int(True_label.split('_')[1])
                # Traget_label = filename_to_target[rel_filename.split('.')[0]] if filename_to_target else 0
                # inputs.append((abs_filename, True_label, Traget_label))
                inputs.append((abs_filename, True_label))
    return inputs


class Dataset(data.Dataset):

    def __init__(self, root, transform=None):
        imgs = find_inputs(root)
        # imgs = find_inputs(root, filename_to_true=True_label, filename_to_target=Target_label)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

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


# def main():
#     dataset = Dataset('../dataset')
#     # inputs = find_inputs('../dataset')


# if __name__ == '__main__':
#     main()

