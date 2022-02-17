import os
import pandas as pd
import json
import shutil
from PIL import Image
IMG_EXTENSIONS = ['.png', '.jpg']


def find_inputs(folder, filename_to_true=None, types=IMG_EXTENSIONS):
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


def label(root, target_file='images.csv'):
    if target_file:
        target_file_path = os.path.join(root, target_file)
        target_df = pd.read_csv(target_file_path)  # , header=None)
        target_df["TrueLabel"] = target_df["TrueLabel"].apply(int)
        True_label = dict(zip(target_df["ImageId"], target_df["TrueLabel"] - 1))  # -1 for 0-999 class ids
    imgs = find_inputs(root, filename_to_true=True_label)
    new_json_path = "./classes_name.json"
    imagenet_json_path = "./imagenet_class_index.json"
    new_label = json.load(open(new_json_path, "r"))
    imagenet_label = json.load(open(imagenet_json_path, "r"))
    # label_dict = dict([(v[1], v[0]) for k, v in imagenet_label.items()])
    number = 0
    if not os.path.exists('./new_image'):
        os.mkdir('./new_image')

    for img in imgs:
        key = imagenet_label[str(img[1])] #获取图片在ImageNet标签文件中的标签
        # print(img[1])
        # print(imagenet_label[str(img[1])])
        if key[0] in new_label:
            print(new_label[key[0]][0])
            img_name = os.path.basename(img[0])  # 读取文件名
            new_img_name = img_name.split('.')[0] + '_'+str(new_label[key[0]][0])
            # img_value = Image.open(img[0]).convert('RGB')
            par_path = os.path.dirname(img[0])
            new_path = os.path.join(os.path.dirname(par_path),'new_image', new_img_name+'.png')
            shutil.copy(img[0], new_path)
            # number += 1

            # print(number)

    # label_dict = dict([(v[1], v[0]) for k, v in label_dict.items()])
    # print('1')


def main():
    label('../archive', 'images.csv')


if __name__ == '__main__':
    main()
