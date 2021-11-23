import os
import os.path

import torch
import torch.utils.data as data
import pretrainedmodels
import argparse
from Dataset import Dataset, default_inception_transform
import datetime
from random import sample
from torchvision import datasets, transforms


def seed_torch(seed=20):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


seed_torch()

parser = argparse.ArgumentParser(description="SSA")
parser.add_argument('--max_epsilon', default=8, type=int, help='max perturbation')
parser.add_argument('--norm', default=1, type=int, help='lp norm')
parser.add_argument('--num_steps', default=10, type=int, help='attack steps')
parser.add_argument('--gpu', default='0', type=str, help='id of gpu')
parser.add_argument('--epoch', default=5, type=int, help='repeated experiment n times')
parser.add_argument('--batch_size', default=2, type=int, help='mini-batch size (default: 4)')
# parser.add_argument('--num_workers', default=4, type=int, help='Max Number of CPU cores - 1')
parser.add_argument('--image_size', default=299, type=int, help='the size of the original image')
parser.add_argument('--local_model', default=['resnet152', 'xception', 'inceptionv3', 'dpn98',
                    'resnet18', 'dpn131', 'resnet34', 'resnet50'], type=list, help='local_models names')
parser.add_argument('--local_model_len', default=4, type=int, help='local model series len')
parser.add_argument('--remote_model', default='dpn68', type=str)
parser.add_argument('--NIPS_data', default='./archive', type=str, help='NIPS dataset directory')
# parser.add_argument('--ImageNet_data', default='./ImageNet/ILSVRC/Data/CLS-LOC', type=str, help='ImageNet dataset directory')
# parser.add_argument('--output_dir', default='./output', type=str, help='Adversarial Images directory')
parser.add_argument('--categories', default='./archive/categories.csv', type=str, help='categories directory')
parser.add_argument('--log_dir', default='./log', type=str, help='output directory')

args = parser.parse_args()
print("args", args)


if not args.gpu == 'None':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

LOG_DIR = os.path.join(args.log_dir, args.remote_model)
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_attack.txt'), 'a')
LOG_FOUT.write(str(args) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


class Attack:
    def __init__(self, max_epsilon=16, norm=float('inf'), num_steps=None):

        self.eps = 2.0 * max_epsilon / 255.0
        self.num_steps = num_steps
        self.norm = norm
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn = self.loss_fn.cuda()

    def non_target_attack(self, models_index, x, targets, steps):
        input_var = torch.autograd.Variable(x, requires_grad=True)
        targets_var = torch.autograd.Variable(targets)
        eps = self.eps
        g = 0.0
        mu = 1.0
        step_alpha = self.eps/steps
        for step in range(steps):
            # select local models
            models = []
            for i in range(0, len(models_index[step])):
                t = pretrainedmodels.__dict__[models_index[step][i]](num_classes=1000, pretrained='imagenet').cuda()
                models.append(t)
                models[i].eval()
            '''
            If you want obtain the better result, you can use learning rate schedule. 
            Because we have used "clamp" for perturbation, you don't have to worry about the perturbation overflow
            '''
            # step_alpha = eps * (1 - (step / steps) ** 0.9)
            for i in range(0, len(models)):
                output = models[i](input_var)
                loss = self.loss_fn(output, targets_var)
                loss.backward()
                # select lp norm
                if self.norm == 2:
                    normed_grad = torch.norm(torch.flatten(input_var.grad.data, start_dim=1), p=2, dim=1)
                    normed_grad = torch.flatten(input_var.grad.data, start_dim=1) / normed_grad.reshape(
                        input_var.shape[0], -1)
                    normed_grad = normed_grad.reshape(input_var.shape)
                elif self.norm == 1:
                    normed_grad = torch.norm(torch.flatten(input_var.grad.data, start_dim=1), p=1, dim=1)
                    normed_grad = torch.flatten(input_var.grad.data, start_dim=1) / normed_grad.reshape(
                        input_var.shape[0], -1)
                    normed_grad = normed_grad.reshape(input_var.shape)
                else:
                    normed_grad = torch.sign(input_var.grad.data)
                g = mu * g + normed_grad
                step_adv = input_var.data + step_alpha * torch.sign(g)

                # calculate total adversarial perturbation from original image and clip to epsilon constraints
                total_adv = step_adv - x
                total_adv = torch.clamp(total_adv, -eps, eps)

                # apply total adversarial perturbation to original image and clip to valid pixel range
                input_adv = x + total_adv
                input_adv = torch.clamp(input_adv, -1.0, 1.0)

                # Update the adversarial example and clear the gradient
                input_var.data = input_adv
                input_var.grad.data.zero_()

        return input_adv


def steps_attack(args, attack):
    # load NIPS 2017 adversarial dataset
    dataset = Dataset(args.NIPS_data, transform=default_inception_transform(args.image_size))
    NIPS_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # val_dir = os.path.join(args.ImageNet_data, 'val')
    # trans = transforms.Compose([
    #     transforms.Resize(299),
    #     transforms.CenterCrop(299),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])])
    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(val_dir, trans),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.num_workers, pin_memory=True)

    # load remote model
    remote_model = pretrainedmodels.__dict__[args.remote_model](num_classes=1000, pretrained='imagenet').cuda()
    remote_model.eval()

    # select local models index
    models_index = []
    for i in range(args.num_steps):
        models_index.append(sample(args.local_model, 4))

    for steps in range(args.num_steps):
        steps += 1
        correct = 0
        n_target_success = 0
        for batch_idx, (input_x, true_label) in enumerate(NIPS_loader):
            input_x = input_x.cuda()
            true_label = true_label.cuda()
            # train adversarial examples
            non_target_adv = attack.non_target_attack(models_index, input_x, true_label, steps)

            # test transferability of adversarial examples
            _, non_target_label = torch.max(remote_model(non_target_adv), 1)
            _, og_label = torch.max(remote_model(input_x), 1)
            n_target_success += torch.sum(non_target_label != true_label)
            correct += torch.sum(og_label == true_label)

        n_target_acc = 1 - (n_target_success / 1000)
        accuracy = correct / 1000
        log_string('steps: %d | non_target_acc: %.3f |  model_original_accuracy: %.3f' % (steps, n_target_acc, accuracy))


def main():
    attack = Attack(
        max_epsilon=args.max_epsilon,
        norm=args.norm,
        num_steps=args.num_steps)
    # to eliminate contingency, repeat the experiment "epoch" times
    for epoch in range(args.epoch):
        log_string('epoch: %d' % epoch)
        start = datetime.datetime.now()
        steps_attack(args, attack)
        end = datetime.datetime.now()
        log_string('time: %s' % (end - start))


if __name__ == '__main__':
    main()
