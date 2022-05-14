# Stochastic-Serial-Attack
## Introduction
This repository include the code for "Boosting the Transferability of Adversarial Examples via Stochastic Serial Attack"

## Method
We propose a novel attack strategy called stochastic serial attack (SSA). It adopts a serial strategy to attack local models, which reduces memory consumption compared to parallel attacks. Moreover, since local models are stochastically selected from a large model set, it can ensure that the adversarial examples do not overfit specific weaknesses of local source models.

## Citations
If you use Stochastic Series Attack in your research, please consider citing


## Getting started
1. `git clone https://github.com/haolingguang/SSA.git`  
2. Install dependencies:  
  `Python >= 3.6`   
  `Pytorch >=1.3`  
  `pip install pretrainedmodels`  
[pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch) is a pretrained models repository of pytorch, which include all pytorch models used in our paper.  
  
3. `archive` include the dataset in NeurIPS 2017 adversarial competition. If you want use `ImageNet`, you need download corresponding dataset and use `torchvision.datasets.ImageFolder` to load dataset.  

4. run attack  
`python SSA.py`  

5. If you need load adversarial training models, you need use https://github.com/haolingguang/tensorflow-to-pytorch.git to convert tensorflow's pre-training models to pytorch's

6. In order to get the mini-ImageNet, you can refer to https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/mini_imagenet
7. Create attack dataset for the models trained by mini-ImageNet, you need run `Rename_image.py` in `mini_dataset`.
# Note
Package "pretrainedmodels" maybe have some problem, But you can use another pytorch model package [timm](https://rwightman.github.io/pytorch-image-models/)   
For example: parallel computing     
t = timm.create_model('resnet50', num_classes=1000, pretrained=True)     
t = torch.nn.DataParallel(t).cuda()     

