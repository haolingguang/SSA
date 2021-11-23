# Stochastic-Serial-Attack
## Introduction
This repository include the code for "Boosting the Transferability of Adversarial Examples via Stochastic Serial Attack"

## Method
We propose a novel attack strategy called stochastic serial attack (SSA). It adopts a serial strategy to attack local models, which reduces memory consumption compared to parallel attacks. Moreover, since local models are stochastically selected from a large model set, it can ensure that the adversarial examples do not overfit specific weaknesses of local source models.

## Citations
If you use Stochastic Series Attack in your research, please consider citing
Corresponding data set

## Getting started
1. Install dependencies:  
  Python >= 3.6   
  Pytorch >=1.3
  pip install pretrainedmodels  
[pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch) is a pretrained models repository of pytorch, which include all pytorch models used in our paper.
  
2. archive include the dataset in NeurIPS 2017 adversarial competition. If you want use ImageNet, you need download corresponding dataset and use .

