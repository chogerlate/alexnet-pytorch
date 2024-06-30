# AlexNet PyTorch Implementation

This project implements AlexNet, a pioneering convolutional neural network architecture, using PyTorch. The implementation is based on the original paper by Krizhevsky et al. (2012).


![AlexNet Architecture](AlexNet-1.png)

## Features

- Full implementation of AlexNet architecture
- Customizable number of classes
- Weight initialization as per the original paper
- Training script with command-line arguments using Typer
- Support for custom datasets

## Getting Started
```
git clone git@github.com:chogerlate/alexnet-pytorch.git
```

## Install the required packages using:
```
pip install -r requirements.txt
```

## Load mini dataset 
```
python load_sample_dataset.py
```

## Training
```
python train.py
```
### Optional parammeters
```
python train.py --help
```



## References

Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton (2012). 
ImageNet classification with deep convolutional neural networks. Part of Advances in Neural Information Processing Systems 25 (NIPS 2012) 
- https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html

PyTorch Going Modular
- https://www.learnpytorch.io/05_pytorch_going_modular/#6-train-evaluate-and-save-the-model-trainpy

Understand Alexnet
- https://learnopencv.com/understanding-alexnet/