import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def main():
    batchse=32

    cifar_train = datasets.CIFAR100('../dataset/cifar100', True, transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor
    ]), download=True)
    cifar_train = DataLoader(cifar_train,batch_size=batchse,shuffle=True)

    cifar_test = datasets.CIFAR100('../dataset/cifar100', False, transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor
    ]), download=True)
    cifar_teat = DataLoader(cifar_train,batch_size=batchse,shuffle=True)


if __name__ == "__main__":
    main()