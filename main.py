import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, datasets

import torchvision
from torchvision import transforms
from torchvision import models

import torch.nn.functional as F
import torchvision.transforms.functional as TF

import albumentations

from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import time
import nibabel as nib
from tqdm import tqdm
import pandas as pd

import skimage
from skimage import img_as_ubyte, img_as_float32

from sklearn.model_selection import StratifiedShuffleSplit

from glob import glob

np.random.seed(108)

plt.style.use('default')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

visualize = False
