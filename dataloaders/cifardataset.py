import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import skimage
from skimage import img_as_ubyte, img_as_float32
import utils.help
import utils.util
import torchvision
from torchvision import models
import pandas as pd
import os



#########################################
# This class generates patches for training
#########################################

class MyDataset(Dataset):
    # def __init__(self, patch_dim, gap, df, validate, transform=None):
    def __init__(self,cfg, patch_dim, gap, annotation_file, validate, transform=None):
        self.patch_dim, self.gap = patch_dim, gap
        self.transform = transform
        self.label_path = os.path.join(cfg.root_path,cfg.data_path,cfg.labels_dir,annotation_file)
        self._load_data()
        # if validate:
        #     self.train_data = df.values
        # else:
        #     self.train_data = df.values

    def _load_data(self):
        self.labels = pd.read_csv(self.label_path)

        self.loaded_data = []
        for i in range(self.labels.shape[0]):
            img_name = self.labels['FileName'][i]
            label = self.labels['Label'][i]
            img = Image.open(img_name)
            self.loaded_data.append((img, label, img_name))
            img.load()

    #            self.read_data.append((img,label))

    def get_patch_from_grid(self, image, patch_dim, gap):
        image = np.array(image)

        offset_x, offset_y = image.shape[0] - (patch_dim * 3 + gap * 2), image.shape[1] - (patch_dim * 3 + gap * 2)
        # start_grid_x, start_grid_y = np.random.randint(0, offset_x), np.random.randint(0, offset_y)
        start_grid_x, start_grid_y = 0, 0
        patch_loc_arr = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]
        loc = np.random.randint(len(patch_loc_arr))
        tempx, tempy = patch_loc_arr[loc]

        patch_x_pt = start_grid_x + patch_dim * (tempx - 1) + gap * (tempx - 1)
        patch_y_pt = start_grid_y + patch_dim * (tempy - 1) + gap * (tempy - 1)
        random_patch = image[patch_x_pt:patch_x_pt + patch_dim, patch_y_pt:patch_y_pt + patch_dim]

        patch_x_pt = start_grid_x + patch_dim * (2 - 1) + gap * (2 - 1)
        patch_y_pt = start_grid_y + patch_dim * (2 - 1) + gap * (2 - 1)
        uniform_patch = image[patch_x_pt:patch_x_pt + patch_dim, patch_y_pt:patch_y_pt + patch_dim]

        random_patch_label = loc

        return uniform_patch, random_patch, random_patch_label

    def __len__(self):
        return len(self.loaded_data)

    def __getitem__(self, index):
        # image = Image.open(self.loaded_data[index]).convert('RGB')
        index = index % len(self.loaded_data)
        image,label,img_name = self.loaded_data[index]
        uniform_patch, random_patch, random_patch_label = self.get_patch_from_grid(image,
                                                                                   self.patch_dim,
                                                                                   self.gap)
        if uniform_patch.shape[0] != 96:
            uniform_patch = skimage.transform.resize(uniform_patch, (96, 96))
            random_patch = skimage.transform.resize(random_patch, (96, 96))

            uniform_patch = img_as_float32(uniform_patch)
            random_patch = img_as_float32(random_patch)

        # Dropped color channels 2 and 3 and replaced with gaussian noise(std ~1/100 of the std of the remaining channel)
        uniform_patch[:, :, 1] = np.random.normal(0.485, 0.01 * np.std(uniform_patch[:, :, 0]),
                                                  (uniform_patch.shape[0], uniform_patch.shape[1]))
        uniform_patch[:, :, 2] = np.random.normal(0.485, 0.01 * np.std(uniform_patch[:, :, 0]),
                                                  (uniform_patch.shape[0], uniform_patch.shape[1]))
        random_patch[:, :, 1] = np.random.normal(0.485, 0.01 * np.std(random_patch[:, :, 0]),
                                                 (random_patch.shape[0], random_patch.shape[1]))
        random_patch[:, :, 2] = np.random.normal(0.485, 0.01 * np.std(random_patch[:, :, 0]),
                                                 (random_patch.shape[0], random_patch.shape[1]))

        random_patch_label = np.array(random_patch_label).astype(np.int64)

        if self.transform:
            uniform_patch = self.transform(uniform_patch)
            random_patch = self.transform(random_patch)

        return uniform_patch, random_patch, random_patch_label


##################################################
# Creating Train/Validation dataset and dataloader
##################################################
# config_path = r'D:\paper\Self_Supervised_Learning\codes\context\config\config.yml'
# Config = utils.help.load_yaml(config_path,config_type='object')
#
# annotation_file = 'cifar100_recognition_train.csv'
# traindataset = MyDataset(Config, Config.patch_dim, Config.gap, annotation_file, False,
#                          transforms.Compose([transforms.ToTensor(),
#                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                                   std=[0.229, 0.224, 0.225])]))
# trainloader = torch.utils.data.DataLoader(traindataset,
#                                           batch_size=Config.batch_size,
#                                           shuffle=True,
#                                           # num_workers=Config.num_workers
#                                           )
# annotation_file = 'cifar100_recognition_val.csv'
# valdataset = MyDataset(Config, Config.patch_dim, Config.gap, annotation_file, True,
#                        transforms.Compose([transforms.ToTensor(),
#                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                                 std=[0.229, 0.224, 0.225])]))
# valloader = torch.utils.data.DataLoader(valdataset,
#                                         batch_size=Config.batch_size,
#                                         shuffle=False)
#
# ##############################
# # Visualizing training dataset
# ##############################
#
# example_batch = next(iter(trainloader))
# concatenated = torch.cat((utils.util.unorm(example_batch[0]),utils.util.unorm(example_batch[1])),0)
# utils.util.imshow(torchvision.utils.make_grid(concatenated))
# print(f'Labels: {example_batch[2].numpy()}')
#
# ##############################
# # Visualizing validation dataset
# ##############################
#
# example_batch_val = next(iter(valloader))
# concatenated = torch.cat((utils.util.unorm(example_batch_val[0]),utils.util.unorm(example_batch_val[1])),0)
# utils.util.imshow(torchvision.utils.make_grid(concatenated))
# print(f'Labels: {example_batch_val[2].numpy()}')