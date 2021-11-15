import numpy as np
import pandas as pd
import random

import cv2
import os

root_dir = r'D:\paper\Self_Supervised_Learning\codes\context'
images_dir = os.path.join(root_dir,'dataset','cifar100','test')

data_dir_list = os.listdir(images_dir)
print ('the data list is: ',data_dir_list)

# Assigning labels to each flower category
num_classes = 100

labels=[]

for i in range(0, num_classes):
    labels.append(i)

labels_name=dict(zip(data_dir_list,labels))

# create two dataframes one for train and other for test with 3 columns as filename,label and classname
test_df = pd.DataFrame(columns=['FileName', 'Label', 'ClassName'])

for dataset in data_dir_list:
    # load the list of image names in each of the flower category
    img_list = os.listdir(os.path.join(images_dir, dataset))
    print('Loading the images of dataset-' + '{}\n'.format(dataset))
    label = labels_name[dataset]
    num_img_files = len(img_list)
    num_corrupted_files = 0

    # read each file and if it is corrupted exclude it , if not include it in either train or test data frames
    for i in range(num_img_files):
        img_name = img_list[i]
        img_filename = os.path.join(images_dir, dataset, img_name)
        try:
            input_img = cv2.imread(img_filename)
            img_shape = input_img.shape
            test_df = test_df.append({'FileName': img_filename, 'Label': label, 'ClassName': dataset},
                                           ignore_index=True)
        except:
            print('{} is corrupted\n'.format(img_filename))
            num_corrupted_files += 1

    print('Read {0} images out of {1} images from data dir {2}\n'.format(num_img_files - num_corrupted_files,
                                                                         num_img_files, dataset))

print('completed reading all the image files and assigned labels accordingly')


dest_path=os.path.join('D:\paper\Self_Supervised_Learning\codes\context','dataset','annotations')
if not os.path.exists(dest_path):
    os.mkdir(dest_path)

test_df.to_csv(os.path.join(dest_path,'cifar100_recognition_test.csv'))
print('The test csv files are saved')