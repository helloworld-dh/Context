import torch
import torch.nn as nn
import torch.optim as optim
import utils.help
import utils.util
from models.models import AlexNetwork
import time
from tqdm import tqdm
from torchvision import models, datasets
from dataloaders.cifardataset import MyDataset
import skimage
from torchvision import transforms
import matplotlib.pyplot as plt
from models.resnet import ResNet

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

config_path = r'D:\paper\Self_Supervised_Learning\codes\context\config\config.yml'
Config = utils.help.load_yaml(config_path,config_type='object')
#############################################
# dataloader
#############################################
annotation_file = 'cifar100_recognition_train.csv'
traindataset = MyDataset(Config, Config.patch_dim, Config.gap, annotation_file, False,
                         transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])]))
trainloader = torch.utils.data.DataLoader(traindataset,
                                          batch_size=Config.batch_size,
                                          shuffle=True,
                                          # num_workers=Config.num_workers
                                          )
annotation_file = 'cifar100_recognition_val.csv'
valdataset = MyDataset(Config, Config.patch_dim, Config.gap, annotation_file, True,
                       transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])]))
valloader = torch.utils.data.DataLoader(valdataset,
                                        batch_size=Config.batch_size,
                                        shuffle=False)
annotation_file = 'cifar100_recognition_test.csv'
testdataset = MyDataset(Config, Config.patch_dim, Config.gap, annotation_file, False,
                         transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])]))
testloader = torch.utils.data.DataLoader(testdataset,
                                          batch_size=Config.batch_size,
                                          shuffle=True,
                                          # num_workers=Config.num_workers
                                          )

#############################################
# network
#############################################
# model = AlexNetwork().cpu()
model = ResNet(Config).cpu()

#############################################
# Initialized Optimizer, criterion, scheduler
#############################################
optimizer = optim.Adam(model.parameters(), lr=Config.lr)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                           mode='min',
                                           patience=5,
                                           factor=0.3, verbose=True)

############################
# Training/Validation Engine
############################

global_trn_loss = []
global_val_loss = []
# previous_val_loss = 100

for epoch in range(Config.num_epochs):
    train_running_loss = []
    val_running_loss = []
    start_time = time.time()
    total_train=0
    correct = 0
    model.train()
    for idx, data in tqdm(enumerate(trainloader), total=int(len(traindataset) / Config.batch_size)):
        uniform_patch, random_patch, random_patch_label = data[0].to(device), data[1].to(device), data[2].to(device)
        optimizer.zero_grad()
        output, output_fc6_uniform, output_fc6_random = model(uniform_patch, random_patch)
        loss = criterion(output, random_patch_label)

        _, predicted = torch.max(output.data, 1)
        total_train += random_patch_label.size(0)
        correct += (predicted == random_patch_label).sum()

        loss.backward()
        optimizer.step()

        train_running_loss.append(loss.item())
    else:
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for idx, data in tqdm(enumerate(valloader), total=int(len(valdataset) / Config.batch_size)):
                uniform_patch, random_patch, random_patch_label = data[0].to(device), data[1].to(device), data[2].to(
                    device)
                output, output_fc6_uniform, output_fc6_random = model(uniform_patch, random_patch)
                loss = criterion(output, random_patch_label)
                val_running_loss.append(loss.item())

                _, predicted = torch.max(output.data, 1)
                total += random_patch_label.size(0)
                correct += (predicted == random_patch_label).sum()
            print('Val Progress --- total:{}, correct:{}'.format(total, correct.item()))
            print('Val Accuracy of the network on the 10000 test images: {}%'.format(100 * correct / total))
    print('train Progress --- total:{}, correct:{}'.format(total_train, correct.item()))
    print('train Accuracy of the network on the 10000 test images: {}%'.format(100 * correct / total_train))

    global_trn_loss.append(sum(train_running_loss) / len(train_running_loss))
    global_val_loss.append(sum(val_running_loss) / len(val_running_loss))

    scheduler.step(global_val_loss[-1])

    print('Epoch [{}/{}], TRNLoss:{:.4f}, VALLoss:{:.4f}, Time:{:.2f}'.format(
        epoch + 1, Config.num_epochs, global_trn_loss[-1], global_val_loss[-1],
        (time.time() - start_time) / 60))

    if epoch % 20 == 0:
        MODEL_SAVE_PATH = f'D:/paper/Self_Supervised_Learning/codes/context/save_path/{Config.batch_size}_{Config.num_epochs}_{Config.lr}_{Config.subset_data}_{Config.patch_dim}_{Config.gap}.pt'
        torch.save(
            {
                'epoch': Config.num_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'global_trnloss': global_trn_loss,
                'global_valloss': global_val_loss
            }, MODEL_SAVE_PATH)

checkpoint = torch.load('D:/paper/Self_Supervised_Learning/codes/context/save_path/64_1_0.0005_100_10_1.pt', map_location='cuda')
plt.plot(range(len(checkpoint['global_trnloss'])), checkpoint['global_trnloss'], label='TRN Loss')
plt.plot(range(len(checkpoint['global_valloss'])), checkpoint['global_valloss'], label='VAL Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Main Network Training/Validation Loss plot')
plt.legend()
plt.show()
