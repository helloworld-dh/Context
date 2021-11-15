import utils
from tqdm import tqdm
from itertools import islice
import torch
import utils.metrics
import os
import dataloaders
import models
import torch.nn as nn

def validate(epoch, model, device, dataloader, criterion):
    """ Test loop, print metrics """
    # progbar = tqdm(total=len(dataloader), desc='Val')


    loss_record = utils.metrics.RunningAverage()
    acc_record = utils.metrics.RunningAverage()
    model.eval()
    with torch.no_grad():
        #    for batch_idx, (data,label,_,_) in enumerate(tqdm(islice(dataloader,10))):
        # for batch_idx, (data, label ,_ ,_) in enumerate(tqdm(dataloader)):
        for idx, data in enumerate(tqdm(dataloader)):
            data, label = data
            output = model(data)
            loss = criterion(output, label)

            # measure accuracy and record loss
            acc = utils.metrics.compute_acc(output, label)
            #        acc_record.update(100 * acc[0].item())
            acc_record.update(100 *acc[0].item( ) /data.size(0))
            loss_record.update(loss.item())
            # print('val Step: {}/{} Loss: {:.4f} \t Acc: {:.4f}'.format(batch_idx,len(dataloader), loss_record(), acc_record()))
            # progbar.set_description('Val (loss=%.4f)' % (loss_record()))
            # progbar.update(1)

    # writer.add_scalar('validation/Loss_epoch', loss_record(), epoch)
    # writer.add_scalar('validation/Acc_epoch', acc_record(), epoch)
    print("val acc: ", acc_record())

    return loss_record() ,acc_record()

def test( model, device, dataloader, criterion):
    """ Test loop, print metrics """
    loss_record = utils.metrics.RunningAverage()
    acc_record = utils.metrics.RunningAverage()
    model.eval()
    with torch.no_grad():
        #   for batch_idx, (data,label,_,_) in enumerate(tqdm(islice(dataloader,10))):
        for batch_idx, data in enumerate(tqdm(dataloader)):
            data, label = data
            output = model(data)
            loss = criterion(output, label)

            # measure accuracy and record loss
            acc = utils.metrics.compute_acc(output, label)
            #        acc_record.update(100 * acc[0].item())
            acc_record.update(100 *acc[0].item( ) /data.size(0))
            loss_record.update(loss.item())
    #            print('Test Step: {}/{} Loss: {:.4f} \t Acc: {:.4f}'.format(batch_idx,len(dataloader), loss_record(), acc_record()))
    print("test acc:", acc_record())
    return loss_record() ,acc_record()