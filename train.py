import time
import os
import logging
from tqdm import tqdm

from utils import unet_dataset
from models import unet
from metrics import eval_metrics

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train(config):

    # train配置
    device = torch.device('cuda:0')

    model = unet.UNet(num_classes=config['num_classes'])
    
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)

    logger = initLogger("unet")

    # loss
    criterion = nn.CrossEntropyLoss()

    # train data
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),

            transforms.ToTensor()
            #输入图像是单通道
            #transforms.Normalize((0.5, ), (0.5, ))
        ]
    )
    dst_train = unet_dataset.UnetDataset(config['train_list'], transform=transform)
    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=config['batch_size'])

    # validation data
    transform = transforms.Compose(
        [
            #transforms.ToPILImage(),
            transforms.ToTensor()
            #输入图像是单通道
            #transforms.Normalize((0.5, ), (0.5, ))
        ]
    )
    dst_valid = unet_dataset.UnetDataset(config['test_list'], transform=transform)
    dataloader_valid = DataLoader(dst_valid, shuffle=False, batch_size=config['batch_size'])

    cur_acc = []
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=[config['momentum'], 0.999], weight_decay=config['weight_decay'])
    for epoch in range(config['num_epoch']):
        epoch_start = time.time()
        # lr
        
        model.train()
        loss_sum = 0.0
        correct_sum = 0.0
        labeled_sum = 0.0
        inter_sum = 0.0
        unoin_sum = 0.0
        pixelAcc = 0.0
        IoU = 0.0
        tbar = tqdm(dataloader_train, ncols=100)
        for batch_idx, (data, target) in enumerate(tbar):
            tic = time.time()

            # data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()

            correct, labeled, inter, unoin = eval_metrics(output, target, config['num_classes'])
            correct_sum += correct
            labeled_sum += labeled
            inter_sum += inter
            unoin_sum += unoin
            pixelAcc = 1.0 * correct_sum / (np.spacing(1)+labeled_sum)
            IoU = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)
            tbar.set_description('TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.4f} | bt {:.2f} et {:.2f}|'.format(
                epoch, loss_sum/((batch_idx+1)*config['batch_size']),
                pixelAcc, IoU.mean(),
                time.time()-tic, time.time()-epoch_start))
            cur_acc.append(pixelAcc)

        logger.info('TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} IOU {}  mIoU {:.4f} '.format(
            epoch, loss_sum / ((batch_idx + 1) * config['batch_size']),
            pixelAcc, toString(IoU), IoU.mean()))
            

        # val
        test_start = time.time()
        max_pixACC = 0.0
        model.eval()
        loss_sum = 0.0
        correct_sum = 0.0
        labeled_sum = 0.0
        inter_sum = 0.0
        unoin_sum = 0.0
        pixelAcc = 0.0
        mIoU = 0.0
        tbar = tqdm(dataloader_valid, ncols=100)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tbar):
                tic = time.time()

                # data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                loss_sum += loss.item()

                correct, labeled, inter, unoin = eval_metrics(output, target, config['num_classes'])
                correct_sum += correct
                labeled_sum += labeled
                inter_sum += inter
                unoin_sum += unoin
                pixelAcc = 1.0 * correct_sum / (np.spacing(1) + labeled_sum)
                mIoU = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)
                tbar.set_description('VAL ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.4f} | bt {:.2f} et {:.2f}|'.format(
                    epoch, loss_sum / ((batch_idx + 1) * config['batch_size']),
                    pixelAcc, mIoU.mean(),
                            time.time() - tic, time.time() - test_start))
            if pixelAcc > max_pixACC:
                max_pixACC = pixelAcc
                torch.save(model.state_dict(), os.path.join(config['save_model']['save_path'], 'unet.pth'))
        logger.info('VAL ({}) | Loss: {:.3f} | Acc {:.2f} IOU {} mIoU {:.4f} |'.format(
            epoch, loss_sum / ((batch_idx + 1) * config['batch_size']),
            pixelAcc, toString(mIoU), mIoU.mean()))

def toString(IOU):
    result = '{'
    for i, num in enumerate(IOU):
        result += str(i) + ': ' + '{:.4f}, '.format(num)

    result += '}'
    return result

def initLogger(model_name):
    # 初始化log
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = r'logs'
    log_name = os.path.join(log_path, model_name + '_' + rq + '.log')
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
