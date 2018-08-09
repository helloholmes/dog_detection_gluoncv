# coding:utf-8
'''
python 3.5
mxnet 1.3.0
gluoncv 0.3.0
visdom 0.1.7
gluonbook 0.6.9
auther: helloholmes
'''
import mxnet as mx
import numpy as np
import os
import time
import pickle
from mxnet import gluon
from mxnet import init
from mxnet import nd
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon.data import vision
from mxnet.gluon.data.vision import transforms as T

def DogDataLoader(opt):
    transform_train = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.RandomFlipLeftRight(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])])

    transform_valid = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])])

    train_set = vision.ImageFolderDataset(opt.train_dir, flag=1)
    valid_set = vision.ImageFolderDataset(opt.valid_dir, flag=1)

    loader = gluon.data.DataLoader

    train_loader = loader(train_set.transform_first(transform_train),
                          batch_size=opt.batch_size,
                          shuffle=True,
                          num_workers=opt.num_workers,
                          last_batch='rollover')

    valid_loader = loader(valid_set.transform_first(transform_valid),
                          batch_size=opt.batch_size,
                          shuffle=False,
                          num_workers=opt.num_workers,
                          last_batch='keep')

    return train_loader, valid_loader