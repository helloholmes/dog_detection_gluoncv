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
import logging
import models
import dataloader
from mxnet import gluon
from mxnet import init
from mxnet import nd
from mxnet import autograd
from mxnet.gluon import nn
from config import DefaultConfig
from utils.visualize import Visualizer

def get_logger(opt):
    logging.basicConfig(format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = opt.log_file_path
    fh = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info('initialize logger')
    return logger

def convert_model_gpu(model):
    # model.initialize()
    model.collect_params().reset_ctx(mx.gpu())

def convert_model_cpu(model):
    model.collect_params().reset_ctx(mx.cpu())

def train(model_train, train_dataloader, val_dataloader, logger, opt):
    # visualization
    vis = Visualizer(opt.env)

    # preload
    if opt.preload:
        model_train.load_parameters(opt.load_file_path)

    # set train mode
    model_train.collect_params().setattr('grad_req', 'write')
    # model_train.collect_train_params().setattr('grad_req', 'write')

    trainer = gluon.Trainer(model_train.collect_params(),
                            'sgd',
                            {'learning_rate': opt.lr,
                             'wd': opt.wd,
                             'momentum': opt.momentum,
                             'clip_gradient': 5})

    # lr decay
    lr_decay = float(opt.lr_decay)

    # train_loss
    train_loss = gluon.loss.SoftmaxCrossEntropyLoss()

    logger.info('Starting training from Epoch {}'.format(opt.start_epoch+1))
    best_acc = 0
    for epoch in range(opt.start_epoch, opt.max_epoch):
        start_time = time.time()
        loss_his = []

        for i, (data, label) in enumerate(train_dataloader):
            data = data.as_in_context(opt.ctx)
            label = label.astype('float32').as_in_context(opt.ctx)
            with autograd.record():
                output = model_train(data)
                loss = train_loss(output, label)
            autograd.backward(loss)
            trainer.step(opt.batch_size)

            loss_ = loss.sum().asscalar()
            if loss_ < 1e5:
                loss_his.append(loss_)

            if loss_ < 1e5 and (i+1) % opt.log_interval == 0:
                logger.info('[Epoch {}] [Batch {}]: train_loss: {:.5f}'.format(epoch+1, i+1, float(loss_/opt.batch_size)))
                vis.plot('train_loss', float(loss_/opt.batch_size))

        # epoch finish
        logger.info('[Epoch {} finishes]: total {} batches, use {:.3f} seconds, speed: {:.3f} s/batch'.format(
                                    epoch+1, i+1, time.time()-start_time, float((time.time()-start_time)/(i+1))))

        vis.plot('train_epoch_loss', float(sum(loss_his)/len(loss_his)/opt.batch_size))

        # validate
        if not (epoch+1) % opt.val_interval:
            val_acc, val_loss = validate(model_train, val_dataloader, opt)
            current_acc = val_acc
            vis.plot('val_acc', val_acc)
            # TODO
            vis.plot('val_loss', val_loss)
            logger.info('[Epoch {}] Validation: predict accuracy {:.2f}'.format(epoch+1, current_acc))
        else:
            current_acc = 0

        # save params
        if current_acc > best_acc:
            best_acc = current_acc
            model_train.save_parameters(opt.save_path+'epoch{}_acc_{:.2f}.params'.format(epoch+1, current_acc))
            logger.info('[Epoch {}] acc: {}, save parameters!!!'.format(epoch+1, current_acc))
        else:
            # learning rate decay
            new_lr = trainer.learning_rate * lr_decay
            trainer.set_learning_rate(new_lr)
            logger.info('[Epoch {}]: set learing rate to {}'.format(epoch+1, new_lr))


def validate(model, val_dataloader, opt):
    total_num = 0
    correct_num = 0
    val_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    val_loss_his = []
    for i, (data, label) in enumerate(val_dataloader):
        output = model(data.as_in_context(opt.ctx))
        output = output.as_in_context(mx.cpu())
        loss = val_loss(output, label)
        val_loss_his.append(loss.sum().asscalar())
        pred = output.argmax(axis=1).astype('int').asnumpy()
        label = label.astype('int').asnumpy()

        total_num += label.shape[0]
        correct_num += (label == pred).sum()
    # print('total correct num: ', total_num)
    val_acc = 100 * float(correct_num) / float(total_num)
    val_mean_loss = float(sum(val_loss_his)/len(val_loss_his)/opt.batch_size)
    return val_acc, val_mean_loss

if __name__ == '__main__':
    opt = DefaultConfig()
    opt.parse({'model': 'VGG16',
               'env': 'VGG16',
               'lr': 0.001,
               'train_dir': '/home/qinliang/dataset/stanford_dog_dataset/cut_images_train',
               'valid_dir': '/home/qinliang/dataset/stanford_dog_dataset/cut_images_val',
               'save_path': './cut_image_checkpoints/',
               'lr_decay': 0.5,
               'preload': True,
               'start_epoch': 0,
               'max_epoch': 50,
               'batch_size': 32,
               'wd': 15e-4,
               'load_file_path': '/home/qinliang/Desktop/kaggle/dog_recognition_gluon/checkpoints/epoch16_acc_99.31.params',
               'log_file_path': './log/VGG16_cut_image.log'})
    logger = get_logger(opt)

    model_train = getattr(models, opt.model)()
    model_train.initialize()
    convert_model_gpu(model_train)
    model_train.hybridize()

    train_dataloader, val_dataloader = dataloader.DogDataLoader(opt)

    train(model_train, train_dataloader, val_dataloader, logger, opt)


