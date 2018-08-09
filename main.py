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
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import logging
import time
import pickle
import dataloader
import models
import gluoncv
from mxnet import gluon
from mxnet import init
from mxnet import nd
from mxnet import autograd
from mxnet.gluon import nn
from metrics import RPNAccMetric, RPNL1LossMetric, RCNNAccMetric, RCNNL1LossMetric
from config import DefaultConfig
from utils.visualize import Visualizer
from matplotlib import pyplot as plt

def get_logger(opt):
    logging.basicConfig(format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = opt.log_file_path
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info('initialize logger')
    return logger

def initialize_model_gpu(model):
    # ignore pretrained network
    for param in model.collect_params().values():
        if param._data is not None:
            continue
        else:
            param.initialize()
    model.collect_params().reset_ctx(mx.gpu())

def initialize_model_cpu(model):
    # ignore pretrained network
    for param in model.collect_params().values():
        if param._data is not None:
            continue
        else:
            param.initialize()
    model.collect_params().reset_ctx(mx.cpu())

def shift_to_gpu(batch):
    # only support one gpu
    new_batch = []
    for data in batch:
        new_data = data[0].as_in_context(mx.gpu())
        new_batch.append(new_data)
    return new_batch

def save_params(model, logger, current_map, best_map):
    pass

def train(model_train, train_dataloader, val_dataloader, eval_metric, logger, opt):
    # visualization
    vis = Visualizer(opt.env)

    # preload
    if opt.preload:
        model_train.load_parameters(opt.load_file_path)

    # set train mode
    model_train.collect_params().setattr('grad_req', 'null')
    model_train.collect_train_params().setattr('grad_req', 'write')

    trainer = gluon.Trainer(model_train.collect_train_params(),
                            'sgd',
                            {'learning_rate': opt.lr,
                             'wd': opt.wd,
                             'momentum': opt.momentum,
                             'clip_gradient': 5})
    # lr decay
    lr_decay = float(opt.lr_decay)
    lr_steps = sorted([float(ls) for ls in opt.lr_decay_epoch.split(',') if ls.strip()])
    # lr_warmup = float(opt.lr_warmup)

    # 4 losses
    rpn_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    rpn_box_loss = mx.gluon.loss.HuberLoss(rho=1./9.)
    rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    rcnn_box_loss = mx.gluon.loss.HuberLoss()
    # merge
    metrics = [mx.metric.Loss('RPN_CE'),
               mx.metric.Loss('RPN_Box'),
               mx.metric.Loss('RCNN_CE'),
               mx.metric.Loss('RCNN_Box')]

    rpn_acc_metric = RPNAccMetric()
    rpn_bbox_metric = RPNL1LossMetric()
    rcnn_acc_metric = RCNNAccMetric()
    rcnn_bbox_metric = RCNNL1LossMetric()
    # merge
    metrics2 = [rpn_acc_metric, rpn_bbox_metric, rcnn_acc_metric, rcnn_bbox_metric]

    # train_dataloader, val_dataloader, val_metric = dataloader.DogDataLoader(model_train)

    logger.info('Starting training from Epoch {}'.format(opt.start_epoch+1))
    best_map = 0
    for epoch in range(opt.start_epoch, opt.max_epoch):
        # lr decay
        '''
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info('Epoch {}: set learning rate to {}'.format(epoch+1, new_lr))            
        '''

        # metric
        for metric in metrics:
            metric.reset()

        start_time = time.time()
        for i, batch in enumerate(train_dataloader):
            # adjust real training rate
            # if epoch == 0 and i <= lr_warmup:
            #     pass

            losses = []
            metric_loss = [[] for _ in metrics]
            add_losses = [[] for _ in metrics2]
            # training
            with autograd.record():
                data, label, rpn_cls_targets, rpn_box_targets, rpn_box_masks = shift_to_gpu(batch)
                gt_label = label[:, :, 4:5]
                gt_box = label[:, :, :4]
                # model
                cls_pred, box_pred, roi, samples, matches, rpn_score, rpn_box, anchors = model_train(data, gt_box)

                # --------------------rpn loss------------------------
                rpn_score = rpn_score.squeeze(axis=-1)
                num_rpn_pos = (rpn_cls_targets >= 0).sum()
                rpn_loss1 = rpn_cls_loss(rpn_score, rpn_cls_targets, rpn_cls_targets>=0) * rpn_cls_targets.size / num_rpn_pos
                rpn_loss2 = rpn_box_loss(rpn_box, rpn_box_targets, rpn_box_masks) * rpn_box.size / num_rpn_pos
                rpn_loss = rpn_loss1 + rpn_loss2
                # --------------------rcnn loss-----------------------
                cls_targets, box_targets, box_masks = model_train.target_generator(roi, samples, matches, gt_label, gt_box)
                num_rcnn_pos = (cls_targets >= 0).sum()
                rcnn_loss1 = rcnn_cls_loss(cls_pred, cls_targets, cls_targets >= 0) * cls_targets.size / cls_targets.shape[0] / num_rcnn_pos
                rcnn_loss2 = rcnn_box_loss(box_pred, box_targets, box_masks) * box_pred.size / box_pred.shape[0] / num_rcnn_pos
                rcnn_loss = rcnn_loss1 + rcnn_loss2
                # 
                losses.append(rpn_loss.sum() + rcnn_loss.sum())
                autograd.backward(losses)
                # metrics
                metric_loss[0].append(rpn_loss1.sum())
                metric_loss[1].append(rpn_loss2.sum())
                metric_loss[2].append(rcnn_loss1.sum())
                metric_loss[3].append(rcnn_loss2.sum())
                add_losses[0].append([[rpn_cls_targets, rpn_cls_targets>=0], [rpn_score]])
                add_losses[1].append([[rpn_box_targets, rpn_box_masks], [rpn_box]])
                add_losses[2].append([[cls_targets], [cls_pred]])
                add_losses[3].append([[box_targets, box_masks], [box_pred]])

                for metric_, record_ in zip(metrics, metric_loss):
                    metric.update(0, record_)
                for metric_, record_ in zip(metrics2, add_losses):
                    for pred in record_:
                        metric.update(pred[0], pred[1])

                # visualization
                if (i+1) % opt.log_interval == 0:
                    # loss, rpn_loss1, rpn_loss2, rcnn_loss1, rcnn_loss2
                    rpn_loss1_ = rpn_loss1.sum().asscalar()
                    rpn_loss2_ = rpn_loss2.sum().asscalar()
                    rcnn_loss1_ = rcnn_loss1.sum().asscalar()
                    rcnn_loss2_ = rcnn_loss2.sum().asscalar()
                    vis.plot('total_loss', rpn_loss1_+rpn_loss2_+rcnn_loss1_+rcnn_loss2_)
                    vis.plot('rpn_loss', rpn_loss1_+rpn_loss2_)
                    vis.plot('rcnn_loss', rcnn_loss1_+rcnn_loss2_)
                    logger.info('[Epoch {}] [Batch {}]: total_loss: {}, rpn_loss: {}, rcnn_loss: {}'.format(
                                                    epoch+1, i+1, losses[-1].asscalar(), rpn_loss1_+rpn_loss2_, rcnn_loss1_+rcnn_loss2_))

            trainer.step(opt.batch_size)
        # epoch finish
        logger.info('[Epoch {} finish]: total {} batches, use {} seconds, speed: {:.1f} s/batch'.format(epoch+1, i+1, time.time()-start_time,
                                                                                        float((time.time()-start_time)/(i+1))))
        # validate
        if not (epoch+1) % opt.val_interval:
            map_name, mean_ap = validate(model_train, val_dataloader, val_metric)
            current_map = float(sum(mean_ap)/len(mean_ap))
            vis.plot('map', current_map)
            logger.info('[Epoch {}] Validation: MAp {}'.format(epoch, current_map))
        else:
            current_map = 0

        # save params
        if current_map > best_map:
            best_map = current_map
            model_train.save_parameters(opt.save_path+'epoch{}_map_{:.4f}.params'.format(epoch+1, current_map))
            logger.info('[Epcoh {}] map: {}, save parameters!!!'.format(epoch+1, current_map))
        else:
            # learnig rate decay
            new_lr = trainer.learning_rate * lr_decay
            trainer.set_learning_rate(new_lr)
            logger.info('Epoch {}: set learning rate to {}'.format(epoch+1, new_lr))

def validate(model, val_dataloader, val_metric):
    # 
    clipper = gluoncv.nn.bbox.BBoxClipToImage()
    val_metric.reset()
    for batch in val_dataloader:
        x, y, im_scale = shift_to_gpu(batch)
        ids, scores, bboxes = model(x)
        det_ids = ids
        det_scores =scores
        det_bboxes = clipper(bboxes, x)

        im_scale = im_scale.reshape((-1)).asscalar()
        det_bboxes *= im_scale

        gt_ids = y.slice_axis(axis=-1, begin=4, end=5)
        gt_bboxes = y.slice_axis(axis=-1, begin=0, end=4)
        gt_bboxes *= im_scale
        gt_difficulty = y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None

        val_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficulty)

    return val_metric.get()

if __name__ == '__main__':
    # model_train = models.vgg16_faster_rcnn()
    # initialize_model(model_train)
    # model_train.initialize(ctx=mx.cpu())
    # rain_dataloader, val_dataloader, val_metric = dataloader.VOCDataLoader(model_train)
    # map_name, mean_ap = validate(model_train, val_dataloader, val_metric)
    # print(map_name, mean_ap)
    # data = mx.nd.random.uniform(shape=(1, 3, 999, 999))
    # ids, scores, bboxes = model_train(data)
    # print(ids.shape, scores.shape, bboxes.shape)
    # im_name = 'person.jpg'
    # x, orig_img = gluoncv.data.transforms.presets.rcnn.load_test(im_name)
    # box_ids, scores, bboxes = model_train(x)
    # print(bboxes)
    # ax = gluoncv.utils.viz.plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=model_train.classes, thresh=0.1)

    # plt.show()
    # for param in model_train.collect_params().values():
        # print(param)
        # print(param._data)
    #     # break
    # for param in model_train.collect_train_params():
    #     print(param)
    opt = DefaultConfig()
    opt.parse({'model': 'vgg16_faster_rcnn',
               'env': 'vgg16',
               'lr_decay_epoch': '8, 16',
               'preload': False,
               'special_load': True,
               'lr': 0.001,
               'start_epoch': 0,
               'max_epoch': 20,
               'load_file_path': '',
               'log_file_path': './log/vgg16_faster_rcnn.log'})

    logger = get_logger(opt)
    if opt.special_load and opt.special_load_path is not None:
        model_train = getattr(models, opt.model)(True, opt.special_load_path)
    else:
        model_train = getattr(models, opt.model)()
    initialize_model_gpu(model_train)
    

    
    train_dataloader, val_dataloader, val_metric = dataloader.DogDataLoader(model_train)
    train(model_train, train_dataloader, val_dataloader, val_metric, logger, opt)
    '''
    data = mx.nd.random.uniform(shape=(1, 3, 999, 999))
    ids, scores, bbox = model_train(data)
    for param in model_train.collect_params().values():
        print(param._data)
        break
    model_train.save_parameters('tmp.params')
    '''