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

class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')

    def update(self, labels, preds):
        rpn_label, rpn_weight = labels
        rpn_cls_logits = preds[0]

        num_inst = nd.sum(rpn_weight)

        pred_label = nd.sigmoid(rpn_cls_logits) >= 0.5

        num_acc = mx.nd.sum((pred_label == rpn_label) * rpn_weight)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += num_inst.asscalar()

class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric, self).__init__('RPNL1Loss')

    def update(self, labels, preds):
        rpn_bbox_target, rpn_bbox_weight = labels
        rpn_bbox_reg = preds[0]

        num_inst = nd.sum(rpn_bbox_weight) / 4

        loss = nd.sum(rpn_bbox_weight * nd.smooth_l1(rpn_bbox_reg-rpn_bbox_target, scalar=3))

        self.sum_metric += loss.asscalar()
        self.num_inst += num_inst.asscalar()

class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNAccMetric, self).__init__('RCNNAcc')

    def update(self, labels, preds):
        rcnn_label = labels[0]
        rcnn_cls = preds[0]

        pred_label = nd.argmax(rcnn_cls, axis=-1)
        num_acc = nd.sum(pred_label == rcnn_label)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += rcnn_label.size

class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')

    def updata(self, labels, preds):
        rcnn_bbox_taregt, rcnn_bbox_weight = labels
        rcnn_bbox_reg = preds[0]

        num_inst = nd.sum(rcnn_bbox_weight) / 4

        loss = nd.sum(rcnn_bbox_weight * nd.smooth_l1(rcnn_bbox_reg-rcnn_bbox_taregt, scalar=1))

        self.sum_metric += loss.asscalar()
        self.num_inst += num_inst.asscalar()