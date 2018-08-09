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
import gluoncv
from mxnet import gluon
from mxnet import init
from mxnet import nd
from mxnet import autograd
from mxnet.gluon import nn
from gluoncv.nn.coder import MultiPerClassDecoder, NormalizedBoxCenterDecoder
from gluoncv.nn.bbox import BBoxCornerToCenter
from gluoncv.model_zoo.faster_rcnn.rcnn_target import RCNNTargetGenerator, RCNNTargetSampler
from gluoncv.model_zoo.rpn.rpn import RPN


class FasterRCNN(nn.HybridBlock):
    def __init__(self, features, top_features, classes, short, max_size, train_patterns=None,
                nms_thresh=0.3, nms_topk=400, post_nms=100, roi_mode='align', roi_size=(14, 14), stride=16, clip=None,
                rpn_channel=1024, base_size=16, scales=(0.5, 1, 2), ratios=(8, 16, 32), alloc_size=(128, 128), rpn_nms_thresh=0.5,
                rpn_train_pre_nms=12000, rpn_train_post_nms=2000, rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=16,
                num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25):
        super(FasterRCNN, self).__init__()
        self.classes = classes
        self.num_classes = len(classes)
        self.short = short
        self.max_size = max_size
        self.train_patterns = train_patterns
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

        self._max_batch = 1
        self._num_sample = num_sample
        self._rpn_test_post_nms = rpn_test_post_nms
        # return cls_target, box_target, box_mask
        self._target_generater = {RCNNTargetGenerator(self.num_classes)}

        self._roi_mode = roi_mode.lower()
        self._roi_size = roi_size
        self._stride = stride

        with self.name_scope():
            self.features = features
            self.top_features = top_features
            self.global_avg_pool = nn.GlobalAvgPool2D()
            self.class_predictor = nn.Dense(
                self.num_classes+1, weight_initializer=mx.init.Normal(0.01))
            self.box_predictor = nn.Dense(
                self.num_classes*4, weight_initializer=mx.init.Normal(0.01))

            # reconstruct valid labels
            self.cls_decoder = MultiPerClassDecoder(num_class=self.num_classes+1)
            # (xmin, ymin, xmax, ymax) -> (x, y, h, w)
            self.box_to_center = BBoxCornerToCenter()
            # reconstructed bounding boxes
            self.box_decoder = NormalizedBoxCenterDecoder(clip=clip)
            # 
            self.rpn = RPN(
                channels=rpn_channel, stride=stride, base_size=base_size,
                scales=scales, ratios=ratios, alloc_size=alloc_size,
                clip=clip, nms_thresh=rpn_nms_thresh, train_pre_nms=rpn_train_pre_nms,
                train_post_nms=rpn_train_post_nms, test_pre_nms=rpn_test_pre_nms, test_post_nms=rpn_test_post_nms, min_size=rpn_min_size)

            self.sampler = RCNNTargetSampler(
                num_image=self._max_batch, num_proposal=rpn_train_post_nms,
                num_sample=num_sample, pos_iou_thresh=pos_iou_thresh, pos_ratio=pos_ratio)

    def collect_train_params(self, select=None):
        if select is None:
            return self.collect_params(self.train_patterns)
        return self.collect_params(select)

    def set_nms(self, nms_thresh=0.3, nms_topk=400, post_nms=100):
        self._clear_cached_op()
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

    @property
    def target_generator(self):
        return list(self._target_generater)[0]

    def hybrid_forward(self, F, x, gt_box=None):
        """
        gt_box is required when training
        """
        def _split(x, axis, num_outputs, squeeze_axis):
            """
            return a list of length of num_outputs
            """
            x = F.split(x, axis=axis, num_outputs=num_outputs, squeeze_axis=squeeze_axis)
            if isinstance(x, list):
                return x
            else:
                return [x]

        # ------------------------------------------------
        feature = self.features(x)
        # RPN proposals
        if autograd.is_training():
            _, rpn_box, raw_rpn_score, raw_rpn_box, anchors = self.rpn(feature, F.zeros_like(x))
            rpn_box, samples, matches = self.sampler(rpn_box, gt_box)
        else:
            _, rpn_box = self.rpn(feature, F.zeros_like(x))

        # generate rpn_roi (rpn_test_post_nms, 5) or (num_sample, 5) when training
        num_roi = self._num_sample if autograd.is_training() else self._rpn_test_post_nms
        with autograd.pause():
            roi_batchid = F.arange(0, self._max_batch, repeat=num_roi)
            rpn_roi = F.concat(*[roi_batchid.reshape((-1, 1)), rpn_box.reshape((-1, 4))], dim=-1)

        # ROIPooling
        if self._roi_mode == 'pool':
            pooled_feature = F.ROIPooling(feature, rpn_roi, self._roi_size, 1./self._stride)
        elif self._roi_mode == 'align':
            pooled_feature = F.contrib.ROIAlign(feature, rpn_roi, self._roi_size, 1./self._stride, sample_ratio=2)
        else:
            raise ValueError("roi mode {} unmatch".format(self._roi_mode))

        # prediction
        top_feature = self.top_features(pooled_feature)
        # (b*num_roi, outchannel, 1, 1)
        top_feature = self.global_avg_pool(top_feature)
        # (b*num_roi, num_classes+1)
        cls_pred = self.class_predictor(top_feature)
        # (b*num_roi, num_classes*4)
        box_pred = self.box_predictor(top_feature)

        # reshape
        # (b*num_roi, num_classes+1) -> (b, num_roi, num_classes+1)
        cls_pred = cls_pred.reshape((self._max_batch, num_roi, self.num_classes+1))
        # (b*num_roi, num_classes*4) -> (b, num_roi, num_classes, 4)
        box_pred = box_pred.reshape((self._max_batch, num_roi, self.num_classes, 4))

        # training target return
        if autograd.is_training():
            return (cls_pred, box_pred, rpn_box, samples, matches, raw_rpn_score, raw_rpn_box, anchors)

        # -----------------------decode-------------------------
        # (b, num_roi, num_classes)
        cls_ids, scores = self.cls_decoder(F.softmax(cls_pred, axis=-1))
        # (b, num_classes, num_roi, 1)
        cls_ids = cls_ids.transpose((0, 2, 1)).reshape((0, 0, 0, 1))
        scores = scores.transpose((0, 2, 1)).reshape((0, 0, 0, 1))
        # (b, num_classes, num_roi, 4)
        box_pred = box_pred.transpose((0, 2, 1, 3))

        # get into list
        # b * (1, num_roi, 4)
        rpn_boxes = _split(rpn_box, 0, self._max_batch, False)
        # b * (num_classes, num_roi, 1)
        cls_ids = _split(cls_ids, 0, self._max_batch, True)
        scores = _split(scores, 0, self._max_batch, True)
        # b * (num_classes, num_roi, 4)
        box_preds = _split(box_pred, 0, self._max_batch, True)

        # return result by batch
        results = []
        for rpn_box, cls_id, score, box_pred in zip(rpn_boxes, cls_ids, scores, box_preds):
            # 
            bbox = self.box_decoder(box_pred, self.box_to_center(rpn_box))
            # 
            res = F.concat(*[cls_id, score, bbox], dim=-1)
            # 
            res = F.contrib.box_nms(
                res, overlap_thresh=self.nms_thresh, topk=self.nms_topk, valid_thresh=0.0001,
                id_index=0, score_index=1, coord_start=2, force_suppress=True)
            # 
            res = res.reshape((-3, 0))
            results.append(res)

        # b * (num_classes*nms_topk, 6) -> (b, num_classes*nms_topk, 6)
        result = F.stack(*results, axis=0)
        ids = F.slice_axis(result, axis=-1, begin=0, end=1)
        scores = F.slice_axis(result, axis=-1, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=-1, begin=2, end=6)
        return ids, scores, bboxes



