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

class VGG16(nn.HybridBlock):
    # input size (b, 3, 224, 224)
    def __init__(self, num_classes=120, **kwargs):
        super(VGG16, self).__init__(**kwargs)
        model = gluon.model_zoo.vision.get_model('vgg16', pretrained=True)
        with self.name_scope():
            self.features = model.features
            self.output = nn.Dense(num_classes)

    def initialize(self, ctx=None):
        for param in self.collect_params().values():
            if param._data is not None:
                continue
            else:
                param.initialize()

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x

if __name__ == '__main__':
    m = VGG16()
    m.initialize()
    data = mx.nd.random.uniform(shape=(1, 3, 224, 224))
    out = m(data)
    print(out.shape)