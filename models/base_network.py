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
from mxnet.gluon.model_zoo.vision import get_model

class VGG16_features(nn.HybridBlock):
    def __init__(self, base_model):
        super(VGG16_features, self).__init__()
        self.features = nn.HybridSequential()
        for layer in base_model.features[:-5]:
            self.features.add(layer)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        return x

class top_features(nn.HybridBlock):
    def __init__(self):
        super(top_features, self).__init__()
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(1024, 3, 1, 1, use_bias=False))
        self.features.add(nn.BatchNorm(epsilon=2e-5, use_global_stats=True))
        self.features.add(nn.Activation('relu'))
        self.features.add(nn.MaxPool2D(2, 2))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        return x
                          
class ResNet50_v2_features(nn.HybridBlock):
    def __init__(self, base_model):
        super(ResNet50_v2_features, self).__init__()
        self.features = nn.HybridSequential()
        for layer in base_model.features[:-4]:
            self.features.add(layer)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        return x

if __name__ == '__main__':
    # vgg16 = get_model('vgg16')
    # m = VGG16_features(vgg16)
    resnet = get_model('resnet50_v2')
    m = ResNet50_v2_features(resnet)
    m.initialize()
    data = mx.nd.random.uniform(shape=(1, 3, 224, 224))
    out = m(data)
    print(out.shape)