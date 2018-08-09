import pickle
from .base_network import *
from .faster_rcnn import FasterRCNN
from .vgg import *
from mxnet.gluon.model_zoo.vision import get_model

def vgg16_faster_rcnn(special_pretrain=False, load_path=None):
    if special_pretrain and load_path is not None:
        vgg = VGG16()
        vgg.load_parameters(load_path)
    else:
        vgg = get_model('vgg16', pretrained=True)
    # with open('class_name', 'rb') as f:
    #     classes = pickle.load(f)
    classes = ('dog',)
    features = VGG16_features(vgg)
    top_features_ = top_features()
    model = FasterRCNN(features=features, stride=16, top_features=top_features_, classes=classes, short=600, max_size=1000)
    return model

def resnet50_faster_rcnn(special_pretrain=False, load_path=None):
    if special_pretrain and load_path is not None:
        resnet = get_model('resnet50_v2', pretrained=False)
        resnet.load_parameters(load_path)
    else:
        resnet = get_model('resnet50_v2', pretrained=True)
    # with open('class_name', 'rb') as f:
    #     classes = pickle.load(f)
    classes = ('dog',)
    features = ResNet50_v2_features(resnet)
    top_features_ = top_features()
    model = FasterRCNN(features=features, stride=32, top_features=top_features_, classes=classes, short=600, max_size=1000)
    return model

