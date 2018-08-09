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
import xml.etree.ElementTree as ET
import gluoncv
from matplotlib import pyplot as plt
from mxnet import gluon
from mxnet import init
from mxnet import nd
from mxnet import autograd
from mxnet.gluon import nn
from gluoncv.data.base import VisionDataset
from gluoncv.data import batchify
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultTrainTransform, FasterRCNNDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric

class DogDetection(VisionDataset):
    # 
    # with open('class_name', 'rb') as f:
    CLASSES = ('dog',)

    def __init__(self, root='./stanford_dog_dataset', splits='train',
                 transform=None, index_map=None, preload_label=False):
        super(DogDetection, self).__init__(root)
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        self._transform = transform
        self._splits = splits
        self._items = self._load_items()
        # print(self.__len__())
        self._anno_path = os.path.join('{}', 'Annotations', '{}')
        self._image_path = os.path.join('{}', 'Images', '{}.jpg')
        self.num_classes = len(self.classes)
        self.index_map = index_map or dict(zip(self.classes, range(self.num_classes)))
        # self._check_label()
        self._label_cache = self._preload_labels() if preload_label else None

    @property
    def classes(self):
        return type(self).CLASSES

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self._image_path.format(*img_id)
        label = self._label_cache[idx] if self._label_cache else self._load_label(idx)
        img = mx.image.imread(img_path, 1)
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def _load_items(self):
        # [(root, file_idx)]
        ids = []
        lf = os.path.join(self._root, self._splits+'.txt')
        with open(lf, 'r') as f:
            ids += [(self._root, line.strip()) for line in f.readlines()]
        return ids

    def _check_label(self):
        max_width = 0
        max_height = 0
        for idx in range(len(self)):
            img_id = self._items[idx]
            anno_path = self._anno_path.format(*img_id)
            root = ET.parse(anno_path).getroot()
            size = root.find('size')
            width = float(size.find('width').text)
            height = float(size.find('height').text)
            if width > 2000:
                print(anno_path)
            if width > max_width:
                max_width = width
            if height > max_height:
                max_height = height
            for obj in root.iter('object'):
                cls_name = obj.find('name').text.strip()
            if cls_name not in self.classes:
                print(anno_path)
            xml_box = obj.find('bndbox')
            xmin = float(xml_box.find('xmin').text)
            ymin = float(xml_box.find('ymin').text)
            xmax = float(xml_box.find('xmax').text)
            ymax = float(xml_box.find('ymax').text)
            if xmin > width or ymin > height or xmax > width or ymax > height:
                print(anno_path)
        print('max width', max_width, 'max height', max_height)

    def _load_label(self, idx):
        img_id = self._items[idx]
        anno_path = self._anno_path.format(*img_id)
        root = ET.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if idx not in self._im_shapes:
            self._im_shapes[idx] = (width, height)
        label = []
        for obj in root.iter('object'):
            difficult = int(obj.find('difficult').text)
            # cls_name = obj.find('name').text.strip()
            cls_name = 'dog'.strip()
            if cls_name not in self.classes:
                continue
            cls_id = self.index_map[cls_name]
            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text))
            ymin = (float(xml_box.find('ymin').text))
            xmax = (float(xml_box.find('xmax').text))
            ymax = (float(xml_box.find('ymax').text))
            try:
                self._validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
            label.append([xmin, ymin, xmax, ymax, cls_id, difficult])
        return np.array(label)

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        assert xmin >= 0 and xmin < width, (
            "xmin must in [0, {}), given {}".format(width, xmin))
        assert ymin >= 0 and ymin < height, (
            "ymin must in [0, {}), given {}".format(height, ymin))
        assert xmax > xmin and xmax <= width, (
            "xmax must in (xmin, {}], given {}".format(width, xmax))
        assert ymax > ymin and ymax <= height, (
            "ymax must in (ymin, {}], given {}".format(height, ymax))

    def _preload_labels(self):
        return [self._load_label(idx) for idx in range(len(self))]

def DogDataLoader(net, root='./stanford_dog_dataset',
                  preload_label=True, batch_size=1, shuffle=True, num_workers=0):
    # dataset
    train_dataset = DogDetection(root=root, splits='train', preload_label=preload_label)
    val_dataset = DogDetection(root=root, splits='test', preload_label=preload_label)

    val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)

    train_batchify = batchify.Tuple(*[batchify.Append() for _ in range(5)])
    val_batchify = batchify.Tuple(*[batchify.Append() for _ in range(3)])

    train_dataloader = gluon.data.DataLoader(train_dataset.transform(FasterRCNNDefaultTrainTransform(net.short, net.max_size, net)),
                                             batch_size=batch_size,
                                             shuffle=True,
                                             batchify_fn=train_batchify,
                                             last_batch='rollover',
                                             num_workers=num_workers)
    val_dataloader = gluon.data.DataLoader(val_dataset.transform(FasterRCNNDefaultValTransform(net.short, net.max_size)),
                                           batch_size=batch_size,
                                           shuffle=False,
                                           batchify_fn=val_batchify,
                                           last_batch='keep',
                                           num_workers=num_workers)

    return train_dataloader, val_dataloader, val_metric


if __name__ == '__main__':
    t = DogDetection()
    print(t.index_map)
    print(t.classes)
    print(t.num_classes)
    image, label = t[1]
    print(image.shape, label.shape)
    bbox = label[:, :4]
    ids = label[:, 4:5]
    gluoncv.utils.viz.plot_bbox(image.asnumpy(), bbox, labels=ids, class_names=t.classes)
    plt.show()