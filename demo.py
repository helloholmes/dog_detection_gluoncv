from matplotlib import pyplot as plt
import gluoncv
import mxnet as mx
import models
import random
from mxnet.gluon.data import vision
from mxnet.gluon.data.vision import transforms as T

def transform_test(image):
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])])
    return transform(image)

def plot_bbox(img, bboxes, scores=None, labels=None, class_names=None, colors=None, thresh=0.5):
    ax = gluoncv.utils.viz.plot_image(img, ax=None, reverse_rgb=False)
    if len(bboxes) < 1:
        return ax

    if colors is None:
        colors = dict()

    for i, bbox in enumerate(bboxes):
        cls_id = int(labels[i])
        score_ = scores[i]
        if score_ > thresh:
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            xmin, ymin, xmax, ymax = [int(x) for x in bbox]
            rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor=colors[cls_id], linewidth=3.5)

            ax.add_patch(rect)

            score = '{:.3f}'.format(scores[i])
            class_name = class_names[cls_id]

            ax.text(xmin, ymin-2, '{:s} {:s}'.format(class_name, score), bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                    fontsize=12, color='white')

    return ax

def extend_image(height, width, bbox, ratio=0.05):
    xmin = int(bbox[0])
    ymin = int(bbox[1])
    xmax = int(bbox[2])
    ymax = int(bbox[3])
    if xmin > (ratio*width):
        xmin = int(xmin - ratio*width)
    else:
        xmin = 0

    if ymin > (ratio*height):
        ymin = int(ymin - ratio*height)
    else:
        ymin = 0

    if xmax < ((1-ratio)*width):
        xmax = int(xmax + ratio*width)
    else:
        xmax = width

    if ymax < ((1-ratio)*height):
        ymax = int(ymax + ratio*height)
    else:
        ymax = height
    return xmin, ymin, xmax, ymax

def plot(orig_image, model, box_ids, scores, bboxes, class_names, thresh=0.5):
    height = orig_image.shape[0]
    width = orig_image.shape[1]

    cls_id = []
    images = []
    c_bboxes = []
    c_scores = []

    for i, bbox in enumerate(bboxes[0]):
        if scores[0][i] > thresh:
            bbox = bbox.asnumpy()
            c_bboxes.append(bbox)
            xmin, ymin, xmax, ymax = extend_image(height, width, bbox, ratio=0.05)
            images.append(orig_image[ymin: ymax, xmin:xmax, :])

    for img in images:
        img = mx.nd.array(img)
        img = transform_test(img)
        img = mx.nd.stack(img)
        output = model(img)
        ids = output.argmax(axis=1).astype('int').asnumpy()[0]
        cls_id.append(ids)
        c_scores.append(mx.nd.softmax(output, axis=1)[0][ids].asnumpy()[0])

    plot_bbox(orig_image, c_bboxes, c_scores, cls_id, cls_name, thresh=0.1)




# model.hybridize()

im_name = 'saluki3.jpg'
x, orig_img = gluoncv.data.transforms.presets.rcnn.load_test(im_name)


model = models.vgg16_faster_rcnn()
model.load_parameters('/home/qinliang/Desktop/dog_detection_gluoncv/checkpoints/epoch9_map_0.8939.params')
box_ids, scores, bboxes = model(x)


del model
sets = vision.ImageFolderDataset('/home/qinliang/dataset/stanford_dog_dataset/cut_images_train', flag=1)
cls_name = sets.synsets
# model = models.VGG16(num_classes=len(cls_name))
model = models.Resnet50_v2(num_classes=len(cls_name))
model.load_parameters('/home/qinliang/Desktop/github/dog_recognition_gluoncv/checkpoints/epoch8_acc_86.87.params')

plot(orig_img, model, box_ids, scores, bboxes, cls_name, thresh=0.5)

plt.show()