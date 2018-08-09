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

def plot(orig_image, model, box_ids, scores, bboxes, class_names, thresh=0.5):
    cls_id = []
    images = []
    c_bboxes = []
    c_scores = []

    for i, bbox in enumerate(bboxes[0]):
        if scores[0][i] > thresh:
            bbox = bbox.asnumpy()
            c_bboxes.append(bbox)
            images.append(orig_image[int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3]), :])

    for img in images:
        img = mx.nd.array(img)
        img = transform_test(img)
        img = mx.nd.stack(img)
        output = model(img)
        ids = output.argmax(axis=1).astype('int').asnumpy()[0]
        cls_id.append(ids)
        c_scores.append(mx.nd.softmax(output, axis=1)[0][ids].asnumpy()[0])

    plot_bbox(orig_image, c_bboxes, c_scores, cls_id, cls_name)


model = models.vgg16_faster_rcnn()
model.load_parameters('/home/qinliang/Desktop/dog_detection_gluoncv/checkpoints/epoch9_map_0.8939.params')

# model.hybridize()

im_name = 'saluki3.jpg'
x, orig_img = gluoncv.data.transforms.presets.rcnn.load_test(im_name)

box_ids, scores, bboxes = model(x)

'''
cls_id = []
images = []
c_bboxes = []
c_scores = []
for i, bbox in enumerate(bboxes[0]):
    if scores[0][i] > 0.5:
        bbox = bbox.asnumpy()
        print(bbox)
        c_bboxes.append(bbox)
        images.append(orig_img[int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3]), :])
'''
del model
sets = vision.ImageFolderDataset('/home/qinliang/dataset/stanford_dog_dataset/cut_images_train', flag=1)
cls_name = sets.synsets
model = models.VGG16(num_classes=len(cls_name))
model.load_parameters('/home/qinliang/Desktop/kaggle/dog_recognition_gluon/cut_image_checkpoints/epoch14_acc_79.66.params')
'''
# images = transform_test(images)
# print(images.shape)
for image in images:
    image = mx.nd.array(image)
    image = transform_test(image)
    # print(image.shape)
    image = mx.nd.stack(image)
    output = model(image)
    ids = output.argmax(axis=1).astype('int').asnumpy()[0]
    # print(output)
    cls_id.append(ids)
    c_scores.append(mx.nd.softmax(output, axis=1)[0][ids].asnumpy()[0])

print(cls_id)
print(c_scores)
# ax = gluoncv.utils.viz.plot_bbox(orig_img, c_bboxes, c_scores, cls_id, class_names=cls_name, thresh=0.5)
plot_bbox(orig_img, c_bboxes, c_scores, cls_id, cls_name)
'''
plot(orig_img, model, box_ids, scores, bboxes, cls_name)

plt.show()
