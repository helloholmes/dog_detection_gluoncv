# dog_detection_gluoncv
a dog detector and classifier trained by Stanford dog dataset
***
## Some examples
![good example](https://github.com/helloholmes/dog_detection_gluoncv/raw/master/pictures/Figure_1-3.png)![good example](https://github.com/helloholmes/dog_detection_gluoncv/raw/master/pictures/Figure_1-4.png)
### Environment
---
[mxnet](http://mxnet.incubator.apache.org/) 1.3.0
[gluoncv](https://gluon-cv.mxnet.io/) 0.3.0
### Dataset
---
[Standford dog dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
### Train and evaluation
---
### Models
---
Use VGG16 as basic network for extracting features.
First train VGG16 for classification.
Second train Faster-RCNN for dog detection, I don't use Faster-RCNN as a dog classifier because it's difficult to detect and classify dog at the same time. Different kinds of dogs share same features and I detect and classify separately to avoid contradiction of features.
### Demo
---

