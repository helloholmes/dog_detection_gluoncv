# dog_detection_gluoncv  
a dog detector and classifier trained by Stanford dog dataset
***
### Some examples
![good example](https://github.com/helloholmes/dog_detection_gluoncv/raw/master/pictures/Figure_1-3.png)![good example](https://github.com/helloholmes/dog_detection_gluoncv/raw/master/pictures/Figure_1-4.png)  
***
### Environment
[mxnet](http://mxnet.incubator.apache.org/) 1.3.0  
[gluoncv](https://gluon-cv.mxnet.io/) 0.3.0  
[visdom](https://github.com/facebookresearch/visdom) 0.1.7
***
### Dataset
[Standford dog dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)  
I also use [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification) competition on [kaggle](https://www.kaggle.com) to train the classification network.
***
### Train and evaluation
run `python main.py` to train the Faster RCNN.  
You can change some hyperparameters in the functino `__main__`.
***
### Models
Use VGG16 as basic network for extracting features.  
First train VGG16 for classification.  
Second train Faster-RCNN for dog detection, I don't use Faster-RCNN as a dog classifier because it's difficult to detect and classify dog at the same time. Different kinds of dogs share same features and I detect and classify separately to avoid contradiction of features.
***
### Demo
run `python demo.py` to show the results.
***
### Something to improve
####Some not so good results  
![bad example](https://github.com/helloholmes/dog_detection_gluoncv/raw/master/pictures/Figure_1-5.png)  
Breeds such as malamute and husky, have similar appearance. It's a little bit difficult for the machine to distinguish them accurately. The confidence will be not so high.  
![bad example](https://github.com/helloholmes/dog_detection_gluoncv/raw/master/pictures/Figure_1-6.png)![bad example](https://github.com/helloholmes/dog_detection_gluoncv/raw/master/pictures/Figure_1-8.png)  
If two objects get too close, it's very easy for machine to detect a wrong object between them. And the IOU doesn't touch the thresh so you can't filter it by nms.