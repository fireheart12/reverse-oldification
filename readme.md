# Reverse-Oldification Of Images

## (I) Abstract : 
Here we aim to study and implement a deep learning architecture revolving around the application of a neural network in order to rejuvenate black and white images, that is by colorizing them, and hence making them ‘alive’ again. Image restoration cum reconstruction has always been a topic of interest, with applications such as extracting useful information from the images of ancient historical artifacts(after reverse-oldifying it to increase the color channels and hence, the amount of information encapsulated), or even bringing a black and white snapshot from the 90s to this century(applications in entertainment industry), or colorizing the popular Mangas(Japanese comics), which are drawn without colors(mostly).  The heavy process has expedited with the advent of the modern deep-learning/Big Data era, where GPUs and TPUs are getting more and more powerful as time progresses, along with a massive surge in the amount of data available to learn from. 

## (II) Dataset used : 
The dataset is a result of seven researches from the website flickr containing real world photos : 
* Landscapes(900+ pictures)
* Landscapes mountains(900+ pictures)
* Landscapes desert(100+ pictures)
* Landscapes sea(500+ pictures)
* Landscapes beach(500+ pictures)
* Landscapes island(500+ pictures)
* Landscapes Japan(500+ pictures)

Flickr : https://www.flickr.com/ 

Some sample images from landscape data set : 

![](https://github.com/CodingWitcher/reverse-oldification/blob/master/images/dataset_image01.jpg) 
![](https://github.com/CodingWitcher/reverse-oldification/blob/master/images/dataset_image02.jpg)
![](https://github.com/CodingWitcher/reverse-oldification/blob/master/images/dataset_image03.jpg)
![](https://github.com/CodingWitcher/reverse-oldification/blob/master/images/dataset_image04.jpg)

This dataset containing 3000-4000 images is publicly hosted on Kaggle from where it’s downloaded and subsequently uploaded(after cleaning) on the following  google drive path:  

https://drive.google.com/drive/folders/10FpGEaeEM2AROcP0zAJ8Oz1QmERD7ifB?usp=sharing    

Kaggle link : https://www.kaggle.com/arnaud58/landscape-pictures# 

## (III) Libraries used : 
### Numpy : 

Fundamental package for scientific computing in Python3, helping us in creating and managing n-dimensional tensors. A vector can be regarded as a 1-D tensor, matrix as 2-D, and so on.
![](https://github.com/CodingWitcher/reverse-oldification/blob/master/images/tensor.jpeg) 

### Matplotlib : 

A Python3 plotting library used for data visualization.

### SkImage : 

Is an open source Python3 package designed for image processing.

### Tensorflow : 

Is an open source deep learning framework for dataflow and differentiable programming.  It’s created and maintained by Google. 

### Tqdm : 

Is a progress bar library with good support for nested loops and Jupyter notebooks.  

## (IV) Image pre-processing : 


