*THIS REPO IS UNDER DEVELOPMENT RIGHT NOW*

The project is a part of our course CS313b : Machine Learning 

@Indian Institute of Information Technology, Design and Manufacting, Jabalpur

# Reverse-Oldification of Image frames

## ABSTRACT :

Here we aim to study and implement a deep learning architecture revolving around the application of a neural network in order to rejuvenate old images, that is by colorizing them, andhence making them ‘alive’ again. Image restoration has always been a topic of interest, with applications of paramount importance such as restoring cum colorizing images of ancientmanuscripts, or extracting useful information from the images of ancient historical artifacts(after de-oldifying it to increase the color channels), or even bringing a black and white snapshot from the 90s to this century. Framework employed in this architecture may also be parallely used in the CSI(Crime Scene Investigation) division to enhance the details of crime scene image; and derive useful features from it. This process has itself expedited with the advent of the modern deep-learning era, where GPUs and TPUs are getting more and more powerful as time passes.

![Sample](https://hendrikholderick.files.wordpress.com/2012/01/old-pictures-recolored.jpg)

## INTRODUCTION:

As said by Ian Goodfellow, the promise of deep learning is to discover rich, hierarchical models that are not only robust but also represent probability distributions over the kinds of data encountered in Artificial Intelligence applications, such as Computer Vision, Natural Language Processing, Machine translation and so on. de-Oldification is also an exponentially expanding unit encapsulated inside the sphere of Artificial Intelligence where we aim to rejuvenate an old black and white image by adding color channels to it(at the very basic). For implementing this, we will take the assistance of Generative Adversarial Networks(GANs). The following sections would describe GAN in a nutshell. 

## GENERATIVE ADVERSARIAL NETWORKS(GANs):

GANs were proposed back in 2014, thanks to the work of Ian Goodfellow as a new framework for estimating generative models via an adversarial process, in which we simultaneously train two deep neural network models. In the proposed adversarial nets framework, the generative model is pitted against an adversary: a discriminative model that learns to determine whether a sample is from the model distribution or the data distribution. The generative model can be thought of as analogous to a team of counterfeiters, trying to produce fake currency and use it without detection, while the discriminative model is analogous to police, trying to detect the 
counterfeit currency. Competition in this game drives both neural networks(generator and discriminator) to improve their methods until the counterfeits are indistinguishable from the genuine articles. 

![Generative Adversarial Network](https://miro.medium.com/max/1600/0*0_067YjiG3afW-ed.png)


## REPO CONTENTS:

1. base_gan_model : Before moving onto relatively complex tasks to enable the reverse-image olification, a base GAN model was created to generate images of numbers, after being trained on the MNIST dataset, publically available. Both generators and discriminators were designed using PyTorch deep learning framework. The .ipynb notebook is included in this repository.

## Tensors : 

Tensors: In simple words, its just an n-dimensional array in PyTorch. Tensors support some additional enhancements which make them unique: Apart from CPU, they can be loaded or the GPU for faster computations.

## AutoGrad Engine of PyTorch :

This class is an engine to calculate derivatives (Jacobian-vector product to be more precise). It records a graph of all the operations performed on a gradient enabled tensor and creates an acyclic graph called the dynamic computational graph. The leaves of this graph are input tensors and the roots are output tensors. Gradients are calculated by tracing the graph from the root to the leaf and multiplying every gradient in the way using the chain rule.

![Mechanics of Autograd engine](https://miro.medium.com/max/942/1*viCEZbSODfA8ZA4ECPwHxQ.png)

## REFERENCES : 

1. Generative Adversarial Networks by Ian Goodfellow, Yoshua Bengio, etc.

(https://arxiv.org/abs/1406.2661)

2. The Heart Of Pytorch 

(https://towardsdatascience.com/pytorch-autograd-understanding-the-heart-of-pytorchs-magic-2686cd94ec95)

