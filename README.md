# Deep Convolutional Generative Adversarial Networks

WORK IN PROGRESS!!

This project is based on the paper: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434), and part of [Udacity Deep Learning Nanodegree program](https://www.udacity.com/course/deep-learning-nanodegree--nd101)

The goal is to train a generator that learns to create new image from random noise. 

## What is a DCGAN?
A DCGAN is anextension of Generative Adversarial Network that uses convolutional and convolutional-transpose layers in the discriminator and generator, respectively. As described in the [paper](https://arxiv.org/pdf/1511.06434.pdf), the **discriminator** is made up of strided [convolutional layers](https://cs231n.github.io/convolutional-networks/#conv), [batch norm layers](https://arxiv.org/abs/1502.03167), and [LeakyReLU activations](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#leakyrelu). The input is an image and the output is a scalar probability that the input is from the real data distribution. The **generator** is comprised of [convolutional-transpose layers](https://medium.com/activating-robotic-minds/up-sampling-with-transposed-convolution-9ae4f2df52d0), batch norm layers, and [ReLU activations](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#relu). The input is a [latent vector](https://towardsdatascience.com/understanding-latent-space-in-machine-learning-de5a7c687d8d), `z`, that is drawn from a standard normal distribution and the output is an image, same size as the input of the discriminator
