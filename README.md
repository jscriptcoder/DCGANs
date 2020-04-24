# Deep Convolutional Generative Adversarial Networks

WORK IN PROGRESS!!

This project is based on the paper: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434), and part of [Udacity Deep Learning Nanodegree program](https://www.udacity.com/course/deep-learning-nanodegree--nd101)

The goal of this project is to get a generator network to creaate new images of faces that look as realistic as possible!

Examples:
<p align="center">
  <img src="assets/dcgan_example1.png" width="80%" /><br />
  <img src="assets/dcgan_example2.png" width="80%" /><br />
  <sub>Images from [paper](https://arxiv.org/pdf/1511.06434.pdf)</sub>
</p>

## What is a DCGAN?
A DCGAN is anextension of Generative Adversarial Network that uses convolutional and convolutional-transpose layers in the discriminator and generator, respectively. As described in the [paper](https://arxiv.org/pdf/1511.06434.pdf), the **discriminator** is made up of strided [convolutional layers](https://cs231n.github.io/convolutional-networks/#conv), [batch norm layers](https://arxiv.org/abs/1502.03167), and [LeakyReLU activations](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#leakyrelu). The input is an image and the output is a scalar probability that the input is from the real data distribution. The **generator** is comprised of [convolutional-transpose layers](https://medium.com/activating-robotic-minds/up-sampling-with-transposed-convolution-9ae4f2df52d0), batch norm layers, and [ReLU activations](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#relu). The input is a [latent vector](https://towardsdatascience.com/understanding-latent-space-in-machine-learning-de5a7c687d8d), `z`, that is drawn from a standard normal distribution and the output is an image, same size as the input of the discriminator

## Architecture guidelines for stable Deep Convolutional GANs

- Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
- Use batchnorm in both the generator and the discriminator.
- Remove fully connected hidden layers for deeper architectures.
- Use ReLU activation in generator for all layers except for the output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers.

## Details of Adversarial Training

- Scaling images to the range of the tanh activation function [-1, 1].
- All weights are initialized from a zero-centered Normal distribution with standard deviation 0.02.
- In the LeakyReLU, the slope of the leak is set to 0.2.
- Adam optimizer with learning rate of 0.0002 and β1 is set to 0.5.
