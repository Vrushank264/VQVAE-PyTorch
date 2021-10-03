Pytorch implementation of [VQVAE](https://arxiv.org/abs/1711.00937).

This paper combines 2 tricks:
1) Vector Quantization (check out this amazing [blog](https://wiki.aalto.fi/pages/viewpage.action?pageId=149883153) for better understanding.)
2) Straight-Through (It solves the problem of back-propagation through discrete latent variables, which are intractable.)

This model has a neural network encoder and decoder, and a prior just like the vanila Variational AutoEncoder(VAE). But this model also has a latent embedding space called `codebook`(size: K x D). Here, K is the size of latent space and D is the dimension of each embedding e.    

In vanilla variational autoencoders, the output from the encoder z(x) is used to parameterize a Normal/Gaussian distribution, which is sampled from to get a latent representation z of the input x using the 'reparameterization trick'. This latent representation is then passed to the decoder. However, In VQVAEs, z(x) is used as a "key" to do nearest neighbour lookup into the embedding codebook c, and get zq(x), the closest embedding in the space. This is called `Vector Quantization(VQ)` operation. Then, zq(x) is passed to the decoder, which reconstructs the input x. The decoder can either parameterize p(x|z) as the mean of Normal distribution using a transposed convolution layer like in vannila VAE, or it can autoregressively generate categorical distribution over [0,255] pixel values like PixelCNN. In this project, the first approach is used.

The loss function is combined of 3 components:
1) `Regular Reconstruction loss`
2) `Vector Quantization loss`
3) `Commitment loss`

Vector Quantization loss encourages the items in the codebook to move closer to the encoder output `||sg[ze(x) - e||^2]` and Commitment loss encourages the output of the encoder to be close to embedding it picked, to commit to its codebook embedding. `||ze(x) - sg[e]]||^2` . commitment loss is multiplied with a constant beta, which is 1.0 for this project. Here, sg means "stop-gradient". Which means we don't propagate the gradients with respect to that term.


## Results:

The Model is trained on MNIST and CIFAR10 datasets.  

<h4> Target :point_right: Reconstructed Image </h4> <br>
<p float="left">
  <img src="https://github.com/Vrushank264/VQVAE-PyTorch/blob/main/Results/target_mnist.png" />
  :point_right:
  <img src="https://github.com/Vrushank264/VQVAE-PyTorch/blob/main/Results/recon_mnist.png" /> 
</p>

<p float="left">
  <img src="https://github.com/Vrushank264/VQVAE-PyTorch/blob/main/Results/target_cifar10.png" />
  :point_right:
  <img src="https://github.com/Vrushank264/VQVAE-PyTorch/blob/main/Results/recon_cifar10.png" /> 
</p>


![gif](https://github.com/Vrushank264/VQVAE-PyTorch/blob/main/Results/interpolation.gif)


## Details:

1) Trained models for MNIST and CIFAR10 are in the `Trained models` directory.
2) Hidden size of the bottleneck(z) for MNIST and CIFAR10 is 128, 256 respectively.  
