rbm
===

A CUDA implementation of a [restricted Boltzmann machine](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) (RBM) with the [Contrastive Divergence](http://www.cs.toronto.edu/~hinton/absps/nccd.pdf) learning algorithm.

In order to test the implementation just type `make` in your command line. The first time you do this the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digits dataset will be download. The dataset is used to train RBM. You can print sample MNIST digits to the console (using ASCII art) by typing `make print`.

The implementation demonstrates [cuBLAS](https://developer.nvidia.com/cublas), [cuRAND](https://developer.nvidia.com/curand) and [Thrust](https://developer.nvidia.com/thrust) libraries. It was tested with CUDA version 7.5 on Nvidia Tesla M2090 GPU.

Please bear in mind that this is a simplified implementation. To make it more applicable some optimizations should be undertaken. For example stochastic gradient descent (or its mini-batch version) should be used instead of batch gradient descent. Other optimization would be to reuse memory buffers among epochs instead of reallocate them within each step.
