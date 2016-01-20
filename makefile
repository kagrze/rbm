all: build test_run
build:
	nvcc -o rbm_test rbm.cu rbm_kernels.cu rbm_test.cpp -lcublas -lcurand
download_mnist:
	wget -P mnist http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
	gunzip mnist/train-images-idx3-ubyte.gz
	wget -P mnist http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
	gunzip mnist/t10k-images-idx3-ubyte.gz
test_run:
	./rbm_test
clean:
	rm -f rbm_test
	rm -rf mnist
