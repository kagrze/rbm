train_file = mnist/train-images-idx3-ubyte
test_file = mnist/t10k-images-idx3-ubyte

all: test
build:
	nvcc -arch=compute_20 -o rbm_test utils.cpp rbm_kernels.cu rbm.cu rbm_test.cpp -lcublas -lcurand
download_mnist:
ifeq ($(wildcard $(train_file)),)
	wget -P mnist http://yann.lecun.com/exdb/$(train_file).gz
	gunzip $(train_file).gz
	wget -P mnist http://yann.lecun.com/exdb/$(test_file).gz
	gunzip $(test_file).gz
endif
test: build download_mnist
	./rbm_test mnist
print: build download_mnist
	./rbm_test print
clean:
	rm -f rbm_test
	rm -rf mnist
