CUDA_HOME=/usr/local/cuda-5.5

LD_LIBRARY_PATH=$(CUDA_HOME)/lib
export LD_LIBRARY_PATH

all: build test_run
build:
	$(CUDA_HOME)/bin/nvcc -o rbm_test rbm.cu rbm_kernels.cu rbm_test.cpp -lcublas -lcurand
test_run:
	./rbm_test
clean:
	rm rbm_test
