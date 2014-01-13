all: build test_run
build:
	nvcc -o rbm_test rbm.cu rbm_kernels.cu rbm_test.cpp -lcublas -lcurand
test_run:
	./rbm_test
clean:
	rm rbm_test
