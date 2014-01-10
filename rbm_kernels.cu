#include "rbm_kernels.h"

__global__ void sigmoid(float * dInputArray, float * dOutputArray, int arraySize) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx<arraySize)
                dOutputArray[idx] = 1.0/(1.0 + exp(-dInputArray[idx]));
}

__global__ void greaterThan(float * dLHS, float * dRHS, float * dOutputArray, int arraySize) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx<arraySize)
                dOutputArray[idx] = dLHS[idx] > dRHS[idx];
}

__global__ void updateWeight(float * dWeights, float * dPositiveAssociation, float * dNegativeAssociation, int weightsNumber, int examplesNumber, float learningRate) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx<weightsNumber)
                dWeights[idx] += learningRate*((dPositiveAssociation[idx] - dNegativeAssociation[idx])/examplesNumber);
}

__global__ void subAndSquare(float * a, float * b, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx<size) {
                float sub = a[idx] - b[idx];
                a[idx] = sub*sub;
        }
}

//TODO optimize by using shared memory
__global__ void sumReduce(float * array, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx>=size/2)
                return;
	for(int i=1; i<size;i*=2) {
		if(idx*2*i+i<size) {
			array[idx*2*i] += array[idx*2*i+i];
			__syncthreads();
		}
	}
}
