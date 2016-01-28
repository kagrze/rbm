#include <iostream>
#include <cublas_v2.h>
#include <curand.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "rbm.h"
#include "rbm_kernels.h"
#include "utils.h"

#define INFO  false
#define DEBUG false

#define BLOCK_SIZE 1024 // it is assumed that a number of training examples is big (exv and exh are bigger than 1024)
                        // otherwise BLOCK_SIZE should be set dynamically

RBM::RBM(int visible, int hidden, float rate) {
    numVisible   = visible;
    numHidden    = hidden;
    learningRate = rate;

    int weightsNumber = (numVisible + 1) * (numHidden + 1);  // +1 because of bias
    checkCudaError(__LINE__, cudaMalloc(&dWeights, weightsNumber * sizeof(float)));
    checkCuRandError(__LINE__, curandCreateGenerator(&generator, CURAND_RNG_QUASI_DEFAULT));
    checkCuRandError(__LINE__, curandGenerateNormal(generator, dWeights, weightsNumber, 0.0, 0.1));

    if (INFO) {
        std::cout << "Initial weights:" << std::endl;
        printDeviceColumnMajorMatrix(dWeights, numVisible + 1, numHidden + 1);
    }

    checkCuBlasError(__LINE__, cublasCreate(&handle));
    std::cout << "RBM initialized" << std::endl;
}

RBM::~RBM() {
    checkCudaError(__LINE__, cudaFree(dWeights));
    cublasDestroy(handle);
    curandDestroyGenerator(generator);
    std::cout << "RBM destroyed" << std::endl;
}

float *RBM::hiddenActivationProbabilities(float *dVisibleUnitsStates, int examplesNumber) {
    float *dHiddenUnitsActivationEnergy;        // matrix of float values of dim exh
    float *dHiddenUnitsActivationProbabilities; // matrix of [0,1] values of dim exh

    int hiddenBufferSize = (numHidden + 1) * examplesNumber;

    checkCudaError(__LINE__, cudaMalloc(&dHiddenUnitsActivationEnergy, hiddenBufferSize * sizeof(float)));
    checkCudaError(__LINE__, cudaMalloc(&dHiddenUnitsActivationProbabilities, hiddenBufferSize * sizeof(float)));

    if (DEBUG) std::cout << "Calculating hidden units activation energies" << std::endl;

    const float alpha = 1;
    const float beta  = 0;
    checkCuBlasError(__LINE__, cublasSgemm(
                         handle,
                         CUBLAS_OP_N,
                         CUBLAS_OP_N,
                         examplesNumber,
                         numHidden + 1,
                         numVisible + 1,
                         &alpha,
                         dVisibleUnitsStates,
                         examplesNumber, // lda
                         dWeights,
                         numVisible + 1, // ldb
                         &beta,
                         dHiddenUnitsActivationEnergy,
                         examplesNumber)); // ldc

    if (DEBUG) printDeviceColumnMajorMatrix(dHiddenUnitsActivationEnergy, examplesNumber, numHidden + 1);

    int blockNumber = hiddenBufferSize / BLOCK_SIZE + 1;
    if (DEBUG) std::cout << "Calculating hidden probabilities " << BLOCK_SIZE << std::endl;
    sigmoid<<<blockNumber, BLOCK_SIZE>>>(dHiddenUnitsActivationEnergy, dHiddenUnitsActivationProbabilities, hiddenBufferSize);
    checkCudaError(__LINE__);
    if (DEBUG) printDeviceColumnMajorMatrix(dHiddenUnitsActivationProbabilities, examplesNumber, numHidden + 1);

    checkCudaError(__LINE__, cudaFree(dHiddenUnitsActivationEnergy));
    return dHiddenUnitsActivationProbabilities;
}

float *RBM::visibleActivationProbabilities(float *dHiddenUnitsStates, int examplesNumber) {
    float *dVisibleUnitsActivationEnergy;        // matrix of float values of dim exv
    float *dVisibleUnitsActivationProbabilities; // matrix of [0,1] values of dim exv

    int visibleBufferSize = (numVisible + 1) * examplesNumber;

    checkCudaError(__LINE__, cudaMalloc(&dVisibleUnitsActivationEnergy, visibleBufferSize * sizeof(float)));
    checkCudaError(__LINE__, cudaMalloc(&dVisibleUnitsActivationProbabilities, visibleBufferSize * sizeof(float)));

    if (DEBUG) std::cout << "Calculating visible units activation energies" << std::endl;

    const float alpha = 1;
    const float beta  = 0;
    checkCuBlasError(__LINE__, cublasSgemm(
                         handle,
                         CUBLAS_OP_N,
                         CUBLAS_OP_T,
                         examplesNumber,
                         numVisible + 1,
                         numHidden + 1,
                         &alpha,
                         dHiddenUnitsStates,
                         examplesNumber, // lda
                         dWeights,
                         numVisible + 1, // ldb
                         &beta,
                         dVisibleUnitsActivationEnergy,
                         examplesNumber)); // ldc

    if (DEBUG) printDeviceColumnMajorMatrix(dVisibleUnitsActivationEnergy, examplesNumber, numVisible + 1);

    int blockNumber = visibleBufferSize / BLOCK_SIZE + 1;
    if (DEBUG) std::cout << "Calculating visible probabilities" << std::endl;

    if (DEBUG) printDeviceColumnMajorMatrix(dVisibleUnitsActivationProbabilities, examplesNumber, numVisible + 1);

    sigmoid<<<blockNumber, BLOCK_SIZE>>>(dVisibleUnitsActivationEnergy, dVisibleUnitsActivationProbabilities, visibleBufferSize);

    checkCudaError(__LINE__);

    if (DEBUG) printDeviceColumnMajorMatrix(dVisibleUnitsActivationProbabilities, examplesNumber, numVisible + 1);

    checkCudaError(__LINE__, cudaFree(dVisibleUnitsActivationEnergy));
    return dVisibleUnitsActivationProbabilities;
}

float *RBM::computeAssociations(float *dVisibleUnitsActivationProbabilities, float *dHiddenUnitsActivationProbabilities, int examplesNumber) {
    float *dAssociations; // vxh matrix

    checkCudaError(__LINE__, cudaMalloc(&dAssociations, (numVisible + 1) * (numHidden + 1) * sizeof(float))); // +1 because of bias

    const float alpha = 1;
    const float beta  = 0;
    checkCuBlasError(__LINE__, cublasSgemm(
                         handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         numVisible + 1,
                         numHidden + 1,
                         examplesNumber,
                         &alpha,
                         dVisibleUnitsActivationProbabilities,
                         examplesNumber, // lda
                         dHiddenUnitsActivationProbabilities,
                         examplesNumber, // ldb
                         &beta,
                         dAssociations,
                         numVisible + 1)); // ldc

    if (DEBUG) std::cout << "Associations:" << std::endl;
    if (DEBUG) printDeviceColumnMajorMatrix(dAssociations, numVisible + 1, numHidden + 1);

    return dAssociations;
}

// a contrastive divergence (CD_1) learning algorithm; batched version
void RBM::train(float *hTrainingData, int examplesNumber, int maxEpochs) {
    float hBias[examplesNumber];                        // will be added as a first column of training data
    std::fill_n(hBias, examplesNumber, 1.0);

    float *dVisibleUnitsStates;                         // device copy of training data
    float *dVisibleUnitsActivationProbabilities;        // matrix of [0,1] of dimensions exv

    float *dHiddenUnitsStates;                          // matrix of boolean values of dimensions exh
    float *dPositiveHiddenUnitsActivationProbabilities; // matrix of [0,1] of dimensions exh

    float *dNegativeHiddenUnitsActivationProbabilities; // matrix of [0,1] of dimensions exh

    float *dPositiveAssociations;                       // matrix of dimensions vxh
    float *dNegativeAssociations;                       // matrix of dimensions vxh

    float *dRandom;                                     // matrix of dimensions exh of random values [0,1]

    int visibleBufferSize = (numVisible + 1) * examplesNumber;
    int hiddenBufferSize  = (numHidden + 1) * examplesNumber;

    checkCudaError(__LINE__, cudaMalloc(&dVisibleUnitsStates, visibleBufferSize * sizeof(float)));
    checkCudaError(__LINE__, cudaMalloc(&dHiddenUnitsStates, hiddenBufferSize * sizeof(float)));
    checkCudaError(__LINE__, cudaMalloc(&dRandom, hiddenBufferSize * sizeof(float)));

    for (int e = 0; e < maxEpochs; e++) {
        // a positive phase of the contrastive divergence

        // copy bias to the first column
        checkCudaError(__LINE__, cudaMemcpy(dVisibleUnitsStates, hBias, examplesNumber * sizeof(float), cudaMemcpyHostToDevice));

        // copy training data to remaining cells
        checkCudaError(__LINE__, cudaMemcpy(&dVisibleUnitsStates[examplesNumber],
                                            hTrainingData,
                                            numVisible * examplesNumber * sizeof(float),
                                            cudaMemcpyHostToDevice));

        if (DEBUG) std::cout << "Visible units states:" << std::endl;
        if (DEBUG) printDeviceColumnMajorMatrix(dVisibleUnitsStates, examplesNumber, numVisible + 1);

        // calculate positive hidden activation probabilities

        dPositiveHiddenUnitsActivationProbabilities = hiddenActivationProbabilities(dVisibleUnitsStates, examplesNumber);

        if (DEBUG) std::cout << "Fixing hidden units activation probabilities by setting bias to the first column" << std::endl;
        checkCudaError(__LINE__, cudaMemcpy(dPositiveHiddenUnitsActivationProbabilities, hBias, examplesNumber * sizeof(float), cudaMemcpyHostToDevice));
        if (DEBUG) printDeviceColumnMajorMatrix(dPositiveHiddenUnitsActivationProbabilities, examplesNumber, numHidden + 1);

        if (DEBUG) std::cout << "Calculating hidden unit states by sampling" << std::endl;
        checkCuRandError(__LINE__, curandGenerateUniform(generator, dRandom, hiddenBufferSize));
        int blockNumber = hiddenBufferSize / BLOCK_SIZE + 1;
        greaterThan<<<blockNumber, BLOCK_SIZE>>>(dPositiveHiddenUnitsActivationProbabilities, dRandom, dHiddenUnitsStates, examplesNumber * (numHidden + 1));
        checkCudaError(__LINE__);
        if (DEBUG) printDeviceColumnMajorMatrix(dHiddenUnitsStates, examplesNumber, numHidden + 1);

        dPositiveAssociations = computeAssociations(dVisibleUnitsStates, dPositiveHiddenUnitsActivationProbabilities, examplesNumber);

        // a negative (reconstruction) phase of the contrastive divergence

        // calculate negative visible probabilities
        dVisibleUnitsActivationProbabilities = visibleActivationProbabilities(dHiddenUnitsStates, examplesNumber);

        if (DEBUG) std::cout << "Visible Units Activation Probabilities:" << std::endl;
        if (DEBUG) printDeviceColumnMajorMatrix(dVisibleUnitsActivationProbabilities, examplesNumber, numVisible + 1);

        if (DEBUG) std::cout << "Fixing visible units activation probabilities by setting bias to the first column" << std::endl;
        checkCudaError(__LINE__, cudaMemcpy(dVisibleUnitsActivationProbabilities, hBias, examplesNumber * sizeof(float), cudaMemcpyHostToDevice));
        if (DEBUG) printDeviceColumnMajorMatrix(dVisibleUnitsActivationProbabilities, examplesNumber, numVisible + 1);

        // negative hidden probabilities
        dNegativeHiddenUnitsActivationProbabilities = hiddenActivationProbabilities(dVisibleUnitsActivationProbabilities, examplesNumber);

        if (DEBUG) std::cout << "Negative Hidden units activation probabilities:" << std::endl;
        if (DEBUG) printDeviceColumnMajorMatrix(dNegativeHiddenUnitsActivationProbabilities, examplesNumber, numHidden + 1);

        if (DEBUG) std::cout << "Calculating negative associations" << std::endl;
        dNegativeAssociations = computeAssociations(dVisibleUnitsActivationProbabilities, dNegativeHiddenUnitsActivationProbabilities, examplesNumber);

        if (DEBUG) std::cout << "Updating weights" << std::endl;
        int weightsNumber = (numHidden + 1) * (numVisible + 1);
        blockNumber = weightsNumber / BLOCK_SIZE + 1;
        updateWeight<<<blockNumber, BLOCK_SIZE>>>(dWeights, dPositiveAssociations, dNegativeAssociations, weightsNumber, examplesNumber, learningRate);
        checkCudaError(__LINE__);
        if (DEBUG) printDeviceColumnMajorMatrix(dWeights, numVisible + 1, numHidden + 1);

        blockNumber = visibleBufferSize / BLOCK_SIZE + 1;
        if (DEBUG) std::cout << "Calculating error - squares of subtractions: " << std::endl;

        // for memory efficiency we will write subtraction result to one of the input matrices (dVisibleUnitsStates)
        subAndSquare<<<blockNumber, BLOCK_SIZE>>>(dVisibleUnitsStates, dVisibleUnitsActivationProbabilities, visibleBufferSize);
        checkCudaError(__LINE__);
        if (DEBUG) printDeviceColumnMajorMatrix(dVisibleUnitsStates, examplesNumber, numVisible + 1);

        if (DEBUG) std::cout << "Calculation error - reducing sum:" << std::endl;
        thrust::device_ptr<float> dVisibleUnitsStatesPtr(dVisibleUnitsStates);
        float hError = thrust::reduce(dVisibleUnitsStatesPtr, dVisibleUnitsStatesPtr + visibleBufferSize, 0.0, thrust::plus<float>());

        if (DEBUG) printDeviceColumnMajorMatrix(dVisibleUnitsStates, examplesNumber, numVisible + 1);

        std::cout << "Reconstruction error after epoch " << e + 1 << " is " << hError << std::endl;

        checkCudaError(__LINE__, cudaFree(dVisibleUnitsActivationProbabilities));
        checkCudaError(__LINE__, cudaFree(dPositiveHiddenUnitsActivationProbabilities));
        checkCudaError(__LINE__, cudaFree(dNegativeHiddenUnitsActivationProbabilities));
        checkCudaError(__LINE__, cudaFree(dPositiveAssociations));
        checkCudaError(__LINE__, cudaFree(dNegativeAssociations));
    }

    checkCudaError(__LINE__, cudaFree(dRandom));
    checkCudaError(__LINE__, cudaFree(dVisibleUnitsStates));
    checkCudaError(__LINE__, cudaFree(dHiddenUnitsStates));

    if (INFO) std::cout << "Learned weights:" << std::endl;
    if (INFO) printDeviceColumnMajorMatrix(dWeights, numVisible + 1, numHidden + 1);
}

float *RBM::hiddenStates(float *hVisible) {
    float *dVisible;
    float *dHidden;
    float *hHidden;
    float *dRandom;

    checkCudaError(__LINE__, cudaMalloc(&dVisible, (numVisible + 1) * sizeof(float)));
    checkCudaError(__LINE__, cudaMalloc(&dHidden, (numHidden + 1) * sizeof(float)));

    float bias = 1.0;
    checkCudaError(__LINE__, cudaMemcpy(dVisible, &bias, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(__LINE__, cudaMemcpy(&dVisible[1], hVisible, numVisible * sizeof(float), cudaMemcpyHostToDevice)); // set bias

    dHidden = hiddenActivationProbabilities(dVisible, 1);

    // sampling
    checkCudaError(__LINE__, cudaMalloc(&dRandom, (numHidden + 1) * sizeof(float)));
    checkCuRandError(__LINE__, curandGenerateUniform(generator, dRandom, numHidden + 1));
    int blockNumber = (numHidden + 1) / BLOCK_SIZE + 1;
    greaterThan<<<blockNumber, BLOCK_SIZE>>>(dHidden, dRandom, dHidden, numHidden + 1);
    checkCudaError(__LINE__);

    hHidden = (float *) malloc(numHidden * sizeof(float));
    cudaMemcpy(hHidden, &dHidden[1], numHidden * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError(__LINE__, cudaFree(dHidden));
    checkCudaError(__LINE__, cudaFree(dVisible));
    checkCudaError(__LINE__, cudaFree(dRandom));
    return hHidden;
}
