#include <iostream>
#include <cublas_v2.h>
#include <curand.h>
#include "rbm.h"
#include "rbm_kernels.h"

#define INFO  false
#define DEBUG false

void printColumnMajorMatrix(float *A, int nrRows, int nrCols) {
    for (int i = 0; i < nrRows; ++i) {
        for (int j = 0; j < nrCols; ++j)
            std::cout << A[nrRows * j + i] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void printDeviceColumnMajorMatrix(float *dA, int nrRows, int nrCols) {
    int   size = nrRows * nrCols;
    float hA[size];

    cudaMemcpy(hA, dA, size * sizeof(float), cudaMemcpyDeviceToHost);
    printColumnMajorMatrix(hA, nrRows, nrCols);
}

void checkCuBlasError(int line, cublasStatus_t stat) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS error at line: " << line << ", status: " << stat << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCuRandError(int line, curandStatus_t stat) {
    if (stat != CURAND_STATUS_SUCCESS) {
        std::cout << "CURAND error at line: " << line << ", status: " << stat << std::endl;
        exit(EXIT_FAILURE);
    }
}

RBM::RBM(int visible, int hidden, float rate) {
    blockSize = 512;    // it is assumed that number of training examples is big (exv and exh are bigger than 512);
                        // otherwise blockSize should be set dynamically
    numVisible   = visible;
    numHidden    = hidden;
    learningRate = rate;

    int weightsNumber = (numVisible + 1) * (numHidden + 1);  // +1 because of bias
    cudaMalloc(&dWeights, weightsNumber * sizeof(float));
    checkCuRandError(__LINE__, curandCreateGenerator(&generator, CURAND_RNG_QUASI_DEFAULT));
    checkCuRandError(__LINE__, curandGenerateNormal(generator, dWeights, weightsNumber, 0.0, 0.1));

    if (INFO) {
        std::cout << "Initial weights=" << std::endl;
        printDeviceColumnMajorMatrix(dWeights, numVisible + 1, numHidden + 1);
    }

    checkCuBlasError(__LINE__, cublasCreate(&handle));
    std::cout << "RBM initialized" << std::endl;
}

RBM::~RBM() {
    cudaFree(dWeights);
    cublasDestroy(handle);
    curandDestroyGenerator(generator);
    std::cout << "RBM destroyed" << std::endl;
}

float *RBM::hiddenActivationProbabilities(float *dVisibleUnitsStates, int examplesNumber) {
    float *dHiddenUnitsActivationEnergy;        // matrix of float values of dim exh
    float *dHiddenUnitsActivationProbabilities; // matrix of [0,1] values of dim exh

    cudaMalloc(&dHiddenUnitsActivationEnergy, (numHidden + 1) * examplesNumber * sizeof(float));
    cudaMalloc(&dHiddenUnitsActivationProbabilities, (numHidden + 1) * examplesNumber * sizeof(float));

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

    int blockNumber = examplesNumber * (numHidden + 1) / blockSize + 1;
    if (DEBUG) std::cout << "Calculating hidden probabilities" << std::endl;
    sigmoid<<<blockNumber, blockSize>>>(dHiddenUnitsActivationEnergy, dHiddenUnitsActivationProbabilities, examplesNumber * (numHidden + 1));
    if (DEBUG) printDeviceColumnMajorMatrix(dHiddenUnitsActivationProbabilities, examplesNumber, numHidden + 1);

    cudaFree(dHiddenUnitsActivationEnergy);
    return dHiddenUnitsActivationProbabilities;
}

float *RBM::visibleActivationProbabilities(float *dHiddenUnitsStates, int examplesNumber) {
    float *dVisibleUnitsActivationEnergy;        // matrix of float values of dim exv
    float *dVisibleUnitsActivationProbabilities; // matrix of [0,1] values of dim exv

    cudaMalloc(&dVisibleUnitsActivationEnergy, (numVisible + 1) * examplesNumber * sizeof(float));
    cudaMalloc(&dVisibleUnitsActivationProbabilities, (numVisible + 1) * examplesNumber * sizeof(float));

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

    int blockNumber = examplesNumber * (numVisible + 1) / blockSize + 1;
    if (DEBUG) std::cout << "Calculating visible probabilities" << blockNumber << " " << blockSize << std::endl;

    sigmoid<<<blockNumber, blockSize>>>(dVisibleUnitsActivationEnergy, dVisibleUnitsActivationProbabilities, examplesNumber * (numVisible + 1));

    if (DEBUG) printDeviceColumnMajorMatrix(dVisibleUnitsActivationProbabilities, examplesNumber, numVisible + 1);

    cudaFree(dVisibleUnitsActivationEnergy);
    return dVisibleUnitsActivationProbabilities;
}

float *RBM::computeAssociations(float *dVisibleUnitsActivationProbabilities,
                                float *dHiddenUnitsActivationProbabilities,
                                int    examplesNumber) {
    float *dAssociations; // vxh matrix

    cudaMalloc(&dAssociations, (numVisible + 1) * (numHidden + 1) * sizeof(float)); // +1 because of bias

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

    cudaMalloc(&dVisibleUnitsStates, (numVisible + 1) * examplesNumber * sizeof(float));
    cudaMalloc(&dHiddenUnitsStates, (numHidden + 1) * examplesNumber * sizeof(float));
    cudaMalloc(&dRandom, examplesNumber * (numHidden + 1) * sizeof(float));

    for (int e = 0; e < maxEpochs; e++) {
        // a positive phase of the contrastive divergence
        if (DEBUG) std::cout << "Epoch " << e << std::endl;
        // copy bias to the first column
        cudaMemcpy(dVisibleUnitsStates, hBias, examplesNumber * sizeof(float), cudaMemcpyHostToDevice);

        // copy training data to remaining cells
        cudaMemcpy(&dVisibleUnitsStates[examplesNumber],
                   hTrainingData,
                   numVisible * examplesNumber * sizeof(float),
                   cudaMemcpyHostToDevice);

        // calculate positive hidden activation probabilities
        dPositiveHiddenUnitsActivationProbabilities = hiddenActivationProbabilities(dVisibleUnitsStates, examplesNumber);

        if (DEBUG) std::cout << "Calculating hidden unit states by sampling" << std::endl;
        checkCuRandError(__LINE__, curandGenerateUniform(generator, dRandom, examplesNumber * (numHidden + 1)));
        int blockNumber = examplesNumber * (numHidden + 1) / blockSize + 1;
        greaterThan<<<blockNumber, blockSize>>>(dPositiveHiddenUnitsActivationProbabilities, dRandom, dHiddenUnitsStates, examplesNumber * (numHidden + 1));
        if (DEBUG) printDeviceColumnMajorMatrix(dHiddenUnitsStates, examplesNumber, numHidden + 1);

        if (DEBUG) std::cout << "Calculating positive associations" << std::endl;
        dPositiveAssociations = computeAssociations(dVisibleUnitsStates,
                                                    dPositiveHiddenUnitsActivationProbabilities,
                                                    examplesNumber);

        // a negative (reconstruction) phase of the contrastive divergence

        // calculate negative visible probabilities
        dVisibleUnitsActivationProbabilities = visibleActivationProbabilities(dHiddenUnitsStates, examplesNumber);

        if (DEBUG) std::cout << "Fixing visible units activation probabilities by setting bias to the first column" << std::endl;
        cudaMemcpy(dVisibleUnitsActivationProbabilities, hBias, examplesNumber * sizeof(float), cudaMemcpyHostToDevice);

        if (DEBUG) printDeviceColumnMajorMatrix(dVisibleUnitsActivationProbabilities, examplesNumber, numVisible + 1);
        // negative hidden probabilities
        dNegativeHiddenUnitsActivationProbabilities = hiddenActivationProbabilities(
            dVisibleUnitsActivationProbabilities,
            examplesNumber);

        if (DEBUG) std::cout << "Calculating negative associations" << std::endl;
        dNegativeAssociations = computeAssociations(dVisibleUnitsActivationProbabilities,
                                                    dNegativeHiddenUnitsActivationProbabilities,
                                                    examplesNumber);

        if (DEBUG) std::cout << "Updating weights" << std::endl;
        int weightsNumber = (numHidden + 1) * (numVisible + 1);
        blockNumber = weightsNumber / blockSize + 1;
        updateWeight<<<blockNumber, blockSize>>>(dWeights, dPositiveAssociations, dNegativeAssociations, weightsNumber, examplesNumber, learningRate);
        if (DEBUG) printDeviceColumnMajorMatrix(dWeights, numVisible + 1, numHidden + 1);

        if (DEBUG) std::cout << "Calculating error - squares of subtractions" << std::endl;
        blockNumber = examplesNumber / blockSize + 1;
        // for memory efficiency we will write subtraction result to one of the input matrices (dVisibleUnitsStates)
        subAndSquare<<<blockNumber, blockSize>>>(dVisibleUnitsStates, dVisibleUnitsActivationProbabilities, (numVisible + 1) * examplesNumber);
        if (DEBUG) printDeviceColumnMajorMatrix(dVisibleUnitsStates, examplesNumber, (numVisible + 1));

        blockNumber = examplesNumber / 2 / blockSize + 1;
        if (DEBUG) std::cout << "Calculation error - reducing sum" << std::endl;
        sumReduce<<<blockNumber, blockSize>>>(dVisibleUnitsStates, (numVisible + 1) * examplesNumber);
        if (DEBUG) printDeviceColumnMajorMatrix(dVisibleUnitsStates, examplesNumber, (numVisible + 1));

        float hError;
        cudaMemcpy(&hError, dVisibleUnitsStates, sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "Error after epoch " << e + 1 << " is " << hError << std::endl;

        cudaFree(dVisibleUnitsActivationProbabilities);
        cudaFree(dPositiveHiddenUnitsActivationProbabilities);
        cudaFree(dNegativeHiddenUnitsActivationProbabilities);
        cudaFree(dPositiveAssociations);
        cudaFree(dNegativeAssociations);
    }

    cudaFree(dRandom);
    cudaFree(dVisibleUnitsStates);
    cudaFree(dHiddenUnitsStates);

    if (INFO) std::cout << "Learned weights:" << std::endl;
    if (INFO) printDeviceColumnMajorMatrix(dWeights, numVisible + 1, numHidden + 1);
}

float *RBM::hiddenStates(float *hVisible) {
    float *dVisible;
    float *dHidden;
    float *hHidden;
    float *dRandom;

    cudaMalloc(&dVisible, (numVisible + 1) * sizeof(float));
    cudaMalloc(&dHidden, (numHidden + 1) * sizeof(float));

    float bias = 1.0;
    cudaMemcpy(dVisible, &bias, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&dVisible[1], hVisible, numVisible * sizeof(float), cudaMemcpyHostToDevice); // set bias

    dHidden = hiddenActivationProbabilities(dVisible, 1);

    // sampling
    cudaMalloc(&dRandom, (numHidden + 1) * sizeof(float));
    checkCuRandError(__LINE__, curandGenerateUniform(generator, dRandom, numHidden + 1));
    int blockNumber = (numHidden + 1) / blockSize + 1;
    greaterThan<<<blockNumber, blockSize>>>(dHidden, dRandom, dHidden, numHidden + 1);

    hHidden = (float *) malloc(numHidden * sizeof(float));
    cudaMemcpy(hHidden, &dHidden[1], numHidden * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dHidden);
    cudaFree(dVisible);
    cudaFree(dRandom);
    return hHidden;
}
