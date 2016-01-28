#include <iostream>
#include <cstdlib>
#include "utils.h"

void checkCudaError(int line, cudaError_t err) {
    if (err != cudaSuccess) {
        std::cout << "CUDA error at line:" << line << ", status: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCudaError(int line) {
    checkCudaError(line, cudaPeekAtLastError());
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

void printColumnMajorMatrix(float *A, int nrRows, int nrCols) {
    int maxRowsToPrint = 12;
    int maxColsToPrint = 12;

    for (int i = 0; i < (nrRows <= maxRowsToPrint ? nrRows : maxRowsToPrint); ++i) {
        for (int j = 0; j < (nrCols <= maxColsToPrint ? nrCols : maxColsToPrint); ++j)
            std::cout << A[nrRows * j + i] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void printDeviceColumnMajorMatrix(float *dA, int nrRows, int nrCols) {
    int   size = nrRows * nrCols;
    float hA[size];

    checkCudaError(__LINE__, cudaMemcpy(hA, dA, size * sizeof(float), cudaMemcpyDeviceToHost));
    printColumnMajorMatrix(hA, nrRows, nrCols);
}
