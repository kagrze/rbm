#ifndef UTILS_H_
#define UTILS_H_

#include <cublas_v2.h>
#include <curand.h>

void checkCudaError(int line, cudaError_t err);
void checkCudaError(int line);
void checkCuBlasError(int line, cublasStatus_t stat);
void checkCuRandError(int line, curandStatus_t stat);
void printColumnMajorMatrix(float *A, int nrRows, int nrCols);
void printDeviceColumnMajorMatrix(float *dA, int nrRows, int nrCols);

#endif  // UTILS_H_
