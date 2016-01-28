#ifndef RBM_H_
#define RBM_H_

#include <cublas_v2.h>
#include <curand.h>

class RBM {
 private:
    int    numVisible;
    int    numHidden;
    float  learningRate;
    float *dWeights; // column-major matrix of dim numVisible+1 x numHidden+1 (+1 because of bias)
    cublasHandle_t    handle;
    curandGenerator_t generator;

    float *hiddenActivationProbabilities(float *dVisibleUnitsStates, int examplesNumber);
    float *visibleActivationProbabilities(float *dHiddenUnitsStates, int examplesNumber);
    float *computeAssociations(float *dVisibleUnitsActivationProbabilities,
                               float *dHiddenUnitsActivationProbabilities,
                               int    examplesNumber);

 public:
    RBM(int visibleNumber, int hiddenNumber, float rate);
    ~RBM();
    // hTrainingData is a matrix of BOOLEAN values (true is represented as 1; false is represented by 0)
    // each row is a training example consisting of the states of visible units
    // matrix is written to array in column-major order
    void train(float *hTrainingData, int examplesNumber, int maxEpochs);
    float *hiddenStates(float *hVisible);
};

#endif  // RBM_H_
