__global__ void sigmoid(float *dInputArray, float *dOutputArray, int arraySize);
__global__ void greaterThan(float *dLHS, float *dRHS, float *dOutputArray, int arraySize);
__global__ void updateWeight(float *dWeights,
                             float *dPositiveAssociation,
                             float *dNegativeAssociation,
                             int    weightsNumber,
                             int    examplesNumber,
                             float  learningRate);
__global__ void subAndSquare(float *a, float *b, int size);
__global__ void sumReduce(float *array, int size);
