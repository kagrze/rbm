#include <cublas_v2.h>
#include <curand.h>

void printColumnMajorMatrix(float *A, int nrRows, int nrCols);

class RBM {
private:
	int numVisible;
	int numHidden;
	float learningRate;
	float *dWeights; // column-major matrix of dim numVisible+1 x numHidden+1 (+1 because of bias)
	cublasHandle_t handle;
	curandGenerator_t generator;
	int blockSize;

	float * hiddenActivationProbability(float * dVisibleUnitsStates, int examplesNumber);
	float * visibleActivationProbability(float * dHiddenUnitsStates, int examplesNumber);
	float * computeAssociations(float * dVisibleUnitsActivationProbabilities, float * dHiddenUnitsActivationProbabilities, int examplesNumber);
public:
	RBM(int visibleNumber, int hiddenNumber, float rate);
	~RBM();
	//hTrainingData is a matrix of BOOLEAN values (true is represented as 1; false is represented by 0)
	//each row is a training example consisting of the states of visible units
	//matrix is written to array in column-major order
	void train(float * hTrainingData, int examplesNumber, int maxEpochs);
	float * hiddenActivationProbability(float * hVisible);
};
