#include <iostream>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include "rbm.h"

using namespace std;

int reverseInt (int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

float * loadMNISTDataSet(string fileName) {
	string filePath(getenv("HOME"));
	filePath.append("/mnist/");
	filePath.append(fileName);
    ifstream file (filePath.c_str());

    float * mnistDataSet;
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
        cout << "Loading "<< number_of_images << " images of dimensions " << n_rows << "x" << n_cols << " from " << filePath <<endl;

        mnistDataSet = (float*) malloc(n_rows*n_cols*number_of_images*sizeof(float));

        unsigned char temp=0;
        float flag=0;
        int pixelIdx;
        int index;
        for (int e=0; e<number_of_images; e++) {
			for (int r=0; r<n_rows; r++) {
				for (int c=0; c<n_cols; c++) {
					file.read((char*)&temp,sizeof(temp));
					flag = temp>127;
					pixelIdx = r*n_cols+c;
					index=pixelIdx*number_of_images+e; //MNIST images are written row-major; CuBLAS matrices need to be column-major
					if (flag)
						mnistDataSet[index]=1;
					else
						mnistDataSet[index]=0;
				}
			}
        }
        cout << "Images loaded into RAM" << endl;
    }

    return mnistDataSet;
}

void testReadMnistImages() {
	float * mnistDataSet = loadMNISTDataSet("train-images-idx3-ubyte");

    int pixelIdx;
    int index;
    for (int e=0; e<5; e++) {
		for (int r=0; r<28; r++) {
			for (int c=0; c<28; c++) {
				pixelIdx = r*28+c;
				index=pixelIdx*60000+e;
				if (mnistDataSet[index] == 1)
					cout << setw(2) << "X";
				else
					cout << setw(2) << "";
			}
			cout << endl;
		}
		cout << endl;
    }
}

void testMNIST() {
	RBM rbm (28*28,500,0.1);

	float * trainingData = loadMNISTDataSet("train-images-idx3-ubyte");

	rbm.train(trainingData,60000,5000);

	float * testData = loadMNISTDataSet("test-images-idx3-ubyte");

	float * hidden = rbm.hiddenStates(&testData[0]);
}

void testSmallDataset() {
	RBM rbm (6,2,0.1);
	//column-major version of matrix:
	// [1,1,1,0,0,0,
	//  1,0,1,0,0,0,
	//  1,1,1,0,0,0,
	//  0,0,1,1,1,0,
	//  0,0,1,1,0,0]
	float trainingData[] = {
			1,1,1,0,0,0,
			1,0,1,0,0,0,
			1,1,1,1,1,1,
			0,0,0,1,1,1,
			0,0,0,1,0,1,
			0,0,0,0,0,0};

	cout << "Training data:" << endl;
	printColumnMajorMatrix(trainingData,6,6);
	rbm.train(trainingData,6,5000);

	float visible1 [] = {0,0,0,1,1,0};
	cout << "Visible=" << endl;
	printColumnMajorMatrix(visible1,1,6);
	float * hidden = rbm.hiddenStates(visible1);
	cout << "Hidden=" << endl;
	printColumnMajorMatrix(hidden,1,2);

	float visible2 [] = {1,0,0,0,0,0};
	cout << "Visible=" << endl;
	printColumnMajorMatrix(visible2,1,6);
	hidden = rbm.hiddenStates(visible2);
	cout << "Hidden=" << endl;
	printColumnMajorMatrix(hidden,1,2);

	float visible3 [] = {0,0,0,0,0,1};
	cout << "Visible=" << endl;
	printColumnMajorMatrix(visible3,1,6);
	hidden = rbm.hiddenStates(visible3);
	cout << "Hidden=" << endl;
	printColumnMajorMatrix(hidden,1,2);
}

int main(int argc, char ** argv) {
	testSmallDataset();
//	testReadMnistImages();
//	testMNIST();
	return 0;
}
