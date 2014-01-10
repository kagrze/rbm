#include<iostream>
#include "rbm.h"

int main(int argc, char ** argv) {
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

        std::cout << "Training data:" << std::endl;
        printColumnMajorMatrix(trainingData,6,6);
        rbm.train(trainingData,6,1);

        float visible1 [] = {0,0,0,1,1,0};
        std::cout << "Visible=" << std::endl;
        printColumnMajorMatrix(visible1,1,6);
        float * hidden = rbm.hiddenActivationProbability(visible1);
        std::cout << "Hidden=" << std::endl;
        printColumnMajorMatrix(hidden,1,2);

        float visible2 [] = {1,0,0,0,0,0};
        std::cout << "Visible=" << std::endl;
        printColumnMajorMatrix(visible2,1,6);
        hidden = rbm.hiddenActivationProbability(visible2);
        std::cout << "Hidden=" << std::endl;
        printColumnMajorMatrix(hidden,1,2);

        float visible3 [] = {0,0,0,0,0,1};
        std::cout << "Visible=" << std::endl;
        printColumnMajorMatrix(visible3,1,6);
        hidden = rbm.hiddenActivationProbability(visible3);
        std::cout << "Hidden=" << std::endl;
        printColumnMajorMatrix(hidden,1,2);

        return 0;
}
