#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <complex>
#include <chrono>
using namespace std;

// Convolution function with stride and variable padding
// put padding = 0 for no padding
vector<vector<float>> convolveMatrix(const vector<vector<float>>& inputMatrix, const vector<vector<float>>& kernel, int stride = 0, int padding = 0) {
    int inputSize = inputMatrix.size();
    int kernelSize = kernel.size();
    int paddedSize = inputSize + 2 * padding;
    vector<vector<float>> paddedMatrix(paddedSize, vector<float>(paddedSize, 0.0f));

    // Copy inputMatrix into paddedMatrix
    for (int i = 0; i < inputSize; i++) {
        for (int j = 0; j < inputSize; j++) {
            paddedMatrix[i + padding][j + padding] = inputMatrix[i][j];
        }
    }

    int outputSize = (paddedSize - kernelSize) / stride + 1;
    vector<vector<float>> outputMatrix(outputSize, vector<float>(outputSize, 0.0f));

    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            for (int k = 0; k < kernelSize; k++) {
                for (int l = 0; l < kernelSize; l++) {
                    outputMatrix[i][j] += paddedMatrix[i * stride + k][j * stride + l] * kernel[k][l];
                }
            }
        }
    }

    return outputMatrix;
}



// Apply ReLU activation to each element of the matrix
vector<vector<float>> reLU(const vector<vector<float>>& inputMatrix) {
    int numRows = inputMatrix.size();
    int numCols = inputMatrix[0].size();
    vector<vector<float>> outputMatrix(numRows, vector<float>(numCols, 0.0f));

    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            outputMatrix[i][j] = max(0.0f,inputMatrix[i][j]);
        }
    }

    return outputMatrix;
}

// Apply tanh activation to each element of the matrix
vector<vector<float>> tanH(const vector<vector<float>>& inputMatrix) {
    int numRows = inputMatrix.size();
    int numCols = inputMatrix[0].size();
    vector<vector<float>> outputMatrix(numRows, vector<float>(numCols, 0.0f));

    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            outputMatrix[i][j] = tanh(inputMatrix[i][j]);
        }
    }

    return outputMatrix;
}

// max pooling function with stride and variable padding
// put padding = 0 for no padding
vector<vector<float>> maxPool(const vector<vector<float>>& inputMatrix, int stride = 0, int padding = 0) {
    int inputSize = inputMatrix.size();
    int paddedSize = inputSize + 2 * padding;
    vector<vector<float>> paddedMatrix(paddedSize, vector<float>(paddedSize, 0.0f));

    // Copy inputMatrix into paddedMatrix
    for (int i = 0; i < inputSize; i++) {
        for (int j = 0; j < inputSize; j++) {
            paddedMatrix[i + padding][j + padding] = inputMatrix[i][j];
        }
    }

    int outputSize = (paddedSize - 2) / stride + 1;
    vector<vector<float>> outputMatrix(outputSize, vector<float>(outputSize, 0.0f));

    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            float maxVal = paddedMatrix[i * stride][j * stride];
            for (int k = 0; k < 2; k++) {
                for (int l = 0; l < 2; l++) {
                    maxVal = max(maxVal, paddedMatrix[i * stride + k][j * stride + l]);
                }
            }
            outputMatrix[i][j] = maxVal;
        }
    }

    return outputMatrix;
}

// average pooling function with stride and variable padding
// put padding = 0 for no padding
vector<vector<float>> averagePool(const vector<vector<float>>& inputMatrix, int stride = 0, int padding = 0) {
    int inputSize = inputMatrix.size();
    int paddedSize = inputSize + 2 * padding;
    vector<vector<float>> paddedMatrix(paddedSize, vector<float>(paddedSize, 0.0f));

    // Copy inputMatrix into paddedMatrix
    for (int i = 0; i < inputSize; i++) {
        for (int j = 0; j < inputSize; j++) {
            paddedMatrix[i + padding][j + padding] = inputMatrix[i][j];
        }
    }

    int outputSize = (paddedSize - 2) / stride + 1;
    vector<vector<float>> outputMatrix(outputSize, vector<float>(outputSize, 0.0f));

    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 2; k++) {
                for (int l = 0; l < 2; l++) {
                    sum += paddedMatrix[i * stride + k][j * stride + l];
                }
            }
            outputMatrix[i][j] = sum / 4;
        }
    }

    return outputMatrix;
}

// sigmoid value of each element of the vector
vector<float> sigmoid(const vector<float>& inputVector) {
    int size = inputVector.size();
    vector<float> outputVector(size, 0.0f);

    for (int i = 0; i < size; i++) {
        outputVector[i] = 1 / (1 + exp(-inputVector[i]));
    }

    return outputVector;
}

// softmax value of each element of the vector
vector<float> softmax(const vector<float>& inputVector) {
    int size = inputVector.size();
    vector<float> outputVector(size, 0.0f);
    float sum = 0.0f;

    for (int i = 0; i < size; i++) {
        outputVector[i] = exp(inputVector[i]);
        sum += outputVector[i];
    }

    for (int i = 0; i < size; i++) {
        outputVector[i] /= sum;
    }

    return outputVector;
}
