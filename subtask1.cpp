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

/*Here is an exact example for subtask 1:

./subtask1 [task of choice 1=convolution, 2=non-linear-activations, 3=subsampling, 4=converting a vector]

1. With convolution, You'll be given N, M, and P. A square matrix of size N and a kernel of size M along with padding value P will be given. All the values will be space seperated and look something like
./subtask1 1 15 5 0 13 73 72 68 82 0 0 0 237 .... 45  4 6 4 8 4 4 5 

2. You'll be given an activation function (0=relu 1=tanh), N and M of the matrix size and matrix itself. 
./subtask1 2 0 15 5 13 73 72 68 82 0 0 0 237 .... 45  4 6 4 8 4 4 5 

3. You will be given pooling function (0=max pool, 1=avg pool), size of matrix N, and the matrix. 

4. You will be given the function (0=sigmoid or 1=softmax), and a vector of numbers.*/
int main(int argc, char** argv){
	if (argc < 2){
		cout << "Please provide the subtask number" << endl;
		return 1;
	}
    int subtask = stoi(argv[1]);
    if (subtask == 1){
        if (argc < 5){
            cout << "Please provide the N, M, P" << endl;
            return 1;
        }
        int N = stoi(argv[2]);
        int M = stoi(argv[3]);
        int P = stoi(argv[4]);
        if (argc < 5 + N * N + M * M){
            cout << "Please provide the matrix and kernel" << endl;
            return 1;
        }
        vector<vector<float>> input_matrix(N, vector<float>(N, 0.0f));
        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++){
                input_matrix[i][j] = stof(argv[5 + i * N + j]);
            }
        }
        vector<vector<float>> kernel(M, vector<float>(M, 0.0f));
        for (int i = 0; i < M; i++){
            for (int j = 0; j < M; j++){
                kernel[i][j] = stof(argv[5 + N * N + i * M + j]);
            }
        }
        cout << "Subtask 1: Convolution" << endl;
        vector<vector<float>> result = convolveMatrix(input_matrix, kernel, 1, P);
        cout << "Result:" << endl;
        for (int i = 0; i < (int) result.size(); i++){
            for (int j = 0; j < (int) result.size(); j++){
                cout << result[i][j] << " ";
            }
            cout << endl;
        }
    }
    else if (subtask == 2){
        if (argc < 5){
            cout << "Please provide the activation function (0=relu 1=tanh), N, M" << endl;
            return 1;
        }
        int activation = stoi(argv[2]);
        int N = stoi(argv[3]);
        // int M = stoi(argv[4]); //! No Idea why they even are giving M for this subtask -> Redundant imo
        if (argc < 5 + N * N){
            cout << "Please provide the matrix" << endl;
            return 1;
        }
        vector<vector<float>> input_matrix(N, vector<float>(N, 0.0f));
        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++){
                input_matrix[i][j] = stof(argv[5 + i * N + j]);
            }
        }
        cout << "Subtask 2: Non-linear activations" << endl;
        string activationName = (activation == 0) ? "ReLU" : "tanh";
        cout << "Activation: " << activationName << endl;
        vector<vector<float>> result;
        if (activation == 0){
            result = reLU(input_matrix);
        }
        else{
            result = tanH(input_matrix);
        }
        cout << "Result:" << endl;
        for (int i = 0; i < (int) result.size(); i++){
            for (int j = 0; j < (int) result.size(); j++){
                cout << result[i][j] << " ";
            }
            cout << endl;
        }
    }
    else if (subtask == 3){
        if (argc < 4){
            cout << "Please provide the pooling function (0=max pool 1=avg pool), N" << endl;
            return 1;
        }
        int pooling = stoi(argv[2]);
        int N = stoi(argv[3]);
        if (argc < 4 + N * N){
            cout << "Please provide the matrix" << endl;
            return 1;
        }
        vector<vector<float>> input_matrix(N, vector<float>(N, 0.0f));
        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++){
                input_matrix[i][j] = stof(argv[4 + i * N + j]);
            }
        }
        cout << "Subtask 3: Subsampling" << endl;
        string poolingName = (pooling == 0) ? "Max Pool" : "Avg Pool";
        cout << "Pooling: " << poolingName << endl;
        vector<vector<float>> result;
        if (pooling == 0){
            result = maxPool(input_matrix, 2, 0);
        }
        else{
            result = averagePool(input_matrix, 2, 0);
        }
        cout << "Result:" << endl;
        for (int i = 0; i < (int) result.size(); i++){
            for (int j = 0; j < (int) result.size(); j++){
                cout << result[i][j] << " ";
            }
            cout << endl;
        }
    }
    else if (subtask == 4){
        if (argc < 3){
            cout << "Please provide the function (0=sigmoid 1=softmax)" << endl;
            return 1;
        }
        int function = stoi(argv[2]);
        //* Note that the vector size is not given in the input
        //* Allocate the vector size based on the number of arguments
        int N = argc - 3;
        if (argc < 3 + N){
            cout << "Please provide the vector" << endl;
            return 1;
        }
        vector<float> input_vector(N, 0.0f);
        for (int i = 0; i < N; i++){
            input_vector[i] = stof(argv[3 + i]);
        }
        cout << "Subtask 4: Converting a vector" << endl;
        string functionName = (function == 0) ? "Sigmoid" : "Softmax";
        cout << "Function: " << functionName << endl;
        vector<float> result;
        if (function == 0){
            result = sigmoid(input_vector);
        }
        else{
            result = softmax(input_vector);
        }
        cout << "Result:" << endl;
        for (int i = 0; i < (int) result.size(); i++){
            cout << result[i] << " ";
        }
        cout << endl;
    }
    else{
        cout << "Invalid subtask number" << endl;
        return 1;
    }
}