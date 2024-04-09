#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <string>

using namespace std;

#define MAXPOOL 0
#define AVGPOOL 1


// Kernel dimensions explained:
// 1) output channel (sets of filters. Each filter corresponds to an output channel)
// 2) input channel (sets of 2d kernel matrices, together called a filter. Each kernel corresponds to each input channel in the input)
// 3) kernel row
// 4) kernel column
__global__ void convLayer(float* inputMatrix, float* outputMatrix, float* kernel, float* bias, int kernel_size, int input_channels, int in_height, int in_width, int out_height, int out_width, int output_channels, int stride = 1) {
	int k = blockIdx.x * blockDim.x + threadIdx.x; // channel of output image
	int i = blockIdx.y * blockDim.y + threadIdx.y; // height of output image
	int j = blockIdx.z * blockDim.z + threadIdx.z; // width of output image
	float val = 0.0f;

	if (i < out_height && j < out_width && k < output_channels) {
		for (int a = 0; a < input_channels; a++) // Iterate through input channels
			for (int b = 0; b < kernel_size; b++) // Iterate through input rows
				for (int c = 0; c < kernel_size; c++) // Iterate through input columns
					val += inputMatrix[(a * in_height + i * stride + b) * in_width + j * stride + c] * kernel[((k * input_channels + a) * kernel_size + b) * kernel_size + c];

		outputMatrix[(k * out_height + i) * out_width + j] = val + bias[k];
	}
}

__global__ void FCLayer1D(float* inputMatrix, float* outputMatrix, float* kernel, float* bias, int kernel_size, int input_channels, int in_height, int in_width, int out_height, int out_width, int output_channels, int stride = 1) {
	// int k = blockIdx.x * blockDim.x + threadIdx.x; // channel of output image
	int a = blockIdx.x * blockDim.x + threadIdx.x; // channel of input image
	int i = blockIdx.y * blockDim.y + threadIdx.y; // height of output image
	int j = blockIdx.z * blockDim.z + threadIdx.z; // width of output image
	float val = 0.0f;

	if (i < out_height && j < out_width && a < input_channels) {
		for (int k = 0; k < output_channels; a++){ // Iterate through output channels
			for (int b = 0; b < kernel_size; b++) // Iterate through input rows
				for (int c = 0; c < kernel_size; c++) // Iterate through input columns
					val += inputMatrix[(a * in_height + i * stride + b) * in_width + j * stride + c] * kernel[((k * input_channels + a) * kernel_size + b) * kernel_size + c];

			outputMatrix[(k * out_height + i) * out_width + j] = val + bias[k];
		}
	}
}


__global__ void pool(float* inputMatrix, float* outputMatrix, int channels, int in_height, int in_width, int out_height, int out_width, int pooldim, int pooltype, int stride = 1) {
	int k = blockIdx.x * blockDim.x + threadIdx.x; // channel of output image
	int i = blockIdx.y * blockDim.y + threadIdx.y; // height of output image
	int j = blockIdx.z * blockDim.z + threadIdx.z; // width of output image

	if (i < out_height && j < out_width && k < channels) {
		float val = 0.0f;
		if (pooltype == MAXPOOL)
			val = inputMatrix[(k * in_height + i * stride) * in_width + j * stride];

		for (int b = 0; b < pooldim; b++) // height
			for (int c = 0; c < pooldim; c++) { // width
				if (pooltype == MAXPOOL)
					val = max(val, inputMatrix[(k * in_height + i * stride + b) * in_width + j * stride + c]);
				else if (pooltype == AVGPOOL)
					val += inputMatrix[(k * in_height + i * stride + b) * in_width + j * stride + c];
			}

		if (pooltype == AVGPOOL)
			val = val / (pooldim * pooldim);

		outputMatrix[(k * out_height + i) * out_width + j] = val;
	}
}


__global__ void padMatrix(float* inputMatrix, float* outputMatrix, int height, int width, int channels, int padding) {
	int k = blockIdx.x * blockDim.x + threadIdx.x; // channel of output image
	int i = blockIdx.y * blockDim.y + threadIdx.y; // height of output image
	int j = blockIdx.z * blockDim.z + threadIdx.z; // width of output image

	if (i < height && j < width && k < channels)
		outputMatrix[(k * (height + 2 * padding) + i + padding) * (width + 2 * padding) + j + padding] = inputMatrix[(k * height + i) * width + j];
}


// Apply ReLU activation to each element of the matrix
__global__ void ReLU(float* inputMatrix, float* outputMatrix, int size) {
	int k = blockIdx.x * blockDim.x + threadIdx.x; // index in the flattened array

	if (k < size)
		outputMatrix[k] = max(0.0f, inputMatrix[k]);
}

//! softmax value of each element of the vector
__global__ void softmax(float* inputVector, float* outputVector, int size) {
	// int k = blockIdx.x * blockDim.x + threadIdx.x; // index in the flattened array
	int k = threadIdx.x; // Only made to run in one block

	if (k < size)
		outputVector[k] = exp(inputVector[k]);

	__syncthreads();

	// The below code forces a serial computation onto just 1 thread and makes the others wait
	// More efficient way to do this is to use a reduction kernel, or split into two parts
	__shared__ float sum;
	sum = 0;
	if(k == 0) {
		for(int j = 0; j < size; j++)
			sum += outputVector[j];
	}
	__syncthreads();

	outputVector[k] /= sum;
}

// softmax value of each element of the vector
void softmax_serial(float* inputVector, float* outputVector, int size) {
    float sum = 0.0f;

    for (int i = 0; i < size; i++) {
        outputVector[i] = exp(inputVector[i]);
        sum += outputVector[i];
    }

    for (int i = 0; i < size; i++) {
        outputVector[i] /= sum;
    }
}

//Memory copying functions

void readFile(std::vector<float>& data, const std::string& filename) {
	std::ifstream file(filename);
	float value;
	while (file >> value)
		data.push_back(value);
}

void makeCUDAMatrix(vector<float> data, float*& d_matrix, int totalSize) {
	cudaMalloc(&d_matrix, totalSize * sizeof(float));
	cudaMemcpy(d_matrix, data.data(), totalSize * sizeof(float), cudaMemcpyHostToDevice);
}

void makeBias(std::vector<float> data, float*& d_bias, int totalSize) {
	cudaMalloc(&d_bias, totalSize * sizeof(float));
	cudaMemcpy(d_bias, &data[data.size() - totalSize], totalSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
}

float* prep_inputs(string filename){
	// cout << "Reading input data" << endl;
	std::vector<float> data;
	readFile(data, filename);
	float* d_input;
	makeCUDAMatrix(data, d_input, 1 * 28 * 28);
	return d_input;
}

struct weights_struct {
	float* conv1_kernel;
	float* conv1_bias;
	float* conv2_kernel;
	float* conv2_bias;
	float* fc1_kernel;
	float* fc1_bias;
	float* fc2_kernel;
	float* fc2_bias;
};

// kernel dim: (filters, channels, kernel_dim)
struct weights_struct prep_weights() {
	// cout << "Reading weights" << endl;
	weights_struct weights;
	std::vector<float> data;
	
	// cout << "Conv1: " << endl;
	readFile(data, "./weights/conv1.txt");
	makeCUDAMatrix(data, weights.conv1_kernel, 20 * 1 * 5 * 5);
	makeBias(data, weights.conv1_bias, 20);

	// cout << "Conv2: " << endl;
	readFile(data, "./weights/conv2.txt");
	makeCUDAMatrix(data, weights.conv2_kernel, 50 * 20 * 5 * 5);
	makeBias(data, weights.conv2_bias, 50);

	// cout << "FC1: " << endl;
	readFile(data, "./weights/fc1.txt");
	makeCUDAMatrix(data, weights.fc1_kernel, 500 * 50 * 4 * 4);
	makeBias(data, weights.fc1_bias, 500);

	// cout << "FC2: " << endl;
	readFile(data, "./weights/fc2.txt");
	makeCUDAMatrix(data, weights.fc2_kernel, 10 * 500 * 1 * 1);
	makeBias(data, weights.fc2_bias, 10);
	// cout << "Done reading weights" << endl;
	return weights;
}


// Helper functions for debugging
void printVector(float* vector, int size) {
	for (int i = 0; i < size; i++)
		cout << vector[i] << " ";
	cout << endl;
}

void print3DMatrix(float* matrix, int channels, int rows, int cols) {
	for (int k = 0; k < channels; k++){
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++)
				cout << matrix[(k * rows + i) * cols + j ] << " ";
			cout << endl;
		}
		cout << "-----------------------------------" << endl;
	}
	cout << "====================================\n\n" << endl;
}

void printCUDAVector(float* vector, int size) {
	float* host_vector = new float[size];
	cudaMemcpy(host_vector, vector, size * sizeof(float), cudaMemcpyDeviceToHost);
	printVector(host_vector, size);
	delete[] host_vector;
}

void printCUDAMatrix(float* matrix, int channels, int rows, int cols) {
	float* host_matrix = new float[channels * rows * cols];
	cudaMemcpy(host_matrix, matrix, channels * rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
	print3DMatrix(host_matrix, channels, rows, cols);
	delete[] host_matrix;
}

//Requires arguments to already be in CUDA memory
void forward_prop(float *inputImage, float *outputVector, weights_struct weights, cudaStream_t stream){
	// cout << "Forward prop" << endl;

	float * c1_out, * p1_out, * c2_out, * p2_out, * fc1_out, *fc1_relu_out, * fc2_out;
	// dimensions: (output_channels, height, width)
	cudaMalloc(&c1_out, 20 * 24 * 24 * sizeof(float));
	cudaMalloc(&p1_out, 20 * 12 * 12 * sizeof(float));
	cudaMalloc(&c2_out, 50 * 8 * 8 * sizeof(float));
	cudaMalloc(&p2_out, 50 * 4 * 4 * sizeof(float));
	cudaMalloc(&fc1_out, 500 * 1 * 1 * sizeof(float));
	cudaMalloc(&fc1_relu_out, 500 * 1 * 1 * sizeof(float));
	cudaMalloc(&fc2_out, 10 * 1 * 1 * sizeof(float));

	// C1: 45 blocks, 256 threads per block
	convLayer<<<dim3(5,3,3), dim3(4,8,8), 0, stream>>>(inputImage, c1_out, weights.conv1_kernel, weights.conv1_bias, 5, 1, 28, 28, 24, 24, 20, 1);
	// cout << "C1 done" << endl;
	// printCUDAMatrix(c1_out, 20, 24, 24);
	
	// P1: 16 blocks, 180 threads per block
	pool<<<dim3(4,2,2), dim3(5,6,6), 0, stream>>>(c1_out, p1_out, 20, 24, 24, 12, 12, 2, MAXPOOL, 2);
	// cout << "P1 done" << endl;
	// printCUDAMatrix(p1_out, 20, 12, 12);

	// C2: 10 blocks, 320 threads per block
	convLayer<<<dim3(5,2,1), dim3(10,4,8), 0, stream>>>(p1_out, c2_out, weights.conv1_kernel, weights.conv1_bias, 5, 20, 12, 12, 8, 8, 50, 1);
	// cout << "C2 done" << endl;
	// printCUDAMatrix(c2_out, 50, 8, 8);

	// P2: 5 blocks, 160 threads per block
	pool<<<dim3(5,1,1), dim3(10,4,4), 0, stream>>>(c2_out, p2_out, 50, 8, 8, 4, 4, 2, MAXPOOL, 2);
	// cout << "P2 done" << endl;
	// printCUDAMatrix(p2_out, 50, 4, 4);

	// FC1: 4 blocks, 125 threads per block
	convLayer<<<dim3(4,1,1), dim3(125,1,1), 0, stream>>>(p2_out, fc1_out, weights.fc1_kernel, weights.fc1_bias, 4, 50, 4, 4, 1, 1, 500, 1);
	// cout << "FC1 done" << endl;
	// printCUDAMatrix(fc1_out, 500, 1, 1);

	// Relu1: 4 blocks, 125 threads per block
	ReLU<<<dim3(4,1,1), dim3(125,1,1), 0, stream>>>(fc1_out, fc1_relu_out, 500);
	// cout << "ReLU done" << endl;
	// printCUDAVector(fc1_relu_out, 500);

	// FC2: 4 block, 125 threads per block
	// Here, x axis is input channels
	FCLayer1D<<<dim3(4,1,1), dim3(125,1,1), 0, stream>>>(fc1_relu_out, fc2_out, weights.fc2_kernel, weights.fc2_bias, 1, 500, 1, 1, 1, 1, 10, 1);
	// cout << "FC2 done" << endl;
	// printCUDAVector(fc2_out, 10);

	// Softmax: 1 block, 10 threads per block
	softmax<<<dim3(1,1,1), dim3(10,1,1), 0, stream>>>(fc2_out, outputVector, 10); //! This would have to be changed
	// cout << "Softmax done" << endl;
	// printCUDAVector(outputVector, 10);

	// cudaMemcpy(outputVector, fc2_out, 10 * sizeof(float), cudaMemcpyDeviceToDevice);
	// printCUDAVector(outputVector, 10);

	//Free memory
	cudaFree(c1_out);
	cudaFree(p1_out);
	cudaFree(c2_out);
	cudaFree(p2_out);
	cudaFree(fc1_out);
	cudaFree(fc1_relu_out);
	cudaFree(fc2_out);
}

int argmax(float* vector, int size) {
	float max = vector[0];
	int index = 0;
	for (int i = 1; i < size; i++) {
		if (vector[i] > max) {
			max = vector[i];
			index = i;
		}
	}
	return index;
}

vector<int> sorted_indices(float* arr, int size) {
   std::vector<std::pair<int, int>> value_index_pairs;

	// Populate the vector with value-index pairs
	for (int i = 0; i < size; ++i) {
		value_index_pairs.push_back(std::make_pair(arr[i], i));
	}

	// Sort the vector based on values
	std::sort(value_index_pairs.begin(), value_index_pairs.end(), 
			  [](const auto& lhs, const auto& rhs) {
				  return lhs.first > rhs.first; // Compare in descending order
			  });
	std::vector<int> sorted_indices;
	for (const auto& pair : value_index_pairs) {
		sorted_indices.push_back(pair.second);
	}
	return sorted_indices;
}

int main() {
	// Loop over the pre-proc-img directory and perform forward prop on each image
	weights_struct weights = prep_weights();

	vector<string> images;
	for (const auto& dirEntry : std::filesystem::recursive_directory_iterator("./pre-proc-img/")){
		images.push_back(dirEntry.path().string());
	}
	//iterate through images
	for (int i = 0; i < images.size(); i++){
	// for (int i = 0; i < 1; i++){
		cout << "Image path: " << images[i] << endl;
		int label = stoi(images[i].substr(25, 1));
		string filename = images[i].substr(15, images[i].length() - 15);
		// cout << "Filename: " << filename << endl;

		float* inputImage = prep_inputs(images[i]);
		// cout << "Input image: " << endl;
		// printCUDAMatrix(inputImage, 1, 28, 28);

		float* outputVector;
		float* host_outputVector = new float[10];

		cudaMalloc(&outputVector, 10 * sizeof(float));
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		forward_prop(inputImage, outputVector, weights, stream);

		cudaMemcpy(host_outputVector, outputVector, 10 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaStreamSynchronize(stream);
		cudaStreamDestroy(stream);
		cudaDeviceSynchronize();

		printVector(host_outputVector, 10);
		int pred = argmax(host_outputVector, 10);
		vector<int> sorted_indices2 = sorted_indices(host_outputVector, 10);

		// cout << "Label: " << label << "; ";
		// cout << "Predicted: " << pred << endl;

		std::string directory = "./output/";
		std::string out_filename = directory + filename;

		std::ofstream outputFile(filename);
		if (!outputFile.is_open()) {
			std::cerr << "Error opening the file." << std::endl;
			return 1;
		}

		// Write each element of the array to the file
		for (int i = 0; i < 5; ++i) {
			outputFile << sorted_indices2[i] << std::endl;
		}

		// Close the file
		outputFile.close();

		delete[] host_outputVector;
		cudaFree(inputImage);
		cudaFree(outputVector);
	}
}
