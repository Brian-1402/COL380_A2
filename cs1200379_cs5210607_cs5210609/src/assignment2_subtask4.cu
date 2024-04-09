#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <dirent.h>
#include <sys/time.h>
#include <omp.h>

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


__global__ void pool(float* inputMatrix, float* outputMatrix, int channels, int in_height, int in_width, int out_height, int out_width,int pooldim, int pooltype, int stride = 1) {
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

void makeCUDAMatrix(vector<float> data, float*& d_matrix, int totalSize, cudaStream_t stream = 0) {
	cudaMallocAsync(&d_matrix, totalSize * sizeof(float), stream);
	cudaMemcpyAsync(d_matrix, data.data(), totalSize * sizeof(float), cudaMemcpyHostToDevice, stream);
}

void makeBias(std::vector<float> data, float*& d_bias, int totalSize, cudaStream_t stream = 0) {
	cudaMallocAsync(&d_bias, totalSize * sizeof(float), stream);
	cudaMemcpyAsync(d_bias, &data[data.size() - totalSize], totalSize * sizeof(float), cudaMemcpyHostToDevice, stream);
	// cudaDeviceSynchronize();
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

void printCUDAVector(float* vector, int size, cudaStream_t stream = 0) {
	float* host_vector = new float[size];
	cudaMemcpyAsync(host_vector, vector, size * sizeof(float), cudaMemcpyDeviceToHost, stream);
	printVector(host_vector, size);
	delete[] host_vector;
}

void printCUDAMatrix(float* matrix, int channels, int rows, int cols, cudaStream_t stream = 0) {
	float* host_matrix = new float[channels * rows * cols];
	cudaMemcpyAsync(host_matrix, matrix, channels * rows * cols * sizeof(float), cudaMemcpyDeviceToHost, stream);
	print3DMatrix(host_matrix, channels, rows, cols);
	delete[] host_matrix;
}

//Requires arguments to already be in CUDA memory
void forward_prop(float *inputImage, float *outputVector, weights_struct weights, cudaStream_t stream = 0){
	// cout << "Forward prop" << endl;

	float * c1_out, * p1_out, * c2_out, * p2_out, * fc1_out, *fc1_relu_out, * fc2_out;
	// dimensions: (output_channels, height, width)
	cudaMallocAsync(&c1_out, 20 * 24 * 24 * sizeof(float), stream);
	cudaMallocAsync(&p1_out, 20 * 12 * 12 * sizeof(float), stream);
	cudaMallocAsync(&c2_out, 50 * 8 * 8 * sizeof(float), stream);
	cudaMallocAsync(&p2_out, 50 * 4 * 4 * sizeof(float), stream);
	cudaMallocAsync(&fc1_out, 500 * 1 * 1 * sizeof(float), stream);
	cudaMallocAsync(&fc1_relu_out, 500 * 1 * 1 * sizeof(float), stream);
	cudaMallocAsync(&fc2_out, 10 * 1 * 1 * sizeof(float), stream);

	// C1: 45 blocks, 256 threads per block
	convLayer<<<dim3(5,3,3), dim3(4,8,8), 0, stream>>>(inputImage, c1_out, weights.conv1_kernel, weights.conv1_bias, 5, 1, 28, 28, 24, 24, 20, 1);
	// cout << "C1 done" << endl;
	// printCUDAMatrix(c1_out, 20, 24, 24);
	
	// P1: 16 blocks, 180 threads per block
	pool<<<dim3(4,2,2), dim3(5,6,6), 0, stream>>>(c1_out, p1_out, 20, 24, 24, 12, 12, 2, MAXPOOL, 2);
	// cout << "P1 done" << endl;
	// printCUDAMatrix(p1_out, 20, 12, 12);

	// C2: 20 blocks, 160 threads per block
	convLayer<<<dim3(5,2,2), dim3(10,4,4), 0, stream>>>(p1_out, c2_out, weights.conv1_kernel, weights.conv1_bias, 5, 20, 12, 12, 8, 8, 50, 1);
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
	cout << "ReLU done" << endl;
	printCUDAVector(fc1_relu_out, 500);

	// FC2: 4 block, 125 threads per block
	// Here, x axis is input channels
	FCLayer1D<<<dim3(4,1,1), dim3(125,1,1), 0, stream>>>(fc1_relu_out, fc2_out, weights.fc2_kernel, weights.fc2_bias, 1, 500, 1, 1, 1, 1, 10, 1);
	cout << "FC2 done" << endl;
	printCUDAVector(fc2_out, 10);

	// Softmax: 1 block, 10 threads per block
	softmax<<<dim3(1,1,1), dim3(10,1,1), 0, stream>>>(fc2_out, outputVector, 10); //! This would have to be changed
	// cout << "Softmax done" << endl;
	// printCUDAVector(outputVector, 10);

	// cudaMemcpy(outputVector, fc2_out, 10 * sizeof(float), cudaMemcpyDeviceToDevice);
	// printCUDAVector(outputVector, 10);

	//Free memory
	cudaFreeAsync(c1_out, stream);
	cudaFreeAsync(p1_out, stream);
	cudaFreeAsync(c2_out, stream);
	cudaFreeAsync(p2_out, stream);
	cudaFreeAsync(fc1_out, stream);
	cudaFreeAsync(fc1_relu_out, stream);
	cudaFreeAsync(fc2_out, stream);
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

bool comp(pair<int, int> a, pair<int, int> b) {
	return a.first > b.first;
}

vector<int> sorted_indices(float* arr, int size) {
   std::vector<std::pair<int, int>> value_index_pairs;

	// Populate the vector with value-index pairs
	for (int i = 0; i < size; ++i) {
		value_index_pairs.push_back(std::make_pair(arr[i], i));
	}

	// Sort the vector based on values
	std::sort(value_index_pairs.begin(), value_index_pairs.end(), comp);
	std::vector<int> sorted_indices;
	for (const auto& pair : value_index_pairs) {
		sorted_indices.push_back(pair.second);
	}
	return sorted_indices;
}

int main() {
	// Loop over the pre-proc-img directory and perform forward prop on each image
	weights_struct weights = prep_weights();
	const string path = "./pre-proc-img/";
	struct dirent *entry;
  	DIR *dp;
	dp = ::opendir(path.c_str());
	vector<string> images;
	while ((entry = ::readdir(dp))) {
		images.push_back(path + entry->d_name);
	}
	::closedir(dp);

	//iterate through images
	struct timeval start, end;
	gettimeofday(&start, NULL);

    int i = 0;
    cudaStream_t stream[12];
    for (i = 0; i < 12; i++)
        cudaStreamCreate(&stream[i]);
    #pragma omp parallel for num_threads(12) private(i) shared(images, weights)
	    for (i = 0; i < images.size(); i++){
	    	// cout << "Image path: " << images[i] << endl;
	    	// int label = stoi(images[i].substr(25, 1));
            int tid = omp_get_thread_num();

	    	float* inputImage = prep_inputs(images[i]);
	    	// cout << "Input image: " << endl;
	    	// printCUDAMatrix(inputImage, 1, 28, 28);

	    	float* outputVector;
	    	float* host_outputVector = new float[10];

	    	cudaMallocAsync(&outputVector, 10 * sizeof(float),stream[tid]);
	    	forward_prop(inputImage, outputVector, weights, stream[tid]);

	    	cudaMemcpyAsync(host_outputVector, outputVector, 10 * sizeof(float), cudaMemcpyDeviceToHost, stream[tid]);

	    	printVector(host_outputVector, 10);
	    	int pred = argmax(host_outputVector, 10);
	    	// cout << "Label: " << label << "; ";
	    	cout << "Predicted: " << pred << endl;

	    	delete[] host_outputVector;
	    	cudaFreeAsync(inputImage, stream[tid]);
	    	cudaFreeAsync(outputVector, stream[tid]);
	    }
	gettimeofday(&end, NULL);
	cout << "Time taken: " << (end.tv_sec - start.tv_sec) * 1000000LL + (end.tv_usec - start.tv_usec) << " microseconds" << endl;
	return 0;
}
