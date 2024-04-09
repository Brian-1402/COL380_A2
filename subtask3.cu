#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <filesystem>
using namespace std;

#define MAXPOOL 0
#define AVGPOOL 1


// Kernel dimensions explained:
// 1) output channel (sets of filters. Each filter corresponds to an output channel)
// 2) input channel (sets of 2d kernel matrices, together called a filter. Each kernel corresponds to each input channel in the input)
// 3) kernel row
// 4) kernel column
__global__ void convLayer(float* inputMatrix, float* outputMatrix, float* kernel, float* bias, int kernel_size, int input_channels, int height, int width, int output_channels, int stride = 1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // height of output image
    int j = blockIdx.y * blockDim.y + threadIdx.y; // width of output image
    int k = blockIdx.z * blockDim.z + threadIdx.z; // channel of output image
    float val = 0.0f;

    if (i < height && j < width && k < output_channels) {
        for (int a = 0; a < input_channels; a++) // Iterate through input channels
            for (int b = 0; b < kernel_size; b++) // Iterate through input rows
                for (int c = 0; c < kernel_size; c++) // Iterate through input columns
                    val += inputMatrix[(a * height + i * stride + b) * width + j * stride + c] * kernel[((k * input_channels + a) * kernel_size + b) * kernel_size + c];

        outputMatrix[(k * height + i) * width + j] = val + bias[k];
    }
}


__global__ void pool(float* inputMatrix, float* outputMatrix, int height, int width, int channels, int pooldim, int pooltype, int stride = 1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // height of output image
    int j = blockIdx.y * blockDim.y + threadIdx.y; // width of output image
    int k = blockIdx.z * blockDim.z + threadIdx.z; // channel of output image
    float val = 0.0f;

    if (i < height && j < width && k < channels) {
        if (pooltype == MAXPOOL)
            val = inputMatrix[(k * height + i * stride) * width + j * stride];

        for (int b = 0; b < pooldim; b++)
            for (int c = 0; c < pooldim; c++)
                if (pooltype == MAXPOOL)
                    val = max(val, inputMatrix[(k * height + i * stride + b) * width + j * stride + c]);
                else if (pooltype == AVGPOOL)
                    val += inputMatrix[(k * height + i * stride + b) * width + j * stride + c];

        if (pooltype == AVGPOOL)
            val = val / (pooldim * pooldim);

        outputMatrix[(k * height + i) * width + j] = val;
    }
}


__global__ void padMatrix(float* inputMatrix, float* outputMatrix, int height, int width, int channels, int padding) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // height of input image
    int j = blockIdx.y * blockDim.y + threadIdx.y; // width of input image
    int k = blockIdx.z * blockDim.z + threadIdx.z; // channel of input image

    if (i < height && j < width && k < channels) {
        outputMatrix[(k * (height + 2 * padding) + i + padding) * (width + 2 * padding) + j + padding] = inputMatrix[(k * height + i) * width + j];
    }
}


// Apply ReLU activation to each element of the matrix
__global__ void ReLU(float* inputMatrix, float* outputMatrix, int size) {
    int k = blockIdx.x * blockDim.x + threadIdx.x; // index in the flattened array

    if (k < size) {
        outputMatrix[k] = max(0.0f, inputMatrix[k]);
    }
}

//! softmax value of each element of the vector
__global__ void softmax(float* inputVector, float* outputVector, int size) {
    int k = blockIdx.x * blockDim.x + threadIdx.x; // index in the flattened array

    if (k < size) {
        outputVector[k] = exp(inputVector[k]);
    }

    __syncthreads();

    // The below code forces a serial computation onto just 1 thread and makes the others wait
    // More efficient way to do this is to use a reduction kernel, or split into two parts
    __shared__ float sum;
    if(threadIdx.x == 0) {
        sum = 0;
        for(int j = 0; j < size; j++)
            sum += outputVector[j];
    }
    __syncthreads();

    outputVector[k] /= sum;
}


#define NUM_FILTERS 0
#define NUM_CHANNELS 1
#define KERNEL_DIM 2

void makeKernel(vector<float>& data, float*& d_kernel, int* dim) {
    int totalSize = dim[NUM_FILTERS] * dim[NUM_CHANNELS] * dim[KERNEL_DIM] * dim[KERNEL_DIM];
    float* kernel = new float[totalSize];

    int index = 0;
    for (int i = 0; i < dim[NUM_FILTERS]; ++i) {
        for (int j = 0; j < dim[NUM_CHANNELS]; ++j) {
            for (int k = 0; k < dim[KERNEL_DIM]; ++k) {
                for (int l = 0; l < dim[KERNEL_DIM]; ++l) {
                    int idx = (((i * dim[NUM_CHANNELS] + j) * dim[KERNEL_DIM] + k) * dim[KERNEL_DIM]) + l;
                    kernel[idx] = data[index++];
                }
            }
        }
    }

    cudaMalloc(&d_kernel, totalSize * sizeof(float));
    cudaMemcpy(d_kernel, kernel, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    delete[] kernel;
}

// dim has 3 elements: (filters, channels, kernel_dim) 
void makeBias(std::vector<float>& data, float*& d_bias, int* dim) {
    cudaMalloc(&d_bias, dim[NUM_FILTERS] * sizeof(float));
    cudaMemcpy(d_bias, &data[data.size() - dim[NUM_FILTERS]], dim[NUM_FILTERS] * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

// dim has 3 elements: (channels, height, width) [in our case height = width = 28]
void makeInput(vector<float>& data, float*& d_input, int* dim) {
    int totalSize = dim[0] * dim[1] * dim[2];
    float* input = new float[totalSize];

    int index = 0;
    for (int k = 0; k < dim[0]; ++k) {
        for (int i = 0; i < dim[1]; ++i) {
            for (int j = 0; j < dim[2]; ++j) {
                int idx = (k * dim[1] + i) * dim[2] + j;
                input[idx] = data[index++];
            }
        }
    }

    cudaMalloc(&d_input, totalSize * sizeof(float));
    cudaMemcpy(d_input, input, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    delete[] input;
}
/*

// Usage of the read functions
int main() {
	std::vector<float> data;
	readFile(data, "conv1.txt");

	float**** kernel;
	to4D(data, kernel);

	float* bias;
	to1D(data, bias);
}

*/
void readFile(std::vector<float>& data, const std::string& filename) {
	std::ifstream file(filename);
	float value;
	while (file >> value) {
		data.push_back(value);
	}
}

float* prep_inputs(string filename){
    // Read input data
    std::vector<float> data;
    readFile(data, filename);
    int dim[3] = {1, 28, 28};
    float* d_input;
    makeInput(data, d_input, dim);
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
	weights_struct weights;
	std::vector<float> data;
	readFile(data, "./weights/conv1.txt");
	int dim[3] = {20, 1, 5};
	makeKernel(data, weights.conv1_kernel, dim);
	makeBias(data, weights.conv1_bias, dim);
	readFile(data, "./weights/conv2.txt");
	dim[0] = 50;
	dim[1] = 20;
	dim[2] = 5;
	makeKernel(data, weights.conv2_kernel, dim);
	makeBias(data, weights.conv2_bias, dim);
	readFile(data, "./weights/fc1.txt");
	dim[0] = 500;
	dim[1] = 50;
	dim[2] = 4;
	makeKernel(data, weights.fc1_kernel, dim);
	makeBias(data, weights.fc1_bias, dim);
	readFile(data, "./weights/fc2.txt");
	dim[0] = 10;
	dim[1] = 500;
	dim[2] = 1;
	makeKernel(data, weights.fc2_kernel, dim);
	makeBias(data, weights.fc2_bias, dim);
	return weights;
}

//Requires arguments to already be in CUDA memory
void forward_prop(float *inputImage, float *outputVector, weights_struct weights, cudaStream_t stream){
	float * c1_out, * p1_out, * c2_out, * p2_out, * fc1_out, *fc1_relu_out, * fc2_out, * fc2_softmax_out;
	// dimensions: (height, width, output_channels)
	cudaMalloc(&c1_out, 24 * 24 * 20 * sizeof(float));
	cudaMalloc(&p1_out, 12 * 12 * 20 * sizeof(float));
	cudaMalloc(&c2_out, 8 * 8 * 50 * sizeof(float));
	cudaMalloc(&p2_out, 4 * 4 * 50 * sizeof(float));
	cudaMalloc(&fc1_out, 1 * 1 * 500 * sizeof(float));
	cudaMalloc(&fc1_relu_out, 1 * 1 * 500 * sizeof(float));
	cudaMalloc(&fc2_out, 1 * 1 * 10 * sizeof(float));
	// cudaMalloc(&fc2_softmax_out, 1 * 1 * 10 * sizeof(float));
	// C1: 45 blocks, 256 threads per block
	convLayer<<<dim3(3,3,5), dim3(8,8,4), 0, stream>>>(inputImage, c1_out, weights.conv1_kernel, weights.conv1_bias, 5, 1, 28, 28, 20, 1);
	// P1: 16 blocks, 180 threads per block
	pool<<<dim3(2,2,4), dim3(6,6,5), 0, stream>>>(c1_out, p1_out, 24, 24, 20, 2, MAXPOOL, 2);
	// C2: 20 blocks, 160 threads per block
	convLayer<<<dim3(2,2,5), dim3(4,4,10), 0, stream>>>(p1_out, c2_out, weights.conv1_kernel, weights.conv1_bias, 5, 1, 12, 12, 50, 1);
	// P2: 5 blocks, 160 threads per block
	pool<<<dim3(1,1,5), dim3(4,4,10), 0, stream>>>(c2_out, p2_out, 8, 8, 50, 2, MAXPOOL, 2);
	// FC1: 4 blocks, 125 threads per block
	convLayer<<<dim3(1,1,4), dim3(1,1,125), 0, stream>>>(p2_out, fc1_out, weights.fc1_kernel, weights.fc1_bias, 4, 1, 4, 4, 500, 1);
	// Relu1: 4 blocks, 125 threads per block
	ReLU<<<dim3(1,1,4), dim3(1,1,125), 0, stream>>>(fc1_out, fc1_relu_out, 500);
	// FC2: 1 block, 10 threads per block
	convLayer<<<dim3(1,1,1), dim3(1,1,10), 0, stream>>>(fc1_relu_out, fc2_out, weights.fc2_kernel, weights.fc2_bias, 1, 1, 1, 1, 10, 1);
	// Softmax: 1 block, 10 threads per block
	// softmax<<<dim3(1,1,1), dim3(1,1,10), 0, stream>>>(fc2_out, outputVector, 10); //! This would have to be changed
	cudaMemcpy(outputVector, fc2_out, 10 * sizeof(float), cudaMemcpyDeviceToDevice);

	//Free memory
	cudaFree(c1_out);
	cudaFree(p1_out);
	cudaFree(c2_out);
	cudaFree(p2_out);
	cudaFree(fc1_out);
	cudaFree(fc1_relu_out);
	cudaFree(fc2_out);
}

int main() {
	// Loop over the pre-proc-img directory and perform forward prop on each image
	weights_struct weights = prep_weights();

	vector<string> images;
	for (const auto& dirEntry : std::filesystem::recursive_directory_iterator("./pre-proc-img/")){
		images.push_back(dirEntry.path().string());
	}
	cout << images[0] << endl;
	//iterate through images
	// for (int i = 0; i < images.size(); i++){
		float* inputImage = prep_inputs(images[0]);
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

		// print along the 3rd index
		for (int i = 0; i < 20; i++){
			cout << host_outputVector[i] << endl;
		}
		delete[] host_outputVector;
		cudaFree(inputImage);
		cudaFree(outputVector);
	// }
	// vector<float> data;
	// readFile(data, "./weights/conv2.txt");
	// for (int i = 0; i < data.size(); i++)
	//	cout << data[i] << endl;
}
