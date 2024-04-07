#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
using namespace std;

#define MAXPOOL 0
#define AVGPOOL 1

// Apply ReLU activation to each element of the matrix
__global__ void reLU(float ** inputMatrix, float ** outputMatrix) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	outputMatrix[i][j] = max(0.0f,inputMatrix[i][j]);
}


// Apply tanh activation to each element of the matrix
__global__ void tanh(float ** inputMatrix, float ** outputMatrix) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	outputMatrix[i][j] = tanh(inputMatrix[i][j]);
}


// Can speed up by using shared memory
__global__ void Pool(float** inputMatrix, float** outputMatrix, int pooldim, int pooltype, int stride = 1) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	float val = 0.0f;
	if (pooltype == MAXPOOL)
		val = inputMatrix[i * stride][j * stride];

	for (int k = 0; k < pooldim; k++) {
		for (int l = 0; l < pooldim; l++) {
			if (pooltype == MAXPOOL)
				val = max(val, inputMatrix[i * stride + k][j * stride + l]);
			else if (pooltype == AVGPOOL)
				val += inputMatrix[i * stride + k][j * stride + l];
		}
	}

	if (pooltype == AVGPOOL)
		val = val / pow(2.0f, pooldim);
	outputMatrix[i][j] = val;
}


// sigmoid value of each element of the vector
__global__ void sigmoid(float* inputVector, float* outputVector) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	outputVector[i] = 1 / (1 + exp(-inputVector[i]));
}

// softmax value of each element of the vector
__global__ void softmax(float* inputVector, float* outputVector) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	outputVector[i] = exp(inputVector[i]);

	__syncthreads();

	// The below code forces a serial computation onto just 1 thread and makes the others wait
	// More efficient way to do this is to use a reduction kernel, or split into two parts
	__shared__ int sum;
	if(threadIdx.x == 0) {
		sum = 0;
		for(int j = 0; j < blockDim.x; j++) {
			sum += outputVector[j];

		}
	}
	__syncthreads();

	outputVector[i] /= sum;
}

// __global__ void sum_reduction_kernel(float* inputVector, float* outputVector, int size) {
// 	extern __shared__ float sdata[];
// 	unsigned int tid = threadIdx.x;
// 	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
// 	sdata[tid] = (i < size) ? inputVector[i] : 0;
// 	__syncthreads();
// 	for(unsigned int s = 1; s < blockDim.x; s *= 2) {
// 		if(tid % (2 * s) == 0) {
// 			sdata[tid] += sdata[tid + s];
// 		}
// 		__syncthreads();
// 	}
// 	if(tid == 0) {
// 		outputVector[blockIdx.x] = sdata[0];
// 	}
// }

int test_pool(){
	int inputsize = 4;
	int outputsize = 2;
	float inputMatrix[4][4] = { {1, 2, 3, 4},
								{5, 6, 7, 8},
								{9, 10, 11, 12},
								{13, 14, 15, 16} };
	// output matrix
	float outputMatrix[2][2];
	// device input matrix
	float** d_inputMatrix;
	cudaMalloc(&d_inputMatrix, inputsize * inputsize * sizeof(float));
	cudaMemcpy(d_inputMatrix, inputMatrix, inputsize * inputsize * sizeof(float), cudaMemcpyHostToDevice);
	// device output matrix
	float** d_outputMatrix;
	cudaMalloc(&d_outputMatrix, outputsize * outputsize * sizeof(float));
	// number of threads per block
	dim3 threadsPerBlock(2, 2);
	// number of blocks
	int blocksPerGrid = 1;
	// call Pool function
	Pool <<<blocksPerGrid, threadsPerBlock>>> (d_inputMatrix, d_outputMatrix, 2, MAXPOOL);
	// copy output matrix from device to host
	cudaMemcpy(outputMatrix, d_outputMatrix, outputsize * outputsize * sizeof(float), cudaMemcpyDeviceToHost);
	// print output matrix
	for (int i = 0; i < outputsize; i++) {
		for (int j = 0; j < outputsize; j++) {
			cout << outputMatrix[i][j] << " ";
		}
		cout << endl;
	}
	// free device memory
	cudaFree(d_inputMatrix);
	cudaFree(d_outputMatrix);
	return 0;
}

void test_softmax(){
	// input vector

	int inputsize = 10;
	float inputVector[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	// output vector
	float outputVector[inputsize];

	// device input vector
	float* d_inputVector;
	cudaMalloc(&d_inputVector, inputsize * sizeof(float));
	cudaMemcpy(d_inputVector, inputVector, inputsize * sizeof(float), cudaMemcpyHostToDevice);
	// device output vector
	float* d_outputVector;
	cudaMalloc(&d_outputVector, inputsize * sizeof(float));
	// number of threads per block
	int threadsPerBlock = 10;
	// number of blocks
	int blocksPerGrid = 1;
	// call sigmoid function
	softmax <<<blocksPerGrid, threadsPerBlock >>> (d_inputVector, d_outputVector);
	// copy output vector from device to host
	cudaMemcpy(outputVector, d_outputVector, inputsize * sizeof(float), cudaMemcpyDeviceToHost);
	// print output vector
	for (int i = 0; i < inputsize; i++) {
		cout << outputVector[i] << " ";
	}
	cout << endl;
	// free device memory
	cudaFree(d_inputVector);
	cudaFree(d_outputVector);
}

int main(){
	test_pool();
}
