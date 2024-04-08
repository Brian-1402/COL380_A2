#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
using namespace std;

#define MAXPOOL 0
#define AVGPOOL 1


// Kernel dimensions explained:
// 1) output channel (sets of filters. Corresponds to output channel)
// 2) input channel (sets of 2d kernel matrices, together called a filter. Each kernel corresponds to each input channel in the input)
// 3) kernel row
// 4) kernel column
__global__ void convLayer(float*** inputMatrix, float*** outputMatrix, float**** kernel, int kernel_size, int input_filters, int stride = 1) {
	int i = blockIdx.x * blockDim.x + threadIdx.x; // height of output image
	int j = blockIdx.y * blockDim.y + threadIdx.y; // width of output image
	int k = blockIdx.z * blockDim.z + threadIdx.z; // channel of output image
	float val = 0.0f;

	for (int a = 0; a < input_filters; a++) // Iterate through input channels
		for (int b = 0; b < kernel_size; b++) // Iterate through input rows
			for (int c = 0; c < kernel_size; c++) // Iterate through input columns
				val += inputMatrix[a][i * stride + b][j * stride + c] * kernel[k][a][b][c];

	outputMatrix[k][i][j] = val;
}


__global__ void Pool(float*** inputMatrix, float*** outputMatrix, int pooldim, int pooltype, int stride = 1) {
	int i = blockIdx.x * blockDim.x + threadIdx.x; // height of output image
	int j = blockIdx.y * blockDim.y + threadIdx.y; // width of output image
	int k = blockIdx.z * blockDim.z + threadIdx.z; // channel of output image
	float val = 0.0f;
	if (pooltype == MAXPOOL)
		val = inputMatrix[k][i * stride][j * stride];

	for (int b = 0; b < pooldim; b++)
		for (int c = 0; c < pooldim; c++)
			if (pooltype == MAXPOOL)
				val = max(val, inputMatrix[k][i * stride + b][j * stride + c]);
			else if (pooltype == AVGPOOL)
				val += inputMatrix[k][i * stride + b][j * stride + c];

	if (pooltype == AVGPOOL)
		val = val / pow(2.0f, pooldim);
	outputMatrix[k][i][j] = val;
}


__global__ void padMatrix(float*** inputMatrix, float*** outputMatix, int padding){
	int i = blockIdx.x * blockDim.x + threadIdx.x; // height of input image
	int j = blockIdx.y * blockDim.y + threadIdx.y; // width of input image
	int k = blockIdx.z * blockDim.z + threadIdx.z; // channel of input image
	outputMatix[k][i + padding][j + padding] = inputMatrix[k][i][j];
}


// Apply ReLU activation to each element of the matrix
__global__ void reLU(float ** inputMatrix, float ** outputMatrix) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	outputMatrix[i][j] = max(0.0f,inputMatrix[i][j]);
}


