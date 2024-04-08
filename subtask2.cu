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

// Can speed up by using shared memory 
//^ Note: Expects inputMatrix to be the padded input matrix
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

// Can speed up using shared memory
//^ Note: Expects inputMatrix to be the padded input matrix
__global__ void convolveMatrix(float** inputMatrix, float** outputMatrix, float** kernel, int kernel_size, int stride = 1) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	float val = 0.0f;
	for (int k = 0; k < kernel_size; k++) {
		for (int l = 0; l < kernel_size; l++) {
			val += inputMatrix[i * stride + k][j * stride + l] * kernel[k][l];
		}
	}
	outputMatrix[i][j] = val;
}


__global__ void padMatrix(float** inputMatrix, float** outputMatix, int padding){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	outputMatix[i + padding][j + padding] = inputMatrix[i][j];
}

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
			cout << "Please provide the N, M, and P" << endl;
			return 1;
		}
		int N = stoi(argv[2]);
		int M = stoi(argv[3]);
		int P = stoi(argv[4]);
		if (argc < 5 + N * N + M * M){
			cout << "Please provide the matrix and kernel" << endl;
			return 1;
		}
		float inputMatrix[N][N];
		float kernel[M][M];
		for (int i = 0; i < N; i++){
			for (int j = 0; j < N; j++){
				inputMatrix[i][j] = stof(argv[5 + i * N + j]);
			}
		}
		for (int i = 0; i < M; i++){
			for (int j = 0; j < M; j++){
				kernel[i][j] = stof(argv[5 + N * N + i * M + j]);
			}
		}
		cout << "Subtask1: Convolution" << endl;
		float outputMatrix[N][N];
		float** d_inputMatrix;
		float** d_kernel;
		cudaMalloc(&d_inputMatrix, N * N * sizeof(float));
		cudaMalloc(&d_kernel, M * M * sizeof(float));
		cudaMemcpy(d_inputMatrix, inputMatrix, N * N * sizeof(float), cudaMemcpyHostToDevice);
		float** d_paddedMatrix;
		cudaMalloc(&d_paddedMatrix, (N + 2 * P) * (N + 2 * P) * sizeof(float));
		padMatrix <<<1, dim3(N, N)>>> (d_inputMatrix, d_paddedMatrix, P); //! Check dimensions
		float** d_outputMatrix;
		cudaMalloc(&d_outputMatrix, N * N * sizeof(float));
		dim3 threadsPerBlock(2, 2);
		int blocksPerGrid = 1;
		convolveMatrix <<<blocksPerGrid, threadsPerBlock>>> (d_paddedMatrix, d_outputMatrix, d_kernel, M, 1); //! Check dimensions
		cudaMemcpy(outputMatrix, d_outputMatrix, N * N * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "Output Matrix:" << endl;
		for (int i = 0; i < N; i++){
			for (int j = 0; j < N; j++){
				cout << outputMatrix[i][j] << " ";
			}
			cout << endl;
		}	
	}
	else if (subtask == 2){
		if (argc < 5){
			cout << "Please provide activation function, N, and M" << endl;
			return 1;
		}
		int activation = stoi(argv[2]);
		int N = stoi(argv[3]);
		// int M = stoi(argv[4]); //! Not used
		if (argc < 5 + N * N){
			cout << "Please provide the matrix" << endl;
			return 1;
		}
		float inputMatrix[N][N];
		for (int i = 0; i < N; i++){
			for (int j = 0; j < N; j++){
				inputMatrix[i][j] = stof(argv[5 + i * N + j]);
			}
		}
		cout << "Subtask2: Non-linear activations" << endl;
		float outputMatrix[N][N];
		float** d_inputMatrix;
		cudaMalloc(&d_inputMatrix, N * N * sizeof(float));
		cudaMemcpy(d_inputMatrix, inputMatrix, N * N * sizeof(float), cudaMemcpyHostToDevice);
		float** d_outputMatrix;
		cudaMalloc(&d_outputMatrix, N * N * sizeof(float));
		dim3 threadsPerBlock(2, 2); //! Check dimensions
		int blocksPerGrid = 1; //! Check dimensions
		if (activation == 0){
			reLU <<<blocksPerGrid, threadsPerBlock>>> (d_inputMatrix, d_outputMatrix);
		}
		else if (activation == 1){
			tanh <<<blocksPerGrid, threadsPerBlock>>> (d_inputMatrix, d_outputMatrix);
		}
		cudaMemcpy(outputMatrix, d_outputMatrix, N * N * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "Output Matrix:" << endl;
		for (int i = 0; i < N; i++){
			for (int j = 0; j < N; j++){
				cout << outputMatrix[i][j] << " ";
			}
			cout << endl;
		}
	}
	else if (subtask == 3){
		if (argc < 4){
			cout << "Please provide the pooling function and N" << endl;
			return 1;
		}
		int pooltype = stoi(argv[2]);
		int N = stoi(argv[3]);
		if (argc < 4 + N * N){
			cout << "Please provide the matrix" << endl;
			return 1;
		}
		float inputMatrix[N][N];
		for (int i = 0; i < N; i++){
			for (int j = 0; j < N; j++){
				inputMatrix[i][j] = stof(argv[4 + i * N + j]);
			}
		}
		cout << "Subtask3: Subsampling" << endl;
		int pool_dim = 2;
		int outputsize = N / pool_dim;
		float outputMatrix[outputsize][outputsize];
		float** d_inputMatrix;
		cudaMalloc(&d_inputMatrix, N * N * sizeof(float));
		cudaMemcpy(d_inputMatrix, inputMatrix, N * N * sizeof(float), cudaMemcpyHostToDevice);
		float** d_outputMatrix;
		cudaMalloc(&d_outputMatrix, outputsize * outputsize * sizeof(float));
		dim3 threadsPerBlock(2, 2); //! Check dimensions
		int blocksPerGrid = 1; //! Check dimensions
		Pool <<<blocksPerGrid, threadsPerBlock>>> (d_inputMatrix, d_outputMatrix, pool_dim, pooltype, pool_dim);
		cudaMemcpy(outputMatrix, d_outputMatrix, outputsize * outputsize * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "Output Matrix:" << endl;
		for (int i = 0; i < outputsize; i++){
			for (int j = 0; j < outputsize; j++){
				cout << outputMatrix[i][j] << " ";
			}
			cout << endl;
		}
	}
	else if (subtask == 4){
		if (argc < 3){
			cout << "Please provide the function" << endl;
			return 1;
		}
		int function = stoi(argv[2]);
		// Note that the input vector size, N is not given
		// Calcuate from argc
		int N = argc - 3;
		float inputVector[N];
		for (int i = 0; i < N; i++){
			inputVector[i] = stof(argv[3 + i]);
		}
		cout << "Subtask4: Converting a vector" << endl;
		float outputVector[N];
		float* d_inputVector;
		cudaMalloc(&d_inputVector, N * sizeof(float));
		cudaMemcpy(d_inputVector, inputVector, N * sizeof(float), cudaMemcpyHostToDevice);
		float* d_outputVector;
		cudaMalloc(&d_outputVector, N * sizeof(float));
		int threadsPerBlock = N; //! Check dimensions
		int blocksPerGrid = 1; //! Check dimensions
		if (function == 0){
			sigmoid <<<blocksPerGrid, threadsPerBlock>>> (d_inputVector, d_outputVector);
		}
		else if (function == 1){
			softmax <<<blocksPerGrid, threadsPerBlock>>> (d_inputVector, d_outputVector);
		}
		cudaMemcpy(outputVector, d_outputVector, N * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "Output Vector:" << endl;
		for (int i = 0; i < N; i++){
			cout << outputVector[i] << " ";
		}
		cout << endl;
	}
	else{
		cout << "Invalid subtask number" << endl;
		cout << "Running test pool instead" << endl;
		test_pool();
	}
}
