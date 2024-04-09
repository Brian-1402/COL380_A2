#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
using namespace std;

#define MAXPOOL 0
#define AVGPOOL 1

// Apply ReLU activation to each element of the matrix
__global__ void ReLU(float* inputMatrix, float* outputMatrix, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < height && j < width) {
        int index = i * width + j;
        outputMatrix[index] = max(0.0f, inputMatrix[index]);
    }
}

// Apply tanh activation to each element of the matrix
__global__ void tanh(float* inputMatrix, float* outputMatrix, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < height && j < width) {
        int index = i * width + j;
        outputMatrix[index] = tanh(inputMatrix[index]);
    }
}

// sigmoid value of each element of the vector
__global__ void sigmoid(float* inputVector, float* outputVector, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        outputVector[i] = 1 / (1 + exp(-inputVector[i]));
    }
}

__global__ void softmax_exp(float* inputVector, float* outputVector, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        outputVector[i] = exp(inputVector[i]);
    }
}

__global__ void softmax_sum_div(float* outputVector, float sum, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        outputVector[i] /= sum;
    }
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
__global__ void Pool(float* inputMatrix, float* outputMatrix, int width, int pooltype, int pooldim = 2, int stride = 1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float val = 0.0f;
    if (i < width/pooldim && j < width/pooldim) {
        if (pooltype == MAXPOOL)
            val = inputMatrix[(i * stride * width) + (j * stride)];

        for (int k = 0; k < pooldim; k++)
            for (int l = 0; l < pooldim; l++)
                if (pooltype == MAXPOOL)
                    val = max(val, inputMatrix[((i * stride + k) * width) + (j * stride + l)]);
                else if (pooltype == AVGPOOL)
                    val += inputMatrix[((i * stride + k) * width) + (j * stride + l)];

        if (pooltype == AVGPOOL)
            val = val / (pooldim * pooldim);
        outputMatrix[i * width + j] = val;
    }
}

// Can speed up using shared memory
//^ Note: Expects inputMatrix to be the padded input matrix
__global__ void convolveMatrix(float* inputMatrix, float* outputMatrix, float* kernel, int width, int kernel_size, int stride = 1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float val = 0.0f;
    if (i < width/kernel_size && j < width/kernel_size) {
        for (int k = 0; k < kernel_size; k++)
            for (int l = 0; l < kernel_size; l++)
                val += inputMatrix[(i * stride + k) * width + (j * stride + l)] * kernel[k * kernel_size + l];
        outputMatrix[i * width + j] = val;
    }
}

__global__ void padMatrix(float* inputMatrix, float* outputMatrix, int width, int padding) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < width && j < width) {
        int outputIndex = (i + padding) * (width + 2 * padding) + (j + padding);
        int inputIndex = i * width + j;
        outputMatrix[outputIndex] = inputMatrix[inputIndex];
    }
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
    float* d_inputMatrix;
    // device output matrix
    float* d_outputMatrix;
    // number of threads per block
    dim3 threadsPerBlock(2, 2);
    // number of blocks
    int blocksPerGrid = 1;
    cudaMalloc(&d_inputMatrix, inputsize * inputsize * sizeof(float));
    cudaMemcpy(d_inputMatrix, inputMatrix, inputsize * inputsize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_outputMatrix, outputsize * outputsize * sizeof(float));
    // call Pool function
    Pool <<<blocksPerGrid, threadsPerBlock>>> (d_inputMatrix, d_outputMatrix, inputsize, MAXPOOL);
    // copy output matrix from device to host
    cudaMemcpy(outputMatrix, d_outputMatrix, outputsize * outputsize * sizeof(float), cudaMemcpyDeviceToHost);
    // print output matrix
    cudaDeviceSynchronize();
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
    // device output vector
    float* d_outputVector;
    // device input vector
    float* d_inputVector;
    // number of threads per block
    int threadsPerBlock = 10;
    // number of blocks
    int blocksPerGrid = 1;
    cudaMalloc(&d_inputVector, inputsize * sizeof(float));
    cudaMemcpy(d_inputVector, inputVector, inputsize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_outputVector, inputsize * sizeof(float));
    // call softmax_exp function
    softmax_exp <<<blocksPerGrid, threadsPerBlock >>> (d_inputVector, d_outputVector, inputsize);
    cudaDeviceSynchronize();
    // calculate sum on host
    float sum = 0.0f;
    cudaMemcpy(outputVector, d_outputVector, inputsize * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < inputsize; i++) {
        sum += outputVector[i];
    }
    // call softmax_sum_div function
    softmax_sum_div <<<blocksPerGrid, threadsPerBlock >>> (d_outputVector, sum, inputsize);
    cudaDeviceSynchronize();
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

// struct dim6{
// 	dim3 thread_idx; // 3D grid where i can access each subdimension as thread_idx.x etc
// 	dim3 block_idx; // 2D block where i can access each subdimension as block_idx.x etc
// };

// // Given inputs 
// struct dim6 get_dim6(int subtask,int N, int M, int P, int pool_dim = 2, int stride = 1){
// 	struct dim6 d;
// 	// d.thread_idx.x = N;
// 	// d.thread_idx.y = N;
// 	// d.thread_idx.z = 1;
// 	// d.block_idx.x = M;
// 	// d.block_idx.y = M;
// 	switch(subtask){
// 		case(1):
// 			// Convolution subtask
// 			// N would be input matrix size, M would be kernel size, P would be padding, remaning useless
// 			// Note that both input and kernel could possibly be really large so we need to make sure that after creation of threads, max number of threads per block is just 512
// 			int padded_size = N + 2 * P;
// 			int threads_per_block = 512;
// 			int blocks_per_grid = (padded_size * padded_size) / threads_per_block;
// 			d.thread_idx.x = 2;
// 			d.thread_idx.y = 2;
// 			d.thread_idx.z = 1;
// 			d.block_idx.x = blocks_per_grid;
// 			d.block_idx.y = 1;
// 			d.block_idx.z = 1;
// 			break;

		
// 	}
// 	return d;
// };

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
		float inputMatrix[N * N];
		float kernel[M * M];
		for (int i = 0; i < N; i++){
			for (int j = 0; j < N; j++){
				inputMatrix[i * N + j] = stof(argv[5 + i * N + j]);
			}
		}
		for (int i = 0; i < M; i++){
			for (int j = 0; j < M; j++){
				kernel[i * M + j] = stof(argv[5 + N * N + i * M + j]);
			}
		}
		cout << "Subtask1: Convolution" << endl;
		float outputMatrix[N * N];
		float* d_inputMatrix;
		float* d_kernel;
		float* d_paddedMatrix;
		float* d_outputMatrix;
		dim3 threadsPerBlock(16,16);
		dim3 blocksPerGrid(4,2);
		cudaMalloc(&d_inputMatrix, N * N * sizeof(float));
		cudaMalloc(&d_kernel, M * M * sizeof(float));
		cudaMalloc(&d_paddedMatrix, (N + 2 * P) * (N + 2 * P) * sizeof(float));
		cudaMalloc(&d_outputMatrix, N * N * sizeof(float));
		cudaMemcpy(d_inputMatrix, inputMatrix, N * N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_kernel, kernel, M * M * sizeof(float), cudaMemcpyHostToDevice);
		padMatrix <<<blocksPerGrid, threadsPerBlock>>> (d_inputMatrix, d_paddedMatrix, N, P);
		convolveMatrix <<<blocksPerGrid, threadsPerBlock>>> (d_paddedMatrix, d_outputMatrix, d_kernel, N + 2 * P, M, 1);
		cudaMemcpy(outputMatrix, d_outputMatrix, N * N * sizeof(float), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		cout << "Output Matrix:" << endl;
		for (int i = 0; i < N; i++){
			for (int j = 0; j < N; j++){
				cout << outputMatrix[i * N + j] << " ";
			}
			cout << endl;
		}
		cudaFree(d_inputMatrix);
		cudaFree(d_kernel);
		cudaFree(d_paddedMatrix);
		cudaFree(d_outputMatrix);
	}
	else if (subtask == 2){
		if (argc < 5){
			cout << "Please provide the activation function, N, M" << endl;
			return 1;
		}
		int activation_fn = stoi(argv[2]);
		int N = stoi(argv[3]);
		int M = stoi(argv[4]);
		if (argc < 5 + N * M){
			cout << "Please provide the matrix and kernel" << endl;
			return 1;
		}
		float inputMatrix[N * M];
		for (int i = 0; i < N; i++){
			for (int j = 0; j < M; j++){
				inputMatrix[i * N + j] = stof(argv[5 + i * N + j]);
			}
		}
		cout << "Subtask2: Non-linear activation" << endl;
		string activationName = (activation_fn == 0) ? "ReLU" : "tanh";
        cout << "Activation: " << activationName << endl;
		float outputMatrix[N * M];
		float* d_inputMatrix;
		float* d_outputMatrix;
		dim3 threadsPerBlock(16,16);
		dim3 blocksPerGrid(4,2);
		cudaMalloc(&d_inputMatrix, N * M * sizeof(float));
		cudaMemcpy(d_inputMatrix, inputMatrix, N * M * sizeof(float), cudaMemcpyHostToDevice);
		cudaMalloc(&d_outputMatrix, N * M * sizeof(float));
		if (activation_fn == 0){
			ReLU <<<blocksPerGrid, threadsPerBlock>>> (d_inputMatrix, d_outputMatrix, M, N);
		}
		else if (activation_fn == 1){
			tanh <<<blocksPerGrid, threadsPerBlock>>> (d_inputMatrix, d_outputMatrix, M, N);
		}
		cudaMemcpy(outputMatrix, d_outputMatrix, N * M * sizeof(float), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		cout << "Output Matrix:" << endl;
		for (int i = 0; i < N; i++){
			for (int j = 0; j < M; j++){
				cout << outputMatrix[i * N + j] << " ";
			}
			cout << endl;
		}
		cudaFree(d_inputMatrix);
		cudaFree(d_outputMatrix);
	}
	else if (subtask == 3){
		if (argc < 5){
			cout << "Please provide the pooling function and N" << endl;
			return 1;
		}
		int pooltype = stoi(argv[2]);
		int pool_dim = stoi(argv[3]);
		int N = stoi(argv[4]);
		if (argc < 5 + N * N){
			cout << "Please provide the matrix" << endl;
			return 1;
		}
		float inputMatrix[N * N];
		for (int i = 0; i < N; i++){
			for (int j = 0; j < N; j++){
				inputMatrix[i * N + j] = stof(argv[5 + i * N + j]);
			}
		}
		cout << "Subtask3: Subsampling" << endl;
		int outputsize = N / pool_dim;
		float outputMatrix[outputsize * outputsize];
		float* d_inputMatrix;
		float* d_outputMatrix;
		dim3 threadsPerBlock(16,16);
		dim3 blocksPerGrid(4,2);
		cudaMalloc(&d_inputMatrix, N * N * sizeof(float));
		cudaMemcpy(d_inputMatrix, inputMatrix, N * N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMalloc(&d_outputMatrix, outputsize * outputsize * sizeof(float));
		Pool <<<blocksPerGrid, threadsPerBlock>>> (d_inputMatrix, d_outputMatrix, pool_dim, pooltype, 1);
		cudaMemcpy(outputMatrix, d_outputMatrix, outputsize * outputsize * sizeof(float), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		cout << "Output Matrix:" << endl;
		for (int i = 0; i < outputsize; i++){
			for (int j = 0; j < outputsize; j++){
				cout << outputMatrix[i * outputsize + j] << " ";
			}
			cout << endl;
		}
		cudaFree(d_inputMatrix);
		cudaFree(d_outputMatrix);
	}
	else if (subtask == 4){
		if (argc < 3){
			cout << "Please provide the function" << endl;
			return 1;
		}
		int function = stoi(argv[2]);
		int N = argc - 3;
		float inputVector[N];
		for (int i = 0; i < N; i++){
			inputVector[i] = stof(argv[3 + i]);
		}
		cout << "Subtask4: Converting a vector" << endl;
		float outputVector[N];
		float* d_inputVector;
		float* d_outputVector;
		int threadsPerBlock = 256;
		int blocksPerGrid = N/256;
		cudaMalloc(&d_inputVector, N * sizeof(float));
		cudaMemcpy(d_inputVector, inputVector, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMalloc(&d_outputVector, N * sizeof(float));
		if (function == 0){
			sigmoid <<<blocksPerGrid, threadsPerBlock>>> (d_inputVector, d_outputVector, N);
		}
		else if (function == 1){
			softmax_exp <<<blocksPerGrid, threadsPerBlock>>> (d_inputVector, d_outputVector, N);
			cudaMemcpy(outputVector, d_outputVector, N * sizeof(float), cudaMemcpyDeviceToHost);
			float sum = 0.0f;
			for (int i = 0; i < N; i++){
				sum += outputVector[i];
			}
			cudaMemcpy(d_outputVector, outputVector, N * sizeof(float), cudaMemcpyHostToDevice);
			softmax_sum_div <<<blocksPerGrid, threadsPerBlock>>> (d_outputVector, sum, N);
		}
		cudaMemcpy(outputVector, d_outputVector, N * sizeof(float), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		cout << "Output Vector:" << endl;
		for (int i = 0; i < N; i++){
			cout << outputVector[i] << " ";
		}
		cout << endl;
		cudaFree(d_inputVector);
		cudaFree(d_outputVector);
	}
	else{
		cout << "Invalid subtask number" << endl;
		cout << "Running test pool instead" << endl;
		test_pool();
	}
}
