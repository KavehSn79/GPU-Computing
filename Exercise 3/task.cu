#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#define CHECK_CUDA(call)                                        \
    if ((call) != cudaSuccess)                                  \
    {                                                           \
        std::cerr << "CUDA error at " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                     \
    }

const int NUM_MATRICES = 10; // Number of matrix multiplications
const int MATRIX_SIZE = 4096;
const int TILE_SIZE = 32;


// Simple kernel for matrix multiplication
__global__ void matrixMultiplyKernel(const float *A, const float *B, float *C, int n)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < n && col < n)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
        {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Tiled kernel for matrix multiplication
__global__ void matrixMultiplyKernelTiled(const float *A, const float *B, float *C, int n)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;


    // TODO: allocate shared memory for two tiles (one for A and one for B)
    __shared__ float sh_A[TILE_SIZE][TILE_SIZE];
    __shared__ float sh_B[TILE_SIZE][TILE_SIZE];
    // TODO: iterate over tiles
    float sum = 0;
    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    for (int T = 0; T < numTiles; T++)
    {
        // TODO: copy tiles from global memory into shared memory
        sh_A[ty][tx] = A[(row * n) + (T * TILE_SIZE + tx)];
        sh_B[ty][tx] = B[(T * TILE_SIZE + ty) * n + col];
        __syncthreads();
        // TODO: compute the matrix multiplication of the two tiles
        for (int k = 0; k < TILE_SIZE; k++)
            sum += sh_A[ty][k] * sh_B[k][tx];
        __syncthreads();
    }    
    // TODO: write back the results into the matrix C
    C[row * n + col] = sum;
}

void matrixMultiplyNoStreams()
{
    // Host and device pointers
    float *h_A[NUM_MATRICES], *h_B[NUM_MATRICES], *h_C[NUM_MATRICES];
    float *d_A[NUM_MATRICES], *d_B[NUM_MATRICES], *d_C[NUM_MATRICES];

    for (int i = 0; i < NUM_MATRICES; i++)
    {
        h_A[i] = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
        h_B[i] = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
        h_C[i] = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

        // Initialize example matrices with random numbers
        for (int j = 0; j < MATRIX_SIZE * MATRIX_SIZE; j++)
        {
            // pick testing values, that allow us to compute the expected result on the CPU cheaply
            h_A[i][j] = 1.0f;
            h_B[i][j] = 0.01f;
            h_C[i][j] = 0.0f;
        }

        CHECK_CUDA(cudaMalloc(&d_A[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));

        // Copy matrices A and B to the device
        CHECK_CUDA(cudaMemcpy(d_A[i], h_A[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B[i], h_B[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice));

        // Launch matrix multiplication kernel
        dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
        dim3 blocksPerGrid(MATRIX_SIZE/TILE_SIZE, MATRIX_SIZE/TILE_SIZE);

        std::cout << "Launch kernel with " << blocksPerGrid.x * blocksPerGrid.y << " blocks each with " << threadsPerBlock.x * threadsPerBlock.y << " threads\n";
        //matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A[i], d_B[i], d_C[i], MATRIX_SIZE);
        matrixMultiplyKernelTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A[i], d_B[i], d_C[i], MATRIX_SIZE);
        CHECK_CUDA(cudaGetLastError());

        // Copy results back to the host
        CHECK_CUDA(cudaMemcpy(h_C[i], d_C[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

        // Verify results
        double eps = 1.e-6;  // machine zero
        for (int j = 0; j < MATRIX_SIZE * MATRIX_SIZE; j++) {
            double abs_err = fabs(h_C[i][j] - (MATRIX_SIZE * 0.01f));
            double dot_length = MATRIX_SIZE;
            double abs_val = fabs(h_C[i][j]);
            double rel_err = abs_err / abs_val / dot_length;

            if (rel_err > eps) {
                printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    j, h_C[i][j], MATRIX_SIZE * 0.01f, eps);
            }
        }

        // Cleanup
        free(h_A[i]);
        free(h_B[i]);
        free(h_C[i]);
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
    }
}

void matrixMultiplyWithStreams()
{
    // Host and device pointers
    float *h_A[NUM_MATRICES], *h_B[NUM_MATRICES], *h_C[NUM_MATRICES];
    float *d_A[NUM_MATRICES], *d_B[NUM_MATRICES], *d_C[NUM_MATRICES];

    cudaStream_t streams[NUM_MATRICES];

    size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(MATRIX_SIZE/TILE_SIZE, MATRIX_SIZE/TILE_SIZE);

    // TODO: Allocate memory, initialize data, create streams and copy data asynchronously
    
    for (int i = 0; i < NUM_MATRICES; ++i){
        cudaStreamCreate(&streams[i]);
    }
    for (int i = 0; i < NUM_MATRICES; ++i){
        CHECK_CUDA(cudaMalloc(&d_A[i], size));
        CHECK_CUDA(cudaMalloc(&d_B[i], size));
        CHECK_CUDA(cudaMalloc(&d_C[i], size));

        CHECK_CUDA(cudaMallocHost(&h_A[i], size));
        CHECK_CUDA(cudaMallocHost(&h_B[i], size));
        CHECK_CUDA(cudaMallocHost(&h_C[i], size));

        // Initialize example matrices with random numbers
        for (int j = 0; j < MATRIX_SIZE * MATRIX_SIZE; j++)
        {
            // pick testing values, that allow us to compute the expected result on the CPU cheaply
            h_A[i][j] = 1.0f;
            h_B[i][j] = 0.01f;
            h_C[i][j] = 0.0f;
        }

        CHECK_CUDA(cudaMemcpyAsync(d_A[i], h_A[i], size, cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaMemcpyAsync(d_B[i], h_B[i], size, cudaMemcpyHostToDevice, streams[i]));
        
        std::cout << "Launch kernel with " << blocksPerGrid.x * blocksPerGrid.y << " blocks each with " << threadsPerBlock.x * threadsPerBlock.y << " threads\n";
        //matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(d_A[i], d_B[i], d_C[i], MATRIX_SIZE);
        matrixMultiplyKernelTiled<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(d_A[i], d_B[i], d_C[i], MATRIX_SIZE);
        CHECK_CUDA(cudaGetLastError());
        // TODO: Launch matrix multiplication kernel for each stream
        CHECK_CUDA(cudaMemcpyAsync(h_C[i], d_C[i], size, cudaMemcpyDeviceToHost, streams[i]));
    }
    for (int i = 0; i < NUM_MATRICES; i++)
        cudaStreamSynchronize(streams[i]);

    

    // TODO: Copy results back to the host asynchronously

    // TODO: Synchronize all streams
    double eps = 1.e-3;  // tolerance for float
    for (int i = 0; i < NUM_MATRICES; ++i) {
        for (int j = 0; j < MATRIX_SIZE * MATRIX_SIZE; ++j) {
            double ref = MATRIX_SIZE * 0.01f;       // expected value
            double abs_err = fabs(h_C[i][j] - ref); // absolute error
            double rel_err = abs_err / ref;         // relative error

            if (rel_err > eps) {
                printf("Error! Matrix[%d][%d] = %.8f, reference = %.8f, rel_err = %.3e\n",
                    i, j, h_C[i][j], ref, rel_err);
            }
        }
    }

    //Verify results (slow! use only for debugging)
    // for (int i = 0; i < NUM_MATRICES; i++)
    // {
    //     std::cout << "Matrix C[" << i << "]:" << std::endl;
    //     for (int row = 0; row < MATRIX_SIZE; row++)
    //     {
    //         for (int col = 0; col < MATRIX_SIZE; col++)
    //         {
    //             std::cout << h_C[i][row * MATRIX_SIZE + col] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }

    // TODO: Cleanup
    for (int i = 0; i < NUM_MATRICES; ++i){
        cudaFreeHost(h_A[i]);
        cudaFreeHost(h_B[i]);
        cudaFreeHost(h_C[i]);
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
        cudaStreamDestroy(streams[i]);
    }
}

int main()
{
    // Measure time for matrixMultiplyNoStreams
    auto start1 = std::chrono::high_resolution_clock::now();
    matrixMultiplyNoStreams();
    auto end1 = std::chrono::high_resolution_clock::now();
    double elapsed_no_streams = std::chrono::duration<double, std::milli>(end1 - start1).count();
    std::cout << "Time for Matrix Multiply with NoStreams: " << elapsed_no_streams << " ms\n";

    // Measure time for matrixMultiplyWithStreams
    auto start2 = std::chrono::high_resolution_clock::now();
    matrixMultiplyWithStreams();
    auto end2 = std::chrono::high_resolution_clock::now();
    double elapsed_with_streams = std::chrono::duration<double, std::milli>(end2 - start2).count();
    std::cout << "Time for Tiling Matrix Multiply With Streams: " << elapsed_with_streams << " ms\n";

    return EXIT_SUCCESS;
}