#include <format>
#include <iostream>
#include <random>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <chrono>

void check(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << std::format("{} (error: {})\n", msg, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

void init(int32_t size, int32_t *vec_a, int32_t *vec_b, int32_t *mat)
{
    // std::random_device dev;
    std::mt19937 prng(2024);
    std::uniform_int_distribution<int32_t> distrib(-16, 16);

    for (auto i = 0; i < size; i++)
    {
        vec_a[i] = distrib(prng);
        vec_b[i] = distrib(prng);
    }

    for (auto i = 0; i < size * size; i++)
        mat[i] = distrib(prng);
}

void cpu_compute(int32_t n, const int32_t *a, const int32_t *b, const int32_t *mat, int32_t *out)
{
    int32_t *tmp = (int32_t *)std::malloc(sizeof(int32_t) * n);
    for (int i = 0; i < n; ++i)
        tmp[i] = a[i] + b[i];

    for (int i = 0; i < n; ++i)
    {
        long long sum = 0;
        for (int j = 0; j < n; ++j)
            sum += (long long)tmp[j] * (long long)mat[i * n + j];
        out[i] = (int32_t)sum;
    }

    std::free(tmp);
}

// Kernels
__global__ void vecAddKernel(int32_t n, const int32_t *a, const int32_t *b, int32_t *tmp)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride)
        tmp[i] = a[i] + b[i];
}

__global__ void matMulRowsKernel(int32_t n, const int32_t *tmp, const int32_t *mat, int32_t *out)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n) return;

    long long sum = 0;
    for (int j = 0; j < n; ++j)
        sum += (long long)tmp[j] * (long long)mat[row * n + j];
    out[row] = (int32_t)sum;
}

int main()
{
    cudaError_t err = cudaSuccess;

    int numElements = 32768;
    std::cout << std::format("[Matrix-vector-like compute with {} elements]\n", numElements);

    size_t bytesVec = (size_t)numElements * sizeof(int32_t);
    size_t bytesMat = (size_t)numElements * (size_t)numElements * sizeof(int32_t);

    // Host allocations
    int32_t *vec_a = (int32_t *)std::malloc(bytesVec);
    int32_t *vec_b = (int32_t *)std::malloc(bytesVec);
    int32_t *mat   = (int32_t *)std::malloc(bytesMat);
    int32_t *out   = (int32_t *)std::malloc(bytesVec);
    int32_t *out_d = (int32_t *)std::malloc(bytesVec);

    // Initialize data
    init(numElements, vec_a, vec_b, mat);

    // Measure CPU runtime
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_compute(numElements, vec_a, vec_b, mat, out);
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();
    std::cout << std::format("CPU computation time: {:.3f} ms\n", cpu_time_ms);

    // Allocate device memory

    int32_t *d_a = NULL;
    err = cudaMalloc((void **)&d_a, bytesVec);
    check(err, "Failed to allocate device vector A");

    int32_t *d_b = NULL;
    err = cudaMalloc((void **)&d_b, bytesVec);
    check(err, "Failed to allocate device vector B");

    int32_t *d_mat = NULL;
    err = cudaMalloc((void **)&d_mat, bytesMat);
    check(err, "Failed to allocate device mat");

    int32_t *d_tmp = NULL;
    err = cudaMalloc((void **)&d_tmp, bytesVec);
    check(err, "Failed to allocate device tmp");

    int32_t *d_out = NULL;
    err = cudaMalloc((void **)&d_out, bytesVec);
    check(err, "Failed to allocate device tmp");

    // Copy from host to device
    std::cout << "Copy input data from the host memory to the CUDA device\n";

    err = cudaMemcpy(d_a, vec_a, bytesVec, cudaMemcpyHostToDevice);
    check(err, "cudaMemcpy d_a");

    err = cudaMemcpy(d_b, vec_b, bytesVec, cudaMemcpyHostToDevice);
    check(err, "cudaMemcpy d_b");

    err = cudaMemcpy(d_mat, mat, bytesMat, cudaMemcpyHostToDevice);
    check(err, "cudaMemcpy d_mat");

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << std::format("CUDA kernel launch with {} blocks of {} threads\n",
                           blocksPerGrid, threadsPerBlock);

    // Measure GPU runtime
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Kernel 1
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(numElements, d_a, d_b, d_tmp);
    check(cudaGetLastError(), "Failed to launch vectorAdd kernel");

    // Kernel 2
    matMulRowsKernel<<<blocksPerGrid, threadsPerBlock>>>(numElements, d_tmp, d_mat, d_out);
    check(cudaGetLastError(), "Failed to launch matMulRowsKernel kernel");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms = 0.0f;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    std::cout << std::format("GPU computation time: {:.3f} ms\n", gpu_time_ms);

    // Copy back results
    cudaMemcpy(out_d, d_out, bytesVec, cudaMemcpyDeviceToHost);
    check(err , "cudaMemcpy device to host");

    // Verify correctness
    for (int i = 0; i < numElements; ++i)
    {
        if (out_d[i] != out[i])
        {
            std::cerr << std::format("Mismatch at {}: CPU={} GPU={}\n", i, out[i], out_d[i]);
            return EXIT_FAILURE;
        }
    }
    std::cout << "Test PASSED!!\n";

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_mat);
    cudaFree(d_tmp); cudaFree(d_out);
    std::free(vec_a); std::free(vec_b); std::free(mat);
    std::free(out); std::free(out_d);

    return 0;
}
