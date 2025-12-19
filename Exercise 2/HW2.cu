#include <format>
#include <iostream>
#include <random>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <curand.h>
#include <stdlib.h>

#include "helper.cu"

void check(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << std::format("{} (error: {})\n", msg, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

void sequential_scan(size_t size, float *in_h, float *out_h) {
    out_h[0] = in_h[0];
    out_h[1] = in_h[1];
    for (auto i = 2; i < size; i += 2) {
        float real_prev = out_h[i - 2];
        float real_cur = in_h[i];
        float im_prev = out_h[i - 1];
        float im_cur = in_h[i + 1];

        out_h[i] = real_prev * real_cur - im_prev * im_cur;
        out_h[i + 1] = real_prev * im_cur + real_cur * im_prev;
    }
}


__device__ void complexMul (float a_real, float a_img, float b_real, float b_img, float &res_real, float &res_img){
    res_real = a_real * b_real - a_img * b_img;
    res_img  = a_real * b_img + a_img * b_real;
}

// per-block inclusive scan
__global__ void scan (float *nums, int N, float *blockSums){
    extern __shared__ float sh_mem[];

    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + tid;

    if (gid < N){
        sh_mem[2 * tid]     = nums[2 * gid];
        sh_mem[2 * tid + 1] = nums[2 * gid + 1];
    } else {
        sh_mem[2 * tid]     = 1.0f;
        sh_mem[2 * tid + 1] = 0.0f;
    }

    __syncthreads();

    for (int d = 0; (1 << d) < blockDim.x; ++d){
        int stride = 1 << d;
        float real = 1.0f;
        float img = 0.0f;

        if (tid >= stride) {
            complexMul(sh_mem[2*(tid - stride)], 
                       sh_mem[2*(tid - stride)+1],
                       sh_mem[2*tid], 
                       sh_mem[2*tid+1], real, img);
        }

        __syncthreads();

        if (tid >= stride) {
            sh_mem[2 * tid]     = real;
            sh_mem[2 * tid + 1] = img;
        }

        __syncthreads();
    }

    if (gid < N) {
        nums[2 * gid]     = sh_mem[2 * tid];
        nums[2 * gid + 1] = sh_mem[2 * tid + 1];
    }

    if (tid == blockDim.x - 1 && blockSums != nullptr) {
        blockSums[2 * blockIdx.x]     = sh_mem[2 * tid];
        blockSums[2 * blockIdx.x + 1] = sh_mem[2 * tid + 1];
    }
}

__global__ void addOffsets(float *nums, int N, float *blockOffsets){
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + tid;
    if (gid >= N) return;

    float off_r = blockOffsets[2 * blockIdx.x];
    float off_i = blockOffsets[2 * blockIdx.x + 1];

    float cur_r = nums[2 * gid];
    float cur_i = nums[2 * gid + 1];

    float res_r, res_i;
    res_r = off_r * cur_r - off_i * cur_i;
    res_i = off_r * cur_i + off_i * cur_r;

    nums[2 * gid]     = res_r;
    nums[2 * gid + 1] = res_i;
}


int main() {
    
    size_t size = 1 << 24; 
    float *in_d, *in_h, *out_d, *out_h;
    int N = size / 2;
    int threadsPerBlock = 128;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate on host
    in_h = (float *)calloc(size, sizeof(float));
    CHECK_ALLOC(in_h);
    out_h = (float *)calloc(size, sizeof(float));
    CHECK_ALLOC(out_h);

    // Allocate on device
    CUDA_CALL(cudaMalloc((void **)&in_d, size * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&out_d, size * sizeof(float)));


    float *blockSums_d = nullptr;    
    float *blockOffsets_d = nullptr; //exclusive prefix
    float *blockSums_h = (float*)malloc(sizeof(float)*2*blocks);
    float *blockOffsets_h = (float*)malloc(sizeof(float)*2*blocks);
    CHECK_ALLOC(blockSums_h);
    CHECK_ALLOC(blockOffsets_h);
    CUDA_CALL(cudaMalloc(&blockSums_d, sizeof(float)*2*blocks));
    CUDA_CALL(cudaMalloc(&blockOffsets_d, sizeof(float)*2*blocks));

    // Initialize input
    int e = random_init(size, in_d, in_h);
    if (e == EXIT_FAILURE)
        return EXIT_FAILURE;

    auto start = std::chrono::system_clock::now();
    sequential_scan(size, in_h, out_h);
    auto end = std::chrono::system_clock::now();

    std::cout << "First 3 entries of In Vec:" << std::endl;
    for (int32_t i = 0; i < 3 * 2; i += 2)
        std::cout << in_h[i] << "," << in_h[i + 1] << std::endl;
    std::cout << "First 3 entries of Out Vec (CPU):" << std::endl;
    for (int32_t i = 0; i < 3 * 2; i += 2)
        std::cout << out_h[i] << " + " << out_h[i + 1] << "i" << std::endl;

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "CPU Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;


    // Copy input to a working array on device
    CUDA_CALL(cudaMemcpy(in_d, in_h, sizeof(float) * size, cudaMemcpyHostToDevice));

    // per-block inclusive scan, producing block sums
    size_t shmem_bytes = sizeof(float) * 2 * threadsPerBlock;
    scan<<<blocks, threadsPerBlock, shmem_bytes>>>(in_d, N, blockSums_d);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    // compute exclusive prefix
    CUDA_CALL(cudaMemcpy(blockSums_h, blockSums_d, sizeof(float) * 2 * blocks, cudaMemcpyDeviceToHost));

    for (int b = 0; b < blocks; ++b){
        if (b == 0) {
            blockOffsets_h[2*b]     = 1.0f; 
            blockOffsets_h[2*b + 1] = 0.0f; 
        } else {
            float a_r = blockOffsets_h[2*(b-1)];
            float a_i = blockOffsets_h[2*(b-1)+1];
            float b_r = blockSums_h[2*(b-1)];
            float b_i = blockSums_h[2*(b-1)+1];
            float res_r = a_r * b_r - a_i * b_i;
            float res_i = a_r * b_i + a_i * b_r;
            blockOffsets_h[2*b]     = res_r;
            blockOffsets_h[2*b + 1] = res_i;
        }
    }

    // copy offsets to device
    CUDA_CALL(cudaMemcpy(blockOffsets_d, blockOffsets_h, sizeof(float) * 2 * blocks, cudaMemcpyHostToDevice));

    // multiply per-block results by offsets
    cudaEvent_t startG, stopG;
    cudaEventCreate(&startG);
    cudaEventCreate(&stopG);
    cudaEventRecord(startG);

    addOffsets<<<blocks, threadsPerBlock>>>(in_d, N, blockOffsets_d);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    cudaEventRecord(stopG);
    cudaEventSynchronize(stopG);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, startG, stopG);
    std::cout << "GPU time: " << ms << " ms\n";
    // copy GPU result back to host
    float *gpu_out = (float *)malloc(sizeof(float) * size);
    CHECK_ALLOC(gpu_out);
    CUDA_CALL(cudaMemcpy(gpu_out, in_d, sizeof(float) * size, cudaMemcpyDeviceToHost));

    std::cout << "First 3 entries of Out Vec (GPU):" << std::endl;
    for (int32_t i = 0; i < 3 * 2; i += 2)
        std::cout << gpu_out[i] << " + " << gpu_out[i + 1] << "i" << std::endl;

    // Verification
    int mismatches = 0;
    const float REL_EPS = 1e-3f;
    const float ABS_EPS = 1e-3f;

    for (int i = 0; i < size; i += 2) {
        float cr = out_h[i];
        float ci = out_h[i + 1];
        float gr = gpu_out[i];
        float gi = gpu_out[i + 1];

        float absErrR = std::abs(cr - gr);
        float absErrI = std::abs(ci - gi);

        float relErrR = absErrR / std::max(1.0f, std::abs(cr));
        float relErrI = absErrI / std::max(1.0f, std::abs(ci));

        if ((absErrR > ABS_EPS && relErrR > REL_EPS) ||
            (absErrI > ABS_EPS && relErrI > REL_EPS)) 
        {
            if (mismatches < 10) {
                std::cout << std::format(
                    "Mismatch at idx {}:\n"
                    "CPU = {} + {}i\nGPU = {} + {}i\n"
                    "Absolute Error = ({}, {})\nRelative Error = ({}, {})\n",
                    i/2, cr, ci, gr, gi, absErrR, absErrI, relErrR, relErrI
                );
            }
            mismatches++;
        }
    }

    if (mismatches == 0) 
        std::cout << "Verification PASSED!\n";
    else
        std::cout << "Verification FAILED: " << mismatches << " mismatches\n";

    // cleanup
    CUDA_CALL(cudaFree(in_d));
    CUDA_CALL(cudaFree(out_d));
    CUDA_CALL(cudaFree(blockSums_d));
    CUDA_CALL(cudaFree(blockOffsets_d));
    free(in_h);
    free(out_h);
    free(blockSums_h);
    free(blockOffsets_h);
    free(gpu_out);

    return EXIT_SUCCESS;
}
