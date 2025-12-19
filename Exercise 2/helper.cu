#include <iostream>
#include <stdlib.h>
#include <curand.h>
#include <chrono>

#define CUDA_CALL(x)                                                                                          \
    do                                                                                                        \
    {                                                                                                         \
        cudaError_t error = x;                                                                                \
        if (error != cudaSuccess)                                                                             \
        {                                                                                                     \
            const char *cuda_err_str = cudaGetErrorString(error);                                             \
            std::cerr << "Cuda Error at" << __FILE__ << ":" << __LINE__ << ": " << cuda_err_str << std::endl; \
            return EXIT_FAILURE;                                                                              \
        }                                                                                                     \
    } while (0)

#define CURAND_CALL(x)                                                                                    \
    do                                                                                                    \
    {                                                                                                     \
        curandStatus_t error = x;                                                                         \
        if (error != CURAND_STATUS_SUCCESS)                                                               \
        {                                                                                                 \
            std::cerr << "CudaRand Error " << error << " at" << __FILE__ << ":" << __LINE__ << std::endl; \
            return EXIT_FAILURE;                                                                          \
        }                                                                                                 \
    } while (0)

#define CHECK_ALLOC(x)                                                                 \
    do                                                                                 \
    {                                                                                  \
        if ((x) == NULL)                                                               \
        {                                                                              \
            std::cerr << "Alloc Error at" << __FILE__ << ":" << __LINE__ << std::endl; \
            return EXIT_FAILURE;                                                       \
        }                                                                              \
    } while (0)

__global__ void scale_vec(float *in_d, float x, size_t size)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        in_d[i] *= x;
    }
}

int random_init(size_t size, float *in_d, float *in_h)
{
    curandGenerator_t gen;
    // Create PRNG
    CURAND_CALL(curandCreateGenerator(&gen,
                                      CURAND_RNG_PSEUDO_DEFAULT));
    // Set seed
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen,
                                                   2048ULL));

    // Generate size floats on device
    CURAND_CALL(curandGenerateUniform(gen, in_d, size));

    // Scale by two so it does not get boring
    scale_vec<<<ceil(size / 256.0), 256>>>(in_d, 2.0, size);

    // Copy device memory to host
    CUDA_CALL(cudaMemcpy(in_h, in_d, size * sizeof(float),
                         cudaMemcpyDeviceToHost));

    CURAND_CALL(curandDestroyGenerator(gen));

    return EXIT_SUCCESS;
}