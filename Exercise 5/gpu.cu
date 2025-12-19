#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>

#define N 128                // Grid size X
#define M 128                // Grid size Y
#define ITERATIONS 100000    // Number of iterations
#define DIFFUSION_FACTOR 0.5 // Diffusion factor
#define CELL_SIZE 0.01       // Cell size for the simulation

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

// CPU initialization
void initializeGrid(float *grid, int n, int m)
{
    for (int y = 0; y < m; ++y)
    {
        for (int x = 0; x < n; ++x)
        {
            if (y > m/2 && x > n/2)
                grid[y*n + x] = 100.0f;
            else
                grid[y*n + x] = 0.0f;
        }
    }
}

// CPU reference simulation
void heatSimulation(float* curr, float* next, int n, int m, int iterations, float dt)
{
    float dx2 = CELL_SIZE*CELL_SIZE;
    float dy2 = CELL_SIZE*CELL_SIZE;

    for (int iter=0; iter<iterations; ++iter)
    {
        for (int y=1; y<m-1; ++y)
        {
            for (int x=1; x<n-1; ++x)
            {
                float center = curr[y*n + x];
                float left   = curr[y*n + x-1];
                float right  = curr[y*n + x+1];
                float below  = curr[(y-1)*n + x];
                float above  = curr[(y+1)*n + x];
                next[y*n + x] = center + DIFFUSION_FACTOR*dt*((left-2*center+right)/dy2 + (above-2*center+below)/dx2);
            }
        }
        std::swap(curr, next);
    }
}

// CUDA kernel
__global__ void heatKernel(float* curr, float* next, int n, int m, float dt, float dx2, float dy2)
{
    extern __shared__ float tile[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x*blockDim.x + tx;
    int y = blockIdx.y*blockDim.y + ty;

    int tile_width  = blockDim.x + 2;
    

    // Global to shared memory indices
    int sidx = (ty+1)*tile_width + (tx+1);

    // Load center
    if (x<n && y<m) tile[sidx] = curr[y*n + x];

    // Load halo cells
    if (tx==0 && x>0) tile[sidx-1] = curr[y*n + x-1];             // left
    if (tx==blockDim.x-1 && x<n-1) tile[sidx+1] = curr[y*n + x+1]; // right
    if (ty==0 && y>0) tile[sidx-tile_width] = curr[(y-1)*n + x];  // top
    if (ty==blockDim.y-1 && y<m-1) tile[sidx+tile_width] = curr[(y+1)*n + x];

    // Load corners
    if (tx==0 && ty==0 && x>0 && y>0) tile[sidx-tile_width-1] = curr[(y-1)*n + x-1];
    if (tx==blockDim.x-1 && ty==0 && x<n-1 && y>0) tile[sidx-tile_width+1] = curr[(y-1)*n + x+1];
    if (tx==0 && ty==blockDim.y-1 && x>0 && y<m-1) tile[sidx+tile_width-1] = curr[(y+1)*n + x-1];
    if (tx==blockDim.x-1 && ty==blockDim.y-1 && x<n-1 && y<m-1) tile[sidx+tile_width+1] = curr[(y+1)*n + x+1];

    __syncthreads();

    if (x>0 && x<n-1 && y>0 && y<m-1)
    {
        float center = tile[sidx];
        float left   = tile[sidx-1];
        float right  = tile[sidx+1];
        float below  = tile[sidx-tile_width];
        float above  = tile[sidx+tile_width];

        next[y*n + x] = center + DIFFUSION_FACTOR*dt*((left-2*center+right)/dy2 + (above-2*center+below)/dx2);
    }
}

int main()
{
    // CPU grids
    float* h_curr = (float*)malloc(N*M*sizeof(float));
    float* h_next = (float*)malloc(N*M*sizeof(float));
    initializeGrid(h_curr,N,M);
    initializeGrid(h_next,N,M);

    float dx2 = CELL_SIZE*CELL_SIZE;
    float dy2 = CELL_SIZE*CELL_SIZE;
    float dt = dx2*dy2/(2.0f*DIFFUSION_FACTOR*(dx2+dy2));

    float* h_ref = (float*)malloc(N*M*sizeof(float));
    memcpy(h_ref, h_curr, N*M*sizeof(float));
    heatSimulation(h_ref,h_next,N,M,ITERATIONS,dt);

    // GPU 
    float *d_curr,*d_next;
    CUDA_CALL(cudaMalloc(&d_curr,N*M*sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_next,N*M*sizeof(float)));
    CUDA_CALL(cudaMemcpy(d_curr,h_curr,N*M*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_next,h_next,N*M*sizeof(float),cudaMemcpyHostToDevice));

    dim3 block(16,16);
    dim3 grid((N+block.x-1)/block.x, (M+block.y-1)/block.y);
    size_t sharedMemSize = (block.x+2)*(block.y+2)*sizeof(float);

    for(int iter=0; iter<ITERATIONS; ++iter)
    {
        heatKernel<<<grid,block,sharedMemSize>>>(d_curr,d_next,N,M,dt,dx2,dy2);
        
        cudaDeviceSynchronize();
        std::swap(d_curr,d_next);
    }

    float* h_result = (float*)malloc(N*M*sizeof(float));
    CUDA_CALL(cudaMemcpy(h_result,d_curr,N*M*sizeof(float),cudaMemcpyDeviceToHost));

    // Max absolute error
    float maxErr=0.0f;
    for(int i=0;i<N*M;i++) maxErr = fmaxf(maxErr,fabs(h_result[i]-h_ref[i]));
    std::cout << "Max absolute error (CPU vs GPU): " << maxErr << "\n\n";

    std::cout << "Final grid (top-left 16x16):\n";
    for(int y=0;y<16;y++)
    {
        for(int x=0;x<16;x++)
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) << h_result[y*N + x] << " ";
        std::cout << "\n";
    }

    free(h_curr); free(h_next); free(h_ref); free(h_result);
    cudaFree(d_curr); cudaFree(d_next);

    return 0;
}
