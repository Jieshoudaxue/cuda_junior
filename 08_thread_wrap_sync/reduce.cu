#include <cstdio>
#include <cstdint>
#include <cooperative_groups.h>

#include "cuda_error.cuh"

// typedef float real;
typedef double real;

const int NUM_REPEATS = 20;
const int N = 1e8;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;
const unsigned int FULL_MASK = 0xffffffff;

__global__ void reduce_syncwarp(const real *d_x, real *d_y) {
    const int tid = threadIdx.x;
    const int n = blockIdx.x * blockDim.x + tid;

    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncwarp();
    }

    if (tid == 0) {
        atomicAdd(d_y, s_y[0]);
    }
}

__global__ void reduce_shfl(const real *d_x, real *d_y) {
    const int tid = threadIdx.x;
    const int n = blockIdx.x * blockDim.x + tid;

    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    real y = s_y[tid];
    for (int offset = 16; offset > 0; offset >>= 1) {
        y += __shfl_down_sync(FULL_MASK, y, offset);
    }

    if (tid == 0) {
        atomicAdd(d_y, y);
    }
}

__global__ void reduce_cooperative_group(const real *d_x, real *d_y) {
    const int tid = threadIdx.x;
    const int n = blockIdx.x * blockDim.x + tid;

    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    real y = s_y[tid];
    cooperative_groups::thread_block_tile<32> g = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    for (int i = g.size() >> 1; i > 0; i >>= 1) {
        y += g.shfl_down(y, i);
    }

    if (tid == 0) {
        atomicAdd(d_y, y);
    }
}

real reduce(const real *d_x, const int method) {
    const int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int shared_mem_size = BLOCK_SIZE * sizeof(real);

    real h_y[1] = {0};
    real *d_y;
    CHECK_CUDA_CALL(cudaMalloc((void **)&d_y, sizeof(real)));
    CHECK_CUDA_CALL(cudaMemcpy(d_y, h_y, sizeof(real), cudaMemcpyHostToDevice));

    switch (method) {
        case 0:
            reduce_syncwarp<<<grid_size, BLOCK_SIZE, shared_mem_size>>>(d_x, d_y);
            break;
        case 1:
            reduce_shfl<<<grid_size, BLOCK_SIZE, shared_mem_size>>>(d_x, d_y);
            break;
        case 2:
            reduce_cooperative_group<<<grid_size, BLOCK_SIZE, shared_mem_size>>>(d_x, d_y);
            break;
        default:
            printf("Invalid method: %d\n", method);
            break;
    }

    CHECK_CUDA_CALL(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaFree(d_y));
    return h_y[0];
}

void timing(real *d_x, const int method) {
    real sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat) {
        cudaEvent_t start, stop;
        CHECK_CUDA_CALL(cudaEventCreate(&start));
        CHECK_CUDA_CALL(cudaEventCreate(&stop));
        CHECK_CUDA_CALL(cudaEventRecord(start));
        cudaEventQuery(start);
        
        sum = reduce(d_x, method);

        CHECK_CUDA_CALL(cudaEventRecord(stop));
        CHECK_CUDA_CALL(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK_CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms\n", elapsed_time);
        
        CHECK_CUDA_CALL(cudaEventDestroy(start));
        CHECK_CUDA_CALL(cudaEventDestroy(stop));
    }

    printf("sum = %f\n", sum);
}

int main(void) {
    real *h_x = (real *)malloc(M);
    for (int i = 0; i < N; i++) {
        h_x[i] = 1.23;
    }

    real *d_x;
    CHECK_CUDA_CALL(cudaMalloc((void **)&d_x, M));
    CHECK_CUDA_CALL(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    printf("\n using syncwarp: \n");
    timing(d_x, 0);
    printf("\n using shfl: \n");
    timing(d_x, 1);
    printf("\n using cooperative group: \n");
    timing(d_x, 2);

    CHECK_CUDA_CALL(cudaFree(d_x));
    free(h_x);
    return 0;
}