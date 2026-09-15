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
const int GRID_SIZE = 10240;

__global__ void reduce_cooperative_group(const real *d_x, real *d_y, const int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    extern __shared__ real s_y[];

    real y = 0.0;
    const int stride = blockDim.x * gridDim.x;
    for (int i = bid * blockDim.x + tid ; i < N; i += stride) {
        y += d_x[i];
    }
    s_y[tid] = y;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    y = s_y[tid];

    cooperative_groups::thread_block_tile<32> cg = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    for (int i = cg.size() >> 1; i > 0; i >>= 1) {
        y += cg.shfl_down(y, i);
    }

    if (tid == 0) {
        d_y[bid] = y;
    }
}

__device__ real static_y[GRID_SIZE];

real reduce(const real *d_x) {
    real *d_y;
    CHECK_CUDA_CALL(cudaGetSymbolAddress((void**)&d_y, static_y));

    const int shared_mem_size = sizeof(real) * BLOCK_SIZE;

    reduce_cooperative_group<<<GRID_SIZE, BLOCK_SIZE, shared_mem_size>>>(d_x, d_y, N);
    reduce_cooperative_group<<<1, 1024, sizeof(real) * 1024>>>(d_y, d_y, GRID_SIZE);

    real h_y[1] = {0};
    CHECK_CUDA_CALL(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
    // CHECK_CUDA_CALL(cudaMemcpyFromSymbol(h_y, static_y, sizeof(real)));     // also ok

    return h_y[0];
}

void timing(real *d_x) {
    real sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat) {
        cudaEvent_t start, stop;
        CHECK_CUDA_CALL(cudaEventCreate(&start));
        CHECK_CUDA_CALL(cudaEventCreate(&stop));
        CHECK_CUDA_CALL(cudaEventRecord(start));
        cudaEventQuery(start);
        
        sum = reduce(d_x);

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
    CHECK_CUDA_CALL(cudaMalloc(&d_x, M));
    CHECK_CUDA_CALL(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    timing(d_x);

    free(h_x);
    CHECK_CUDA_CALL(cudaFree(d_x));
    return 0;
}