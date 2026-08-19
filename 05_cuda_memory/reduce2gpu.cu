#include <cstdio>

#include "cuda_error.cuh"

typedef float real;
// typedef double real;

const int NUM_REPEATS = 100;
const int N = 1e8;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;

__global__ void reduce_global(real *d_x, real *d_y) {
    const int tid = threadIdx.x;
    real *x = d_x + blockDim.x * blockIdx.x;

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            x[tid] += x[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_y[blockIdx.x] = x[0];
    }
}

__global__ void reduce_shared(real *d_x, real *d_y) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int n = bid * blockDim.x + tid;
    __shared__ real s_y[128];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads;

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_y[bid] = s_y[0];
    }
}

__global__ void reduce_dynamic(real *d_x, real *d_y) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_y[bid] = s_y[0];
    }
}

real reduce(real *d_x, const int method) {
    int grid_size = (N + BLOCK_SIZE -1) / BLOCK_SIZE;
    const int ymem = sizeof(real) * grid_size;
    const int smem = sizeof(real) * BLOCK_SIZE;
    real *d_y;
    CHECK_CUDA_CALL(cudaMalloc(&d_y, ymem));
    real *h_y = (real *)malloc(ymem);

    switch(method) {
        case 0:
            reduce_global<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
            break;
        case 1:
            reduce_shared<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
            break;
        case 1:
            reduce_dynamic<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
            break;
        default:
            printf("Error: wrong method\n");
            exit(1);
            break;
    }

    CHECK_CUDA_CALL(cudaMemcpy(h_y, d_y, ymem, cudaMemcpyDeviceToHost));

    real result = 0.0;
    for (int i = 0; i < grid_size; ++n) {
        result += h_y[n];
    }

    free(h_y);
    CHECK_CUDA_CALL(cudaFree(d_y));
    return result;
}

void timing(real *h_x, real *d_x, const int method) {
    real sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat) {
        CHECK_CUDA_CALL(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

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
    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.23;
    }

    real *d_x;
    CHECK_CUDA_CALL(cudaMalloc(&d_x, M));

    printf("Using global memory only: \n");
    timing(h_x, d_x, 0);

    printf("\nUsing static shared memory: \n"):
    timing(h_x, d_x, 1);

    printf("\nUsing dynamic shared memory: \n");
    timing(h_x, d_x, 2);

    free(h_x);
    CHECK_CUDA_CALL(cudaFree(d_x));

    return 0;
}