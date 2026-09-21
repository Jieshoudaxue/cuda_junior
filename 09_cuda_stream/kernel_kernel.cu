#include <cmath>
#include <cstdio>

#include "cuda_error.cuh"

typedef float real;
// typedef double real;

const int NUM_REPEATS = 10;
const int N1 = 1024;
const int MAX_NUM_STREAMS = 30;

const int N = N1 * MAX_NUM_STREAMS;
const int M = sizeof(real) * N;
const int block_size = 128;
const int grid_size = (N1 - 1) / block_size + 1;

cudaStream_t streams[MAX_NUM_STREAMS];

__global__ void add(const real *d_x, const real *d_y, real *d_z) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N1) {
        for (int i = 0; i < 1e5; i++) {
            d_z[i] = d_x[i] + d_y[i];
        }
    }
}

void timing(const real *d_x, const real *d_y, real *d_z, const int num) {
    float t_sum = 0;
    float t2_sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat) {
        cudaEvent_t start, stop;
        CHECK_CUDA_CALL(cudaEventCreate(&start));
        CHECK_CUDA_CALL(cudaEventCreate(&stop));
        CHECK_CUDA_CALL(cudaEventRecord(start));
        cudaEventQuery(start);

        for (int i = 0; i < num; i ++) {
            int offset = i * N1;
            add<<<grid_size, block_size, 0, streams[i]>>>(d_x + offset, d_y + offset, d_z + offset);
        }

        CHECK_CUDA_CALL(cudaEventRecord(stop));
        CHECK_CUDA_CALL(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK_CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms\n", elapsed_time);
        
        if (repeat > 0) {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK_CUDA_CALL(cudaEventDestroy(start));
        CHECK_CUDA_CALL(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);
}


int main(void) {
    real *h_x = (real *)malloc(M);
    real *h_y = (real *)malloc(M);
    for (int i = 0; i < N; i ++) {
        h_x[i] = 1.23;
        h_y[i] = 2.34;
    }

    real *d_x, *d_y, *d_z;
    CHECK_CUDA_CALL(cudaMalloc(&d_x, M));
    CHECK_CUDA_CALL(cudaMalloc(&d_y, M));
    CHECK_CUDA_CALL(cudaMalloc(&d_z, M));

    CHECK_CUDA_CALL(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    
    for (int i = 0; i < MAX_NUM_STREAMS; i++) {
        CHECK_CUDA_CALL(cudaStreamCreate(&streams[i]));
    }

    for (int i = 1; i <= MAX_NUM_STREAMS; i++) {
        timing(d_x, d_y, d_z, i);
    }

    for (int i = 0; i < MAX_NUM_STREAMS; i++) {
        CHECK_CUDA_CALL(cudaStreamDestroy(streams[i]));
    }

    free(h_x);
    free(h_y);
    CHECK_CUDA_CALL(cudaFree(d_x));
    CHECK_CUDA_CALL(cudaFree(d_y));
    CHECK_CUDA_CALL(cudaFree(d_z));
    return 0;
}