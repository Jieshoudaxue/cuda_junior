#include <cmath>
#include <cstdio>

#include "cuda_error.cuh"

typedef float real;
// typedef double real;

const int NUM_REPEATS = 10;
const int N = 1 << 22;
const int M = sizeof(real) * N;
const int MAX_NUM_STREAMS = 64;

cudaStream_t streams[MAX_NUM_STREAMS];

__global__ void add(const real *x, const real *y, real *z, int N) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        for (int i = 0; i < 40; i ++) {
            z[tid] = x[tid] + y[tid];
        }
    }
}

void timing(const real *h_x, const real *h_y, real *h_z,
            real *d_x, real *d_y, real *d_z, const int num) {
    int N1 = N / num;
    int M1 = M / num;

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
            CHECK_CUDA_CALL(cudaMemcpyAsync(d_x + offset, h_x + offset, M1, cudaMemcpyHostToDevice, streams[i]));
            CHECK_CUDA_CALL(cudaMemcpyAsync(d_y + offset, h_y + offset, M1, cudaMemcpyHostToDevice, streams[i]));
            
            int block_size = 128;
            int grid_size = (N1 - 1) / block_size + 1;
            add<<<grid_size, block_size, 0, streams[i]>>>(d_x + offset, d_y + offset, d_z + offset, N1);

            CHECK_CUDA_CALL(cudaMemcpyAsync(h_z + offset, d_z + offset, M1, cudaMemcpyDeviceToHost, streams[i]));
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
    real *h_x, *h_y, *h_z;
    CHECK_CUDA_CALL(cudaMallocHost(&h_x, M));
    CHECK_CUDA_CALL(cudaMallocHost(&h_y, M));
    CHECK_CUDA_CALL(cudaMallocHost(&h_z, M));
    for (int i = 0; i < N; i ++) {
        h_x[i] = 1.23;
        h_y[i] = 2.34;
    }

    real *d_x, *d_y, *d_z;
    CHECK_CUDA_CALL(cudaMalloc(&d_x, M));
    CHECK_CUDA_CALL(cudaMalloc(&d_y, M));
    CHECK_CUDA_CALL(cudaMalloc(&d_z, M));

    for (int i = 0; i < MAX_NUM_STREAMS; i++) {
        CHECK_CUDA_CALL(cudaStreamCreate(&streams[i]));
    }

    for (int i = 1; i <= MAX_NUM_STREAMS; i *= 2) {
        timing(h_x, h_y, h_z, d_x, d_y, d_z, i);
    }

    for (int i = 0; i < MAX_NUM_STREAMS; i ++) {
        CHECK_CUDA_CALL(cudaStreamDestroy(streams[i]));
    }


    CHECK_CUDA_CALL(cudaFreeHost(h_x));
    CHECK_CUDA_CALL(cudaFreeHost(h_y));
    CHECK_CUDA_CALL(cudaFreeHost(h_z));
    CHECK_CUDA_CALL(cudaFree(d_x));
    CHECK_CUDA_CALL(cudaFree(d_y));
    CHECK_CUDA_CALL(cudaFree(d_z));
    return 0;
}