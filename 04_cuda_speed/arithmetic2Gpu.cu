#include <cmath>
#include <cstdlib>
#include <cstdio>

#include "cuda_error.cuh"

// typedef double real;
typedef float real;

const int NUM_REPEATS = 10;
const real x_flag = 100.0;
const real a = 1.23;

__global__ void arithmetic(real *x, const real x_flag, const int N) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        real x_tmp = x[tid];
        while (sqrt(x_tmp) < x_flag) {
            ++x_tmp;
        }
        x[tid] = x_tmp;
    }
}

int main(void) {
    const int N = 1e4;
    const int M = sizeof(real) * N;

    real *h_px = (real*)malloc(M);

    real *d_px;
    cudaMalloc((void **)&d_px, M);

    const int block_size = 128;
    const int grid_size = (N % block_size == 0) ? (N / block_size) : (N / block_size + 1);
    printf("grid_size = %d, block_size = %d\n", grid_size, block_size);

    // 利用 cuda event 计时，测试 CPU 版本的 add() 函数的运行时间
    float t1_sum = 0;
    float t2_sum = 0;
    // 循环 11 次，第一次用于预热，不参与计算，后面 10 次用于计时
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat) {
        for (int i = 0; i < N; ++i) {
            h_px[i] = a;
        }

        cudaEvent_t start, stop;
        CHECK_CUDA_CALL(cudaEventCreate(&start));
        CHECK_CUDA_CALL(cudaEventCreate(&stop));
        CHECK_CUDA_CALL(cudaEventRecord(start));
        cudaEventQuery(start);

        cudaMemcpy(d_px, h_px, M, cudaMemcpyHostToDevice);
        arithmetic<<<grid_size, block_size>>>(d_px, x_flag, N);
        cudaMemcpy(h_px, d_px, M, cudaMemcpyDeviceToHost);

        CHECK_CUDA_CALL(cudaEventRecord(stop));
        CHECK_CUDA_CALL(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK_CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));
        
        printf("GPU arithmetic() elapsed time: %f ms\n", elapsed_time);

        // 由于第一次用于预热，不参与计算
        if (repeat > 0) {
            t1_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK_CUDA_CALL(cudaEventDestroy(start));
        CHECK_CUDA_CALL(cudaEventDestroy(stop));
    }

    const float t_ave = t1_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("GPU arithmetic() average elapsed time: %f ms, error: %f ms\n", t_ave, t_err);

    free(h_px);
    cudaFree(d_px);
    return 0;
}
