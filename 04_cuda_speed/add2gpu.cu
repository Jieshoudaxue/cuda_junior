#include <cmath>
#include <cstdlib>
#include <cstdio>

#include "cuda_error.cuh"

typedef double real;
const real EPSILON = 1e-15;
// typedef float real;
// const real EPSILON = 1e-6;

const int NUM_REPEATS = 10;
const real a = 1.23;
const real b = 2.34;
const real c = 3.57;


__global__ void add(const real  *px, const real *py, real *pz, const int N) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // printf("tid = %d\n", tid);
    if (tid >= N) {
        return;
    }
    pz[tid] = px[tid] + py[tid];
}

void check(const real *pz, const int N) {
    bool has_error = false;
    for (int i = 0; i < N; ++i) {
        if (fabs(pz[i] - c) > EPSILON) {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

int main(void) {
    const int N = 1e7;
    const int M = sizeof(real) * N;

    real *h_px = (real *)malloc(M);
    real *h_py = (real *)malloc(M);
    real *h_pz = (real *)malloc(M);

    for (int i = 0; i < N; ++i) {
        h_px[i] = a;
        h_py[i] = b;
    }

    real *d_px, *d_py, *d_pz;
    cudaMalloc((void **)&d_px, M);
    cudaMalloc((void **)&d_py, M);
    cudaMalloc((void **)&d_pz, M);
    cudaMemcpy(d_px, h_px, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, h_py, M, cudaMemcpyHostToDevice);

    const int block_size = 128;
    const int grid_size = (N % block_size == 0) ? (N / block_size) : (N / block_size + 1);
    printf("grid_size = %d, block_size = %d\n", grid_size, block_size);

    // 利用 cuda event 计时，测试 CPU 版本的 add() 函数的运行时间
    float t1_sum = 0;
    float t2_sum = 0;
    // 循环 11 次，第一次用于预热，不参与计算，后面 10 次用于计时
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat) {
        cudaEvent_t start, stop;
        CHECK_CUDA_CALL(cudaEventCreate(&start));
        CHECK_CUDA_CALL(cudaEventCreate(&stop));
        CHECK_CUDA_CALL(cudaEventRecord(start));
        cudaEventQuery(start);

        // 被度量的 CUDA 核函数调用
        add<<<grid_size, block_size>>>(d_px, d_py, d_pz, N);

        CHECK_CUDA_CALL(cudaEventRecord(stop));
        CHECK_CUDA_CALL(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK_CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));
        
        printf("GPU add() elapsed time: %f ms\n", elapsed_time);

        // 由于第一次用于预热，不参与计算
        if (repeat > 0) {
            t1_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK_CUDA_CALL(cudaEventDestroy(start));
        CHECK_CUDA_CALL(cudaEventDestroy(stop));
    }

    // 计算平均时间和误差
    // 针对双精度浮点数，我的机器的运行结果是（相比 CPU 平均 26.976746 ms，GPU 快了约 8 倍）：
    // GPU add() average elapsed time: 3.254361 ms, error: 0.004883 ms
    // 针对单精度浮点数，我的机器的运行结果是：
    // GPU add() average elapsed time: 1.614038 ms, error: 0.002930 ms
    const float t_ave = t1_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("GPU add() average elapsed time: %f ms, error: %f ms\n", t_ave, t_err);

    cudaMemcpy(h_pz, d_pz, M, cudaMemcpyDeviceToHost);
    check(h_pz, N);

    free(h_px);
    free(h_py);
    free(h_pz);
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_pz);
    return 0;
}
