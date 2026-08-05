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

void add(const real *px, const real *py, real *pz, const int N) {
    for (int i = 0; i < N; ++i) {
        pz[i] = px[i] + py[i];
    }
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

    real *px = (real *)malloc(M);
    real *py = (real *)malloc(M);
    real *pz = (real *)malloc(M);

    for (int i = 0; i < N; ++i) {
        px[i] = a;
        py[i] = b;
    }

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

        add(px, py, pz, N);

        CHECK_CUDA_CALL(cudaEventRecord(stop));
        CHECK_CUDA_CALL(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK_CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));
        
        printf("CPU add() elapsed time: %f ms\n", elapsed_time);

        // 由于第一次用于预热，不参与计算
        if (repeat > 0) {
            t1_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK_CUDA_CALL(cudaEventDestroy(start));
        CHECK_CUDA_CALL(cudaEventDestroy(stop));
    }

    // 计算平均时间和误差
    // 针对双精度浮点数，我的机器的运行结果是：
    // CPU add() average elapsed time: 26.976746 ms, error: 0.986945 ms
    const float t_ave = t1_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("CPU add() average elapsed time: %f ms, error: %f ms\n", t_ave, t_err);
    
    check(pz, N);

    free(px);
    free(py);
    free(pz);
    return 0;
}
