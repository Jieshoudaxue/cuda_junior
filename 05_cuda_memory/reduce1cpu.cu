#include <cstdio>

#include "cuda_error.cuh"

typedef float real;
// typedef double real;

const int NUM_REPEATS = 20;

real reduce(const real *x, const int N) {
    real sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += x[i];
    }
    return sum;
}

void timing(const real *x, const int N) {
    real sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat) {
        cudaEvent_t start, stop;
        CHECK_CUDA_CALL(cudaEventCreate(&start));
        CHECK_CUDA_CALL(cudaEventCreate(&stop));
        CHECK_CUDA_CALL(cudaEventRecord(start));
        cudaEventQuery(start);
        
        sum = reduce(x, N);

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
    const int N = 1e8;
    const int M = sizeof(real) * N;

    real *x = (real *)malloc(M);
    for (int i = 0; i < N; ++i) {
        x[i] = 1.23;
    }

    timing(x, N);

    free(x);
}
