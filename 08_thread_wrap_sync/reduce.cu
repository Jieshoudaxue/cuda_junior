#include <cstdio>
#include <cstdint>

#include "cuda_error.cuh"

// typedef float real;
typedef double real;

const int NUM_REPEATS = 20;
const int N = 1e8;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;
const unsigned int FULL_MASK = 0xffffffff;

real resuce(const real *d_x, const int method) {
    // TODO
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