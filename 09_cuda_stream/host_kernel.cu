#include <cmath>
#include <cstdlib>

#include "cuda_error.cuh"

// typedef float real;
typedef double real;

const int NUM_REPEATS = 10;
const int N = 1e7;
const int M = sizeof(real) * N;
const int block_size = 128;
const int grid_size = (N - 1)/block_size + 1;

void cpu_sum(const real *x, const real *y, real *z, const int N_host) {
    for (int i = 0; i < N_host; i++) {
        z[i] = x[i] + y[i];
    }
}

__global__ void gpu_sum(const real *x, const real *y, real *z) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        z[tid] = x[tid] + y[tid];
    }
}

void timing(const real *h_x, const real *h_y, real *h_z, 
            const real *d_x, const real *d_y, real *d_z,
            const int ratio, bool overlap) {
    float t_sum = 0;
    float t2_sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat) {
        cudaEvent_t start, stop;
        CHECK_CUDA_CALL(cudaEventCreate(&start));
        CHECK_CUDA_CALL(cudaEventCreate(&stop));
        CHECK_CUDA_CALL(cudaEventRecord(start));
        cudaEventQuery(start);

        if (!overlap) {
            cpu_sum(h_x, h_y, h_z, N/ratio);
        }
        gpu_sum<<<grid_size, block_size>>>(d_x, d_y, d_z);

        if (overlap) {
            cpu_sum(h_x, h_y, h_z, N/ratio);
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
    real *h_z = (real *)malloc(M);
    for (int i = 0; i < N; i ++) {
        h_x[i] = 1.23;
        h_y[i] = 2.34;
    }

    real *d_x, *d_y, *d_z;
    CHECK_CUDA_CALL(cudaMalloc(&d_x, M));
    CHECK_CUDA_CALL(cudaMalloc(&d_y, M));
    CHECK_CUDA_CALL(cudaMalloc(&d_z, M));

    CHECK_CUDA_CALL(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

    printf("without cpu-gpu overlap(ratio = 10)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 10, false);
    printf("with cpu-gpu overlap(ratio = 10)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 10, true);

    printf("without cpu-gpu overlap(ratio = 1)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 1, false);
    printf("with cpu-gpu overlap(ratio = 1)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 1, true);

    printf("without cpu-gpu overlap(ratio = 1000)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 1000, false);
    printf("with cpu-gpu overlap(ratio = 1000)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 1000, true);

    free(h_x);
    free(h_y);
    free(h_z);
    CHECK_CUDA_CALL(cudaFree(d_x));
    CHECK_CUDA_CALL(cudaFree(d_y));
    CHECK_CUDA_CALL(cudaFree(d_z));
    return 0;
}