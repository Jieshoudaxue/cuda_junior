#include <cmath>
#include <cstdlib>
#include <cstdint>

#include "cuda_error.cuh"

typedef float real;
// typedef double real;

const int NUM_REPEATS = 10;
const int N = 1e8;
const int M = sizeof(real) * N;
const int block_size = 128;
const int grid_size = (N - 1)/block_size + 1;

enum OverlapMode : uint8_t {
    GPU_ONLY,
    CPU_GPU,
    GPU_CPU_OVERLAP
};

void cpu_sum(const real *x, const real *y, real *z) {
    int new_n = N / 20;
    for (int i = 0; i < new_n; i++) {
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
            OverlapMode mode) {
    float t_sum = 0;
    float t2_sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat) {
        cudaEvent_t start, stop;
        CHECK_CUDA_CALL(cudaEventCreate(&start));
        CHECK_CUDA_CALL(cudaEventCreate(&stop));
        CHECK_CUDA_CALL(cudaEventRecord(start));
        cudaEventQuery(start);

        switch (mode) {
            case GPU_ONLY:
                gpu_sum<<<grid_size, block_size>>>(d_x, d_y, d_z);
                break;
            case CPU_GPU:
                // 如果 cpu 计算在 gpu 之前调用，由于程序是顺序执行的，因此时间是两者之和，
                // 在这里例子中，gpu_sum 耗时大概是 16.2 ms, cpu_sum 耗时是 12.2 ms，这里的总时间是 28.4 ms
                cpu_sum(h_x, h_y, h_z);
                gpu_sum<<<grid_size, block_size>>>(d_x, d_y, d_z);
                break;
            case GPU_CPU_OVERLAP:
                // 如果 cpu 计算在 gpu 之后调用，由于核函数的启动是异步的，也叫非阻塞的，即主机调用 gpu_sum 后，不会等待核函数执行完毕，可以立即做别的事情。
                // 因此，cpu 和 gpu 同时计算，即 overlap ，两者的时间就会有一定的重叠。利用这个特性，可以对程序进行加速。
                // 在这里例子中，gpu_sum 耗时大概是 16.2 ms, cpu_sum 耗时是 12.2 ms，这里的总时间是 16.2 ms，即 cpu 耗时被 gpu 耗时完全遮盖。
                gpu_sum<<<grid_size, block_size>>>(d_x, d_y, d_z);
                cpu_sum(h_x, h_y, h_z);
                break;
            default:
                break;
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

    printf("GPU ONLY\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, GPU_ONLY);
    
    printf("cpu-gpu, no overlap\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, CPU_GPU);

    printf("gpu-cpu, overlap\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, GPU_CPU_OVERLAP);

    free(h_x);
    free(h_y);
    free(h_z);
    CHECK_CUDA_CALL(cudaFree(d_x));
    CHECK_CUDA_CALL(cudaFree(d_y));
    CHECK_CUDA_CALL(cudaFree(d_z));
    return 0;
}