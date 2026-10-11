#include <cmath>
#include <cstdio>

#include "cuda_error.cuh"

typedef float real;
// typedef double real;

const int NUM_REPEATS = 10;
const int N1 = 1024;
const int MAX_NUM_STREAMS = 10;

const int N = N1 * MAX_NUM_STREAMS;
const int M = sizeof(real) * N;
const int block_size = 128;
const int grid_size = (N1 - 1) / block_size + 1;

cudaStream_t streams[MAX_NUM_STREAMS];

__global__ void add(const real *d_x, const real *d_y, real *d_z) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N1) {
        // 提示：这里的循环只是为了增加耗时
        for (int i = 0; i < 1e5; ++i) {
            d_z[tid] = d_x[tid] + d_y[tid];
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

        for (int i = 0; i < num; i++) {
            int offset = i * N1;
            // CUDA 核函数有且只有三种调用方式，分别如下：
            // 1. kernel<<<grid_size, block_size>>>(args...);
            // 2. kernel<<<grid_size, block_size, shared_mem_size>>>(args...);
            // 3. kernel<<<grid_size, block_size, shared_mem_size, stream>>>(args...);
            // 下面的写法是错误的，请注意：
            // kernel<<<grid_size, block_size, stream>>>(args...);

            // d_x, d_y, d_z 是长度为 N 的数组，N = N1 * MAX_NUM_STREAMS
            // MAX_NUM_STREAMS 个非空流各自处理 N1 长度的数据元素
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


// 一个 CUDA 流指的是由主机发出的在一个设备中执行的 CUDA 操作序列，如主机－设备数据传输和核函数执行。
// 一个 CUDA 流中各个操作的次序是由主机控制的，按照主机发布的次序执行，而且同一个 CUDA 流中的所有 CUDA 操作都是顺序执行的。
// 然而，来自于两个不同 CUDA 流中的操作不一定按照某个次序执行，而有可能并发或交错地执行。
// 任何 CUDA 操作都存在于某个 CUDA 流中，要么是默认流（default stream），也称为空流（null stream），要么是明确指定的非空流，非空流由主机端负责创建和销毁。
// 前面几章所有的样例，包括本目录的 09_cuda_stream/host_kernel.cu，使用的都是默认流。而本样例将演示非空流的创建和销毁
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
    CHECK_CUDA_CALL(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));
    
    for (int i = 0; i < MAX_NUM_STREAMS; i++) {
        // 一个 CUDA 流由类型为 cudaStream_t 的变量表示，由 CUDA 运行时 API 函数创建和销毁，两个函数签名如下：
        // cudaError_t cudaStreamCreate(cudaStream_t*);
        // cudaError_t cudaStreamDestroy(cudaStream_t);
        // 一个典型的非空流的创建和销毁过程如下：
        // cudaStream_t stream;
        // cudaStreamCreate(&stream);
        // ...
        // cudaStreamDestroy(stream);

        // 本样例中，streams 是一个全局变量，类型为 cudaStream_t 数组: cudaStream_t streams[MAX_NUM_STREAMS];
        // 这里创建了 MAX_NUM_STREAMS 个非空流
        CHECK_CUDA_CALL(cudaStreamCreate(&streams[i]));
    }

    // 下面的循环演示了如何使用不同数量的非空流来执行相同的核函数 add
    // 在我的机器上，五个以内的非空流的执行时间几乎相同，五个以上的非空流的执行时间明显增加，见 09_cuda_stream/kernel_kernel_result.txt
    // 可见，合理的使用非空流可以加速运算，让算力充分发挥
    for (int i = 1; i <= MAX_NUM_STREAMS; i++) {
        printf("Timing with %d streams:\n", i);
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