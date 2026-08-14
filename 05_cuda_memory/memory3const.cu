#include <cstdio>

#include "cuda_error.cuh"

#define CONST_SIZE 8

// 常量内存（constant memory）是一块特殊的全局内存，只读不可写，容量小（64KB），带有缓存，访问速度比全局内存高，
// 常量内存的作用域是整个网格，生命周期是整个应用程序（由主机分配和释放），位置在 GPU 设备，片外。
// 常量内存使用 __constant__ 修饰，属于设备，在所有主机和设备函数外定义，
// 下面是常量内存通用的两种定义方式（第一种用的最多）：
__constant__ float const_scalar;
__constant__ float const_scale[CONST_SIZE];

// 这里的参数 N（传值），也是放在常量内存中;
// 也可以传结构体（传值），也是放在常量内存中;
__global__ void my_kernel(float* d_output, int N) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // printf("tid = %d\n", tid);
    if (tid >= N) {
        return;
    }
    // 核函数内可直接访问，可读不可写
    d_output[tid] = const_scale[tid] * const_scalar;
}

int main(void) {
    const int N = CONST_SIZE;
    const int M = sizeof(float) * N;

    float* h_input = (float*)malloc(M);
    float* h_output = (float*)malloc(M);

    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i % 100);
    }

    float *d_output;
    CHECK_CUDA_CALL(cudaMalloc((void **)&d_output, M));

    float scale_value = 3.14159f;
    
    // 常量内存同样使用 cudaMemcpyToSymbol，从主机端复制数据
    // 踩坑：注意第二个参数，要使用 & 取主机端变量的地址，但第一个常量内存的变量不用取地址
    CHECK_CUDA_CALL(cudaMemcpyToSymbol(const_scalar, &scale_value, sizeof(float)));
    CHECK_CUDA_CALL(cudaMemcpyToSymbol(const_scale, h_input, sizeof(float) * N));

    const int block_size = CONST_SIZE;
    const int grid_size = (N % block_size == 0) ? (N / block_size) : (N / block_size + 1);

    my_kernel<<<grid_size, block_size>>>(d_output, N);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    CHECK_CUDA_CALL(cudaMemcpy(h_output, d_output, M, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        printf("d_output[%d] = %f\n", i, h_output[i]);
    }

    free(h_input);
    free(h_output);
    CHECK_CUDA_CALL(cudaFree(d_output));
    return 0;
}