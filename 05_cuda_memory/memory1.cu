#include <cmath>
#include <cstdio>

#include "cuda_error.cuh"

const double EPSILON = 1e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

__global__ void add(const double  *px, const double *py, double *pz, const int N) {
    // 寄存器(register)的作用域是单个线程，生命周期是所属线程的生命周期，位置在 SM 上，且位于 GPU 片上，可读可写，
    // 寄存器由于在 gpu 片上，因此是所有内存中访问速度最高的，但数量也是最小的，
    // 一个寄存器占 4 个字节，32 bit，通常情况单个线程寄存器数量上限是 255 个，单个线程块寄存器数量上限是 64×1024 个，一个 SM 的寄存器数量上限是 64×1024 个。
    // 局部内存（local memory），也叫本地内存，作用域是单个线程，生命周期是所属线程的生命周期，位置在 GPU 设备，但位于 GPU 片外，可读可写，
    // 局部内存的用法与寄存器一样，但局部内存在硬件上属于全局内存的一部分，每个线程最多能使用 512KB ，速度不快。
    // 类比：CUDA 寄存器和局部内存类似 C++ 中的局部变量所用的栈内存
    // 在核函数中，不加任何限定符的变量，一般就存储在寄存器中，比如这里的 tid，
    // 如果是不加任何限定符的数组，有可能在寄存器中，也有可能存放在局部内存中，这由编译器自动实现。
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // printf("tid = %d\n", tid);
    if (tid >= N) {
        return;
    }
    pz[tid] = px[tid] + py[tid];
}

void check(const double *pz, const int N) {
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
    const int M = sizeof(double) * N;

    double *h_px = (double *)malloc(M);
    double *h_py = (double *)malloc(M);
    double *h_pz = (double *)malloc(M);

    for (int i = 0; i < N; ++i) {
        h_px[i] = a;
        h_py[i] = b;
    }

    // 全局内存（global memmory）的作用域是整个网格，生命周期是整个应用程序（由主机分配和释放），位置在 GPU 设备，但位于片外，可读可写。
    // 全局内存的访问速度不快，但容量最大，基本就是显存的大小，
    // 类比： CUDA 的全家内存不是 C++ 中的全局变量所在的全局数据区，更类似 C++ 中的堆内存。
    // 这里使用 cudaMalloc 申请的三个内存，就是全局内存
    double *d_px, *d_py, *d_pz;
    CHECK_CUDA_CALL(cudaMalloc((void **)&d_px, M));
    CHECK_CUDA_CALL(cudaMalloc((void **)&d_py, M));
    CHECK_CUDA_CALL(cudaMalloc((void **)&d_pz, M));
    CHECK_CUDA_CALL(cudaMemcpy(d_px, h_px, M, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(d_py, h_py, M, cudaMemcpyHostToDevice));

    const int block_size = 128;
    const int grid_size = (N % block_size == 0) ? (N / block_size) : (N / block_size + 1);
    printf("grid_size = %d, block_size = %d\n", grid_size, block_size);
    add<<<grid_size, block_size>>>(d_px, d_py, d_pz, N);

    CHECK_CUDA_CALL(cudaMemcpy(h_pz, d_pz, M, cudaMemcpyDeviceToHost));
    check(h_pz, N);

    free(h_px);
    free(h_py);
    free(h_pz);
    CHECK_CUDA_CALL(cudaFree(d_px));
    CHECK_CUDA_CALL(cudaFree(d_py));
    CHECK_CUDA_CALL(cudaFree(d_pz));
    return 0;
}
