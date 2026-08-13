#include <cmath>
#include <cstdio>

#include "cuda_error.cuh"

const double EPSILON = 1e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

__global__ void add(const double  *px, const double *py, double *pz, const int N) {
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

    // 这里使用 cudaMalloc 申请的三个内存，就是全局内存，global memmory，
    // 全局内存的作用域是整个网格，生命周期是整个应用程序（由主机分配和释放），位置在 GPU 设备，可读可写。
    // 全局内存在 GPU 片外，速度不快，但容量最大，基本就是显存的大小，
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
