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

    double *d_px, *d_py, *d_pz;
    CHECK_CUDA_CALL(cudaMalloc((void **)&d_px, M));
    CHECK_CUDA_CALL(cudaMalloc((void **)&d_py, M));
    CHECK_CUDA_CALL(cudaMalloc((void **)&d_pz, M));
    // 这里故意将 cudaMemcpyHostToDevice 错写成 cudaMemcpyDeviceToHost，测试 CHECK_CUDA_CALL 宏的功能
    // CHECK_CUDA_CALL(cudaMemcpy(d_px, h_px, M, cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaMemcpy(d_px, h_px, M, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(d_py, h_py, M, cudaMemcpyHostToDevice));

    // 由于 block 最大为 1024, 这里故意写成 1280, 测试核函数调用失败的情况
    // const int block_size = 1280;
    const int block_size = 128;
    const int grid_size = (N % block_size == 0) ? (N / block_size) : (N / block_size + 1);
    printf("grid_size = %d, block_size = %d\n", grid_size, block_size);
    add<<<grid_size, block_size>>>(d_px, d_py, d_pz, N);
    // 这两句是检查核函数调用的错误，cudaGetLastError() 的作用是在主机与设备同步之前，捕捉最后一个错误
    // 通常情况下，主机异步调用核函数，调用后立即执行下一条语句，
    // 因此，捕捉核函数调用的错误需要在主机与设备同步之前，调用 cudaGetLastError() 函数。
    // 除了显示调用 cudaDeviceSynchronize() 实现主机与设备同步，还可以临时设置环境变量 CUDA_LAUNCH_BLOCKING=1，强制主机与设备同步。
    // 设置之后，所有核函数的调用都变为同步的，主机在调用核函数之后，必须等待设备执行完核函数之后，才能继续执行下一条语句。
    // CHECK_CUDA_CALL(cudaGetLastError());
    // CHECK_CUDA_CALL(cudaDeviceSynchronize());

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
