#include <cmath>
#include <cstdio>

#include "cuda_error.cuh"

typedef float real;
// typedef double real;

const int N = 1 << 22;
const int M = sizeof(real) * N;
const int MAX_NUM_STREAMS = 64;

cudaStream_t streams[MAX_NUM_STREAMS];

__global__ void add(const real *x, const real *y, real *z, int N) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        z[tid] = x[tid] + y[tid];
    }
}

void check(const real *h_z, const int N) {
    bool has_error = false;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_z[i] - 3.57) > 1e-6) {
            has_error = true;
            printf("h_z[%d] = %g\n", i, h_z[i]);
            break;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

int main(void) {
    real *h_x, *h_y, *h_z;
    CHECK_CUDA_CALL(cudaMallocHost(&h_x, M));
    CHECK_CUDA_CALL(cudaMallocHost(&h_y, M));
    CHECK_CUDA_CALL(cudaMallocHost(&h_z, M));
    for (int i = 0; i < N; i ++) {
        h_x[i] = 1.23;
        h_y[i] = 2.34;
    }

    real *d_x, *d_y, *d_z;
    CHECK_CUDA_CALL(cudaMalloc(&d_x, M));
    CHECK_CUDA_CALL(cudaMalloc(&d_y, M));
    CHECK_CUDA_CALL(cudaMalloc(&d_z, M));

    for (int i = 0; i < MAX_NUM_STREAMS; i++) {
        CHECK_CUDA_CALL(cudaStreamCreate(&streams[i]));
    }

    for (int num = 1; num <= MAX_NUM_STREAMS; num *= 2) {
        printf("Using %d streams:\n", num);

        int N1 = N / num;
        int M1 = M / num;
        for (int i = 0; i < num; i++) {
            int offset = i * N1;
            CHECK_CUDA_CALL(cudaMemcpyAsync(d_x + offset, h_x + offset, M1, cudaMemcpyHostToDevice, streams[i]));
            CHECK_CUDA_CALL(cudaMemcpyAsync(d_y + offset, h_y + offset, M1, cudaMemcpyHostToDevice, streams[i]));

            int block_size = 128;
            int grid_size = (N1 - 1) / block_size + 1;
            add<<<grid_size, block_size, 0, streams[i]>>>(d_x + offset, d_y + offset, d_z + offset, N1);

            CHECK_CUDA_CALL(cudaMemcpyAsync(h_z + offset, d_z + offset, M1, cudaMemcpyDeviceToHost, streams[i]));
        }

        // 这里演示流同步第一种之主机等待 stream 完成： 
        // 由于所有的数据传输和核函数执行都是在非空流中进行的，如果主机要对最终结果进行检查，就必须等待所有的 stream 完成操作。
        // 这里主机调用 cudaStreamSynchronize(streams[i]) 阻塞主机直到所有 stream 完成操作，然后主机才对结果进行检查，此时检查结果就是正确的。
        for (int i = 0; i < num; ++i) {
            CHECK_CUDA_CALL(cudaStreamSynchronize(streams[i]));
        }
        check(h_z, N);
    }

    for (int i = 0; i < MAX_NUM_STREAMS; i ++) {
        CHECK_CUDA_CALL(cudaStreamDestroy(streams[i]));
    }

    CHECK_CUDA_CALL(cudaFreeHost(h_x));
    CHECK_CUDA_CALL(cudaFreeHost(h_y));
    CHECK_CUDA_CALL(cudaFreeHost(h_z));
    CHECK_CUDA_CALL(cudaFree(d_x));
    CHECK_CUDA_CALL(cudaFree(d_y));
    CHECK_CUDA_CALL(cudaFree(d_z));
    return 0;
}