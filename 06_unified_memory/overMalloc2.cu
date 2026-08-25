#include <cstdio>
#include <cstdint>

#include "cuda_error.cuh"

const int N = 64;

__global__ void gpu_touch(uint64_t *px, const size_t data_size) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < data_size) {
        px[tid] = 0;
    }
}

int main(void) {
    for (int i = 1; i <= N; i+=8) {
        const size_t mem_size = size_t(i) * 1024 * 1024 * 1024;
        const size_t data_size = mem_size / sizeof(uint64_t);
        
        uint64_t *px;
        // 我的机器的主机内存是 32G，显存是 1.8G。
        // 在我的机器上，使用统一内存能申请超过 2G 的空间，但无法超过 34G（32G+1.8G）
        /*
        CUDA Error: 
        Error code: 700
        Error string: an illegal memory access was encountered
        */
        CHECK_CUDA_CALL(cudaMallocManaged(&px, mem_size));

        const size_t block_size = 1024;
        const size_t gird_size = (data_size - 1) / block_size + 1;
        gpu_touch<<<gird_size, block_size>>>(px, data_size);
        CHECK_CUDA_CALL(cudaGetLastError());
        CHECK_CUDA_CALL(cudaDeviceSynchronize());


        CHECK_CUDA_CALL(cudaFree(px));
        printf("Allocated %d GB unified memory without touch\n", i);
    }

    return 0;
}