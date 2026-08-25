#include <cstdio>
#include <cstdint>

#include "cuda_error.cuh"

const int N = 64;

int main(void) {
    for (int i = 1; i <= N; i++) {
        const size_t mem_size = size_t(i) * 1024 * 1024 * 1024;
        uint16_t *px;

        // 我的机器的主机内存是 32G，显存是 1.8G。但这里可以申请 64G，而且成功了。
        // 原因是 cudaMallocManaged 调用成功，只是预订了一段地址空间，实际分配发生在第一次访问预订的空间，
        // 由于这里申请后就释放了，所以实际上并没有真正的分配统一内存。
        CHECK_CUDA_CALL(cudaMallocManaged(&px, mem_size));
        CHECK_CUDA_CALL(cudaFree(px));
        printf("Allocated %d GB unified memory without touch\n", i);

        // 我的机器的显存是 1.8G，因此这里执行到第二轮，申请 2G 时就失败了
        /*
        CUDA Error: 
        Error code: 2
        Error string: out of memory
        */
        // CHECK_CUDA_CALL(cudaMalloc(&px, mem_size));
        // CHECK_CUDA_CALL(cudaFree(px));
        // printf("Allocated %d GB device memory\n", i);
    }

    return 0;
}