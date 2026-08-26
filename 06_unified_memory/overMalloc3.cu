#include <cstdio>
#include <cstdint>

#include "cuda_error.cuh"

const int N = 64;

void cpu_touch(uint64_t *px, const size_t data_size) {
    for (size_t i = 0; i < data_size; i++) {
        px[i] = 0;
    }
}

int main(void) {
    for (int i = 1; i <= N; i++) {
        const size_t mem_size = size_t(i) * 1024 * 1024 * 1024;
        const size_t data_size = mem_size / sizeof(uint64_t);
        
        uint64_t *px;
        // 我的机器的主机内存是 32G，显存是 1.8G。
        // 在我的机器上，仅使用 cpu 访问统一内存的情况下，当申请到 29GB 的时候出现： 6128 Killed。
        // 这说明，仅仅在 CPU 中访问统一内存，在使用完主机内存后，不会自动使用设备内存，
        // 因此，若要将主机内存和显存都纳入统一内存，需要及时让 GPU 访问统一内存。
        CHECK_CUDA_CALL(cudaMallocManaged(&px, mem_size));

        cpu_touch(px, data_size);

        CHECK_CUDA_CALL(cudaFree(px));
        printf("Allocated %d GB unified memory with cpu touch\n", i);
    }

    return 0;
}