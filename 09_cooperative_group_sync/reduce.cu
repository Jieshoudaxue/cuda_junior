#include <cstdio>
#include <cstdint>
#include <cooperative_groups.h>

#include "cuda_error.cuh"

// typedef float real;
typedef double real;

const int NUM_REPEATS = 20;
const int N = 1e8;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;

__global__ void reduce_cooperative_group(const real *d_x, real *d_y) {
    const int tid = threadIdx.x;
    const int n = blockIdx.x * blockDim.x + tid;

    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    real y = s_y[tid];

    // 前面的样例，我们学习了线程块内的同步机制，和线程束内的同步机制，以及线程束内的表决和洗牌函数，他们自带同步机制
    // CUDA 提供了一种更加灵活的线程协作方式，即协作组（cooperative groups），目前协作组主要作用于线程块内的同步，
    // 协作组所有的元素都位于 cooperative_groups 命名空间内

    // 最简单的协作组就是一个线程块，定义方式： cooperative_groups::thread_block cg = cooperative_groups::this_thread_block()
    // 此时定义的 cg 就是包装成 thread_block 类型的线程块，有如下方法：
    // void sync();                 协作组内的线程同步函数，与 __syncthreads() 等效
    // unsigned int size();         返回组的大小
    // unsigned int thread_rank();  返回当前线程在组内的标号
    // bool is_valid();             返回定义的组是否有效
    // dim3 group_index();          返回当前线程块的 blockIdx
    // dim3 thread_index();         返回当前线程的 threadIdx

    // 由线程块定义的协作组，可以使用模板函数 tiled_partition 将协作组再分为线程块片（thread block tile）,
    // 注意线程块片的大小必须是 2, 4, 8, 16, 32，如果设为 32, 那么这个线程块片就是线程束，正如下面的例子。
    // 线程块片提供了与线程束基本函数等效的一组表决和洗牌函数：
    // unsigned int ballot(int predicate);
    // int all(int predicate);
    // int any(int predicate);
    // T shfl(T var, int srcLane);
    // T shfl_up(T var, unsigned int delta);
    // T shfl_down(T var, unsigned int delta);
    // T shfl_xor(T var, int laneMask);
    cooperative_groups::thread_block_tile<32> cg = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    // cg.size() 就是 32
    for (int i = cg.size() >> 1; i > 0; i >>= 1) {
        // cf.shfl_down 与 __shfl_down_sync 是等效的
        y += cg.shfl_down(y, i);
    }

    if (tid == 0) {
        atomicAdd(d_y, y);
    }
}

real reduce(const real *d_x) {
    const int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int shared_mem_size = BLOCK_SIZE * sizeof(real);

    real h_y[1] = {0};
    real *d_y;
    CHECK_CUDA_CALL(cudaMalloc((void **)&d_y, sizeof(real)));
    CHECK_CUDA_CALL(cudaMemcpy(d_y, h_y, sizeof(real), cudaMemcpyHostToDevice));

    reduce_cooperative_group<<<grid_size, BLOCK_SIZE, shared_mem_size>>>(d_x, d_y);

    CHECK_CUDA_CALL(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaFree(d_y));
    return h_y[0];
}

void timing(real *d_x) {
    real sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat) {
        cudaEvent_t start, stop;
        CHECK_CUDA_CALL(cudaEventCreate(&start));
        CHECK_CUDA_CALL(cudaEventCreate(&stop));
        CHECK_CUDA_CALL(cudaEventRecord(start));
        cudaEventQuery(start);
        
        sum = reduce(d_x);

        CHECK_CUDA_CALL(cudaEventRecord(stop));
        CHECK_CUDA_CALL(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK_CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms\n", elapsed_time);
        
        CHECK_CUDA_CALL(cudaEventDestroy(start));
        CHECK_CUDA_CALL(cudaEventDestroy(stop));
    }

    printf("sum = %f\n", sum);
}

int main(void) {
    real *h_x = (real *)malloc(M);
    for (int i = 0; i < N; i++) {
        h_x[i] = 1.23;
    }

    real *d_x;
    CHECK_CUDA_CALL(cudaMalloc((void **)&d_x, M));
    CHECK_CUDA_CALL(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    printf("\n using cooperative group: \n");
    timing(d_x);

    CHECK_CUDA_CALL(cudaFree(d_x));
    free(h_x);
    return 0;
}