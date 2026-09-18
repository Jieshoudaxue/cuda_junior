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
const int GRID_SIZE = 10240;

__global__ void reduce_cooperative_group(const real *d_x, real *d_y, const int N) {
    const int tid = threadIdx.x;

    // 由于总的线程数量小于数组长度，因此单个线程需要处理多个元素，又由于全局内存的合并访问限制（详情见 05_cuda_memory/memory6global.cu），
    // 单个线程不能处理连续的数组元素，而连续的线程必须处理连续的数组元素，因此合理的方案就是设置跨步，即 stride，这里将 stride 设置为总的线程数。
    // 每个线程将自己处理的数组元素累加到寄存器 y 中，然后再将寄存器 y 拷贝到当前线程 id 对应的共享内存中，准备进行归约计算。
    real y = 0.0;
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += stride) {
        y += d_x[i];
    }
    // 共享内存属于线程块（详见 05_cuda_memory/memory4shared.cu），每个线程块有自己的共享内存副本，
    // 因此，这里的 s_y 不是长度为总线程数（GRID_SIZE × BLOCK_SIZE）的长数组，而是 GRID_SIZE 个独立的长度为 BLOCK_SIZE 的短数组。
    // 这一步做完，相当于完成了 08_thread_wrap_sync/reduce.cu 中的： s_y[tid] = (n < N) ? d_x[n] : 0.0; 
    // 数据从全局内存，稍加处理后，放入了共享内存中。
    extern __shared__ real s_y[];
    s_y[tid] = y;
    __syncthreads();

    // blockDim.x 就是 BLOCK_SIZE，当 offset 大于等于 32 时，下面的执行语句需要在线程块内部进行同步，因此使用 __syncthreads()
    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    // 对于最后要处理的 32 个数据，他们由一个线程束负责，每一个线程将自己要处理的数组元素拷贝到寄存器中，然后进行累加处理，比直接操作共享内存效率要高
    y = s_y[tid];
    // 使用协作组，在线程束内部完成最后 32 个数据的计算
    // 下面的写法，可以理解为将 32 长的数组，连续向左平移 16, 8, 4, 2, 1， 每次平移都与当前值累加，最终零号线程的 y 值就是归约的最终结果。
    cooperative_groups::thread_block_tile<32> cg = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    for (int i = cg.size() >> 1; i > 0; i >>= 1) {
        y += cg.shfl_down(y, i);
    }

    if (tid == 0) {
        d_y[blockIdx.x] = y;
    }
}

real reduce(const real *d_x) {
    const int y_mem_size = sizeof(real) * GRID_SIZE;
    const int shared_mem_size = sizeof(real) * BLOCK_SIZE;

    real h_y[1] = {0};
    real *d_y;
    CHECK_CUDA_CALL(cudaMalloc(&d_y, y_mem_size));

    // 这里两次调用同一个核函数实现归约计算，
    // 第一次调用， GRID_SIZE 是 10240, BLOCK_SIZE 是 128，总线程数小于 N = 1e8，得到 d_y 数组，他的长度是 GRID_SIZE = 10240
    // 第二次调用，grid_size 是 1, block_size 取最大值 1024, 总线程数小于 d_y 长度 10240，得到新的 d_y 数组，第 0 个元素即为最终结果
    reduce_cooperative_group<<<GRID_SIZE, BLOCK_SIZE, shared_mem_size>>>(d_x, d_y, N);
    reduce_cooperative_group<<<1, 1024, sizeof(real) * 1024>>>(d_y, d_y, GRID_SIZE);

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

// 进一步优化归约计算
// 前面所有的归约实现，grid_size ，block_size 和 数组长度 N 的关系都使用如下计算方法，可以直观理解为，整个网格的线程数等于数组长度。
// const int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
// 在折半计算过程中，空闲线程数会越来越多，最后一步只有一个线程在工作，其他 N -1 个线程都是空闲的。过低的线程利用率，导致性能浪费，整体耗时也是增加的。
// 要提高计算过程的线程利用率，核心是不使用那么多线程，即总线程数比数组长度小一些，让每个线程多处理一些数据
int main(void) {
    real *h_x = (real *)malloc(M);
    for (int i = 0; i < N; i++) {
        h_x[i] = 1.23;
    }

    real *d_x;
    CHECK_CUDA_CALL(cudaMalloc(&d_x, M));
    CHECK_CUDA_CALL(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    timing(d_x);

    free(h_x);
    CHECK_CUDA_CALL(cudaFree(d_x));
    return 0;
}