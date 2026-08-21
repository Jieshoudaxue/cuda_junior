#include <cstdio>

#include "cuda_error.cuh"

// typedef float real;
typedef double real;

const int NUM_REPEATS = 20;
const int N = 1e8;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;

// 这个归约算法称之为折半归约（binary reduction），
// 他要求数组长度 N 必须被 BLOCK_SIZE 整除，且 BLOCK_SIZE 为 2 的整数次方，比如这里使用的 BLOCK_SIZE 为 128。
// 具体做法是将数组分为 BLOCK_SIZE 部分，每部分交由一个线程块处理，
// 对于每个 BLOCK_SIZE 子数组，将后半部分的各个元素与前半部分对应的数组元素相加，重复此过程，最后得到的第一个数组元素就是最初的数组中各个元素的和。
// 这个过程要求对同一个线程块的线程进行同步，避免数据竞争。
// 由于不同线程块负责不同的子数组，因此不需要进行线程同步。
__global__ void reduce_global(real *d_x, real *d_y) {
    const int tid = threadIdx.x;
    // 这句等价于：real *x = &d_x[blockDim.x * blockIdx.x];
    // 他的作用是从整个数组中，取出 BLOCK_SIZE 个元素的首地址，交由一个线程块处理
    real *x = d_x + blockDim.x * blockIdx.x;

    // blockDim.x >> 1 等价于 blockDim.x / 2; offset >>= 1 等价于 offset /= 2; 
    // 使用位运算，是因为效率比较高。
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        // 这个判断确保随着归约的进行，越来越多的线程空闲下来
        if (tid < offset) {
            x[tid] += x[tid + offset];
        }
        // 用于同步单个线程块里的线程，只能用于核函数（也包括被核函数调用的设备函数）
        // 他的作用是保证一个线程块中的所有线程，在执行该语句后面的语句之前，完全执行了该语句前面的语句,
        // 针对这个例子，该函数确保每次归约涉及的读和写都能顺利走完，不会被线程块的其他线程干扰，避免数据竞争（data race）。
        __syncthreads();
    }

    // 这个函数归约的结果，是将 N 长的数组，变为 N/BLOCK_SIZE 长的数组，而不是直接归约为 1，请注意这一点
    // 后续的逻辑会将 d_y 拷贝到主机端，使用循环完成最后的归约。
    if (tid == 0) {
        d_y[blockIdx.x] = x[0];
    }
}

__global__ void reduce_shared(real *d_x, real *d_y) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int n = bid * blockDim.x + tid;
    __shared__ real s_y[BLOCK_SIZE];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_y[bid] = s_y[0];
    }
}

__global__ void reduce_dynamic(real *d_x, real *d_y) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_y[bid] = s_y[0];
    }
}

real reduce(real *d_x, const int method) {
    // 等价于： (N % BLOCK_SIZE == 0) ? (N / BLOCK_SIZE) : (N / BLOCK_SIZE + 1);
    // 也等价于：(N - 1) / BLOCK_SIZE + 1;
    int grid_size = (N + BLOCK_SIZE -1) / BLOCK_SIZE;
    const int ymem = sizeof(real) * grid_size;
    const int smem = sizeof(real) * BLOCK_SIZE;
    real *d_y;
    CHECK_CUDA_CALL(cudaMalloc(&d_y, ymem));
    real *h_y = (real *)malloc(ymem);

    switch(method) {
        case 0:
            reduce_global<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
            break;
        case 1:
            reduce_shared<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
            break;
        case 2:
            reduce_dynamic<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y);
            break;
        default:
            printf("Error: wrong method\n");
            exit(1);
            break;
    }

    CHECK_CUDA_CALL(cudaMemcpy(h_y, d_y, ymem, cudaMemcpyDeviceToHost));

    // 在主机端完成最后的归约
    real result = 0.0;
    for (int i = 0; i < grid_size; ++i) {
        result += h_y[i];
    }

    free(h_y);
    CHECK_CUDA_CALL(cudaFree(d_y));
    return result;
}

void timing(real *h_x, real *d_x, const int method) {
    real sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat) {
        CHECK_CUDA_CALL(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CHECK_CUDA_CALL(cudaEventCreate(&start));
        CHECK_CUDA_CALL(cudaEventCreate(&stop));
        CHECK_CUDA_CALL(cudaEventRecord(start));
        cudaEventQuery(start);
        
        sum = reduce(d_x, method);

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


// 数组归约：把数组中的所有元素，通过某个操作（如加法、求最大值），合并成一个值
// CPU 归约：循环遍历
// GPU 归约：把数组分成很多组，每组独立计算，最后再合并结果。每一轮线程数量减半，像树一样从叶子到根计算，时间复杂度从 O(N) 降到 O(log₂N)
// 归约是 CUDA 编程的 Hello World，会同时用到共享内存，CUDA 线程同步，树形归约算法。理解了归约，就基本掌握了 CUDA 加速的核心思想，面试必考。

// 共享内存是一种可被程序员直接操控的缓存，主要作用有两个：
// 一个是减少核函数中对全局内存的访问次数，实现高效的线程块内部的通信，另一个是提高全局内存访问的合并度。
int main(void) {
    real *h_x = (real *)malloc(M);
    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.23;
    }

    real *d_x;
    CHECK_CUDA_CALL(cudaMalloc(&d_x, M));

    printf("Using global memory only: \n");
    timing(h_x, d_x, 0);

    printf("\nUsing static shared memory: \n");
    timing(h_x, d_x, 1);

    printf("\nUsing dynamic shared memory: \n");
    timing(h_x, d_x, 2);

    free(h_x);
    CHECK_CUDA_CALL(cudaFree(d_x));

    return 0;
}