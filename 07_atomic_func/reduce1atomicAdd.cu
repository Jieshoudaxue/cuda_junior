#include <cstdio>
#include <cstdint>

#include "cuda_error.cuh"

// typedef float real;
typedef double real;

const int NUM_REPEATS = 100;
const int N = 1e8;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;

__global__ void reduce(const real *d_x, real *d_y, const int N) {
    const int tid = threadIdx.x;
    const int n = blockIdx.x * blockDim.x + tid;

    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    // 原子函数在一次原子事务（atomic transaction）中完成，不会被别的线程中的原子操作所干扰。
    // 原子函数不能保证各个线程的执行具有特定的次序，但是能够保证每个线程的操作一气呵成，不被其他线程干扰，所以能够保证得到正确的结果。
    // CUDA 的原子函数类似 C++ 的原子变量（atomic variable），原子变量是 C++11 标准引入的，位于头文件 <atomic> 中，使用 std::atomic<T> 定义原子变量。
    // C++ 的原子变量提供了原子操作函数，比如 std::atomic<T>::fetch_add()，它的语义与 CUDA 的 atomicAdd() 函数类似。

    // atomicAdd 是 CUDA 提供的原子操作函数（atomic function）： atomicAdd(address, val)
    // 第一个参数是待累加变量的地址，可以是全局内存，也可以是共享内存; 第二个参数是要加的值，函数返回的是原来的值，即 old 值。
    // 这个函数在语义上相当于：d_y[0] += s_y[0]
    if (tid == 0) {
        atomicAdd(d_y, s_y[0]);
    }

    // 这里我们系统梳理 CUDA 的原子函数，他们都是 __device__ 函数，只能在核函数中时间，所有的原子函数返回值都是原来的值（old value），而不是累加后的值。
    // 1 加法： T atomicAdd(T *address, T val); 功能： new = old + val; 返回值：old
    // 2 减法： T atomicSub(T *address, T val); 功能： new = old - val; 返回值：old
    // 3 交换： T atomicExch(T *address, T val); 功能： new = val; 返回值：old
    // 4 最小值： T atomicMin(T *address, T val); 功能： new = min(old, val); 返回值：old
    // 5 最大值： T atomicMax(T *address, T val); 功能： new = max(old, val); 返回值：old
    // 6 自增： T atomicInc(T *address, T val); 功能： new = (old >= val) ? 0 : old + 1; 返回值：old
    // 7 自减： T atomicDec(T *address, T val); 功能： new = (old == 0 || old > val) ? val : old - 1; 返回值：old
    // 8 按位与： T atomicAnd(T *address, T val); 功能： new = old & val; 返回值：old
    // 9 按位或： T atomicOr(T *address, T val); 功能： new = old | val; 返回值：old
    // 10 按位异或： T atomicXor(T *address, T val); 功能： new = old ^ val; 返回值：old
    // 11 比较和交换（Compare and Swap, CAS）: T atomicCAS(T *address, T compare, T val); 功能： new = (old == compare) ? val : old; 返回值：old
    
    // 在所有 CUDA 原子函数中，atomicCAS 函数是比较特殊的, 所有其他原子函数都可以用它实现。
    // C++ 中也有 CAS 的实现，利用的是 C++ atomic 的 compare_exchange_weak() 函数，他能实现比较和赋值两步操作的原子化，是很多无锁数据结构的核心，样例如下：
    // bool success = x.compare_exchange_strong(expected, desired); 解释：如果 x == expected，则把 x 设为 desired，返回 true
}

// 这个 reduce 是主机函数，与上面的核函数同名，但参数不同，这里说明 CUDA C++ 允许主机函数和核函数进行重载（overload），
// 这里的 reduce 函数相当于一个封装函数，封装了核函数的调用过程，方便主机端调用。
real reduce(const real *d_x) {
    const int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int smem_size = sizeof(real) * BLOCK_SIZE;

    // 这里的 h_y 是一个主机上的数组，长度为 1，位于主机栈空间
    real h_y[1] = {0};
    real *d_y;
    CHECK_CUDA_CALL(cudaMalloc((void **)&d_y, sizeof(real)));
    CHECK_CUDA_CALL(cudaMemcpy(d_y, h_y, sizeof(real), cudaMemcpyHostToDevice));

    reduce<<<grid_size, BLOCK_SIZE, smem_size>>>(d_x, d_y, N);

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
    for (int i = 0; i < N; i ++) {
        h_x[i] = 1.23;
    }

    real *d_x;
    CHECK_CUDA_CALL(cudaMalloc((void **)&d_x, M));
    CHECK_CUDA_CALL(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    printf("\n using atomicAdd: \n");
    timing(d_x);

    free(h_x);
    CHECK_CUDA_CALL(cudaFree(d_x));
    return 0;
}