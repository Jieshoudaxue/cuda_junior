#include <cstdio>
#include <cstdint>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include "cuda_error.cuh"

// typedef float real;
typedef double real;

const int NUM_REPEATS = 10;
const int N = 1e8;

real reduce_cpu_loop(const std::vector<real> &x) {
    real sum = 0.0;
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        sum += x[i];
    }
    return sum;
}

real reduce_gpu(const thrust::device_vector<real> &x) {
    return thrust::reduce(thrust::device, x.begin(), x.end(), real(0.0), thrust::plus<real>());
}

void timing(const std::vector<real> &std_x, const thrust::device_vector<real> &d_x, const int method) {
    real sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat) {
        cudaEvent_t start, stop;
        CHECK_CUDA_CALL(cudaEventCreate(&start));
        CHECK_CUDA_CALL(cudaEventCreate(&stop));
        CHECK_CUDA_CALL(cudaEventRecord(start));
        cudaEventQuery(start);

        if (method == 0) {
            sum = reduce_cpu_loop(std_x);
        } else if (method == 1) {
            sum = reduce_gpu(d_x);
        } else {
            printf("Invalid method: %d\n", method);
            sum = 0;
        }

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

// 前几章我们详细介绍了数组归约的实现，但实际工作中，我们使用 thrust 就能很轻易的完成数组归约操作
// 本样例，我们使用 thrust 在 GPU 完成数组归约。在我的机器上，这个样例的耗时与最后一版 09_cooperative_group_sync/reduce1parallelism.cu 的耗时是一样的，都是 10ms 左右
// 工作中尽可能用库，但学习中必须学习里面的实现原理，学原理很慢，学库很快，但不懂原理，只学库，就是一知半解，无法精进
int main(void) {
    std::vector<real> std_x(N);
    for (int i = 0; i < N; ++i) {
        std_x[i] = 1.23;
    }

    thrust::device_vector<real> d_x = std_x;

    printf("\n using std::vector + loop reduce: \n");
    timing(std_x, d_x, 0);
    printf("\n using thrust::device_vector + thrust::reduce(thrust::device): \n");
    timing(std_x, d_x, 1);

    return 0;
}