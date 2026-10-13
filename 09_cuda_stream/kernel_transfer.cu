#include <cmath>
#include <cstdio>

#include "cuda_error.cuh"

typedef float real;
// typedef double real;

const int NUM_REPEATS = 10;
const int N = 1 << 22;
const int M = sizeof(real) * N;
const int MAX_NUM_STREAMS = 10;

cudaStream_t streams[MAX_NUM_STREAMS];

__global__ void add(const real *x, const real *y, real *z, int N) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        z[tid] = x[tid] + y[tid];
    }
}

void timing(const real *h_x, const real *h_y, real *h_z,
            real *d_x, real *d_y, real *d_z, const int num) {
    int N1 = N / num;
    int M1 = M / num;

    float t_sum = 0;
    float t2_sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat) {
        cudaEvent_t start, stop;
        CHECK_CUDA_CALL(cudaEventCreate(&start));
        CHECK_CUDA_CALL(cudaEventCreate(&stop));
        CHECK_CUDA_CALL(cudaEventRecord(start));
        cudaEventQuery(start);

        for (int i = 0; i < num; i ++) {
            int offset = i * N1;
            CHECK_CUDA_CALL(cudaMemcpyAsync(d_x + offset, h_x + offset, M1, cudaMemcpyHostToDevice, streams[i]));
            CHECK_CUDA_CALL(cudaMemcpyAsync(d_y + offset, h_y + offset, M1, cudaMemcpyHostToDevice, streams[i]));
            
            int block_size = 128;
            int grid_size = (N1 - 1) / block_size + 1;
            add<<<grid_size, block_size, 0, streams[i]>>>(d_x + offset, d_y + offset, d_z + offset, N1);

            CHECK_CUDA_CALL(cudaMemcpyAsync(h_z + offset, d_z + offset, M1, cudaMemcpyDeviceToHost, streams[i]));
        }

        CHECK_CUDA_CALL(cudaEventRecord(stop));
        CHECK_CUDA_CALL(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK_CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms\n", elapsed_time);
        
        if (repeat > 0) {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK_CUDA_CALL(cudaEventDestroy(start));
        CHECK_CUDA_CALL(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);
}

// 这个例子我们演示第三种任务级别的并行：核函数与数据传输之间的并行
// 09_cuda_stream/add.cu 的注释里我们提到，数据传输是同步的，也叫阻塞的，即主机调用 cudaMemcpy 后，会等待命令执行完毕再往前走。
// 而核函数的启动是异步的，也叫非阻塞的，即主机调用 add 后，不会等待核函数执行完毕，可以立即做别的事情。

// 如果要实现核函数与数据传输之间的并行，数据传输必须使用 cudaMemcpy 函数的异步版本，即 cudaMemcpyAsync 函数。
// 他的底层由 GPU 的 DMA（direct memory access）直接实现，不需要主机参与，因此可以实现核函数与数据传输之间的并行。
// 除了必须使用 cudaMemcpyAsync 函数外，核函数与数据传输两个操作还必须处于不同的非默认流，才能实现并行。

// cudaMemcpyAsync 函数签名如下，他比 cudaMemcpy 多了一个 stream 参数，表示数据传输操作属于哪个流。
// cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream = 0);

// 使用 cudaMemcpyAsync 函数时，还有一个要求，即主机内存必须使用 cudaMallocHost 或 cudaHostAlloc 函数分配，而不是使用 malloc 直接分配。
// 使用 cudaMallocHost 或 cudaHostAlloc 函数分配的主机内存是不可分页内存（non-pageablememory）或者固定内存（pinned memory），其在程序运行期间，其物理地址将保持不变，
// 使用 malloc 分配的主机内存是可分页内存（pageable memory），其在程序运行期间，其物理地址有可能被操作系统改变，
// 如果给 cudaMemcpyAsync 传入可分页内存，CUDA 驱动会隐式地将其暂存到不可分页内存，从而使异步调用退化为同步行为，因此只有不可分页内存才能实现核函数与数据传输之间的并行，

// cudaMallocHost 或 cudaHostAlloc 函数签名如下，本样例我们只使用 cudaMallocHost 函数：
// cudaError_t cudaMallocHost(void** ptr, size_t size);
// cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags);
// 对应的释放函数签名如下:
// cudaError_t cudaFreeHost(void* ptr);
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

    // 在这个例子中，程序要处理 1<<22 固定长度的数组求和计算，通常情况下，处理逻辑是先拷贝到设备，然后在设备上执行核函数，最后再拷贝回主机，这三个操作是串行的，无法并行化。
    // 三个环节中，从主机到设备的数据传输是最慢的，细节参考博客：https://blog.csdn.net/cy1641395022/article/details/163591448 。
    // 本样例中，我们将数组分成 MAX_NUM_STREAMS 份，每份长度为 N1 = N / MAX_NUM_STREAMS，使用 MAX_NUM_STREAMS 个非空流来处理这 MAX_NUM_STREAMS 份数据，
    // 每个非空流内部串行处理 N1 长度的数据元素，而不同非空流之间是并行的，这样就实现了核函数与数据传输之间的并行，
    // 而实现核函数与数据传输之间的并行，可以减少数据传输的等待时间，从而加速计算。效果如下（H2D： 主机到设备，KER： 核函数，D2H： 设备到主机）：
    // Stream 1：H2D -> KER -> D2H
    // Stream 2：       H2D -> KER -> D2H
    // Stream 3：              H2D -> KER -> D2H
    // Stream 4：                      H2D -> KER -> D2H
    
    // 在我的机器上，五个以内的非空流的执行时间是逐步减少的，五个以上的非空流的执行时间几乎不变，见 09_cuda_stream/kernel_transfer_result.txt,
    // 这说明，使用多个非空流实现核函数与数据传输之间的并行，可以加速计算，让算力充分发挥。
    for (int i = 1; i <= MAX_NUM_STREAMS; i++) {
        printf("Using %d streams:\n", i);
        timing(h_x, h_y, h_z, d_x, d_y, d_z, i);
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