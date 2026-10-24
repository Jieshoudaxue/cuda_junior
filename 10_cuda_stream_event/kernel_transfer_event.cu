#include <cmath>
#include <cstdio>

#include "cuda_error.cuh"

typedef float real;
// typedef double real;

const int N = 1 << 22;
const int M = sizeof(real) * N;
const int NUM_STREAMS = 8;

cudaStream_t streams[NUM_STREAMS];
// 定义两个事件数组，分别用于记录 kernel 完成事件和 D2H 拷贝完成事件
cudaEvent_t kernel_done[NUM_STREAMS];
cudaEvent_t copy_done[NUM_STREAMS];

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

// 这个例子我们演示流同步第二种之跨 stream 依赖： 
// 使用 Cuda Event（cudaEventRecord + cudaStreamWaitEvent），可以让一个 Stream 等待另一个 Stream 中的某个操作完成，而不必阻塞整个主机线程。
// CUDA Event 是一种将标记插入 CUDA 流的机制，可用于跟踪流中任务的进度。
// cudaEventRecord 将事件插入流中，表示当流中的所有操作完成后，事件就会被触发。
// cudaStreamWaitEvent 让一个流等待另一个流中的某个事件完成后才会执行，表示跨 stream 依赖。
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

    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA_CALL(cudaStreamCreate(&streams[i]));
        // 创建事件
        // cudaError_t cudaEventCreate(cudaEvent_t* event)
        CHECK_CUDA_CALL(cudaEventCreate(&kernel_done[i]));
        CHECK_CUDA_CALL(cudaEventCreate(&copy_done[i]));
    }

    printf("Using %d streams:\n", NUM_STREAMS);

    // 我们将数组分成 NUM_STREAMS 份，每份 N1 个元素，每份 M1 字节, 每个流负责处理自己这一份的 H2D、KER 操作，以及上一份数据的 D2H 拷贝操作，
    // 于是当前流的 D2H 拷贝操作就依赖于上一个流的 H2D、KER 操作，这样就形成了一个环形依赖关系。
    // 为了实现跨 stream 依赖，我们创建两个事件，kernel_done[i] 和 copy_done[i] 事件，具体解释如下：
    int N1 = N / NUM_STREAMS;
    int M1 = M / NUM_STREAMS;
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * N1;
        CHECK_CUDA_CALL(cudaMemcpyAsync(d_x + offset, h_x + offset, M1, cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA_CALL(cudaMemcpyAsync(d_y + offset, h_y + offset, M1, cudaMemcpyHostToDevice, streams[i]));

        int block_size = 128;
        int grid_size = (N1 - 1) / block_size + 1;
        add<<<grid_size, block_size, 0, streams[i]>>>(d_x + offset, d_y + offset, d_z + offset, N1);

        // 将事件插入流中，即将 kernel_done[i] 事件插入 streams[i] 流中，表示当 streams[i] 中的 H2D 和 KER 操作完成后，kernel_done[i] 事件就会被触发。
        // cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
        CHECK_CUDA_CALL(cudaEventRecord(kernel_done[i], streams[i]));

        // streams[i] 负责 H2D 和 KER 操作，下一个流 streams[(i + 1) % NUM_STREAMS] 负责 D2H 拷贝操作，整体形成一个环形依赖关系。
        int copy_stream_index = (i + 1) % NUM_STREAMS;
        // 调用 cudaStreamWaitEvent，让 streams[copy_stream_index] 等待 kernel_done[i] 事件完成后才会执行 D2H 拷贝操作。
        // 注意： cudaStreamWaitEvent 阻塞的是流内的所有操作，而非主机线程。
        // cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
        // flags 参数目前必须为 0，表示标准的等待行为，即让指定流等待事件完成。
        CHECK_CUDA_CALL(cudaStreamWaitEvent(streams[copy_stream_index], kernel_done[i], 0));
        // 在调用 cudaStreamWaitEvent 后，streams[copy_stream_index] 中的所有操作都会等待 kernel_done[i] 事件完成后才会执行，
        // 这里的所有操作包括下面的 D2H 拷贝操作，也包括循环进行到下一轮时， streams[copy_stream_index] 本身的 H2D 和 KER 操作。
        CHECK_CUDA_CALL(cudaMemcpyAsync(h_z + offset, d_z + offset, M1, cudaMemcpyDeviceToHost, streams[copy_stream_index]));
        // 为 streams[copy_stream_index] 插入一个事件，表示当 streams[copy_stream_index] 中的所有操作完成后，copy_done[i] 事件就会被触发。
        // 主机将会等待所有 copy_done[i] 事件完成后才会进行结果检查。
        CHECK_CUDA_CALL(cudaEventRecord(copy_done[i], streams[copy_stream_index]));
    }

    // 由于主机最后要进行结果检查，所以需要等待所有 D2H 拷贝完成，因此我们在主机线程中使用 cudaEventSynchronize 等待所有 copy_done[i] 事件完成。
    // cudaError_t cudaEventSynchronize(cudaEvent_t event);
    // 提示：这里使用的是 cudaEventSynchronize，而不是 cudaStreamSynchronize，前者粒度更细，用的更多
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA_CALL(cudaEventSynchronize(copy_done[i]));
    }
    check(h_z, N);

    for (int i = 0; i < NUM_STREAMS; i ++) {
        CHECK_CUDA_CALL(cudaStreamDestroy(streams[i]));
        // 销毁事件
        // cudaError_t cudaEventDestroy(cudaEvent_t event);
        CHECK_CUDA_CALL(cudaEventDestroy(kernel_done[i]));
        CHECK_CUDA_CALL(cudaEventDestroy(copy_done[i]));
    }

    CHECK_CUDA_CALL(cudaFreeHost(h_x));
    CHECK_CUDA_CALL(cudaFreeHost(h_y));
    CHECK_CUDA_CALL(cudaFreeHost(h_z));
    CHECK_CUDA_CALL(cudaFree(d_x));
    CHECK_CUDA_CALL(cudaFree(d_y));
    CHECK_CUDA_CALL(cudaFree(d_z));
    return 0;
}