# include <cstdio>

// __device__ 是一个 CUDA 的关键字，他表明这个函数是一个设备函数（device function），
// 由设备端（GPU）执行，由设备端（GPU）调用。
__device__ void hello_from_gpu_inner() {
    printf("hello world from the gpu inner function!\n");
}

// __global__ 是一个 CUDA 的关键字，他表明这个函数是一个 CUDA 核函数（kernel function），
// 由设备端（GPU）执行，由主机端（CPU）调用。
__global__ void hello_from_gpu() {
    hello_from_gpu_inner();
}

// __host__ 是一个 CUDA 的关键字，他表明这个函数是一个主机函数（host function），
// 由主机端（CPU）执行，由主机端（CPU）调用。
// 通常情况下，主机函数不需要显式地声明 __host__
__host__ void hello_from_cpu() {
    printf("hello world from the cpu!\n");
}

// gpu 程序必须从主机（cpu）程序中启动，因此程序的入口也是 main() 函数
int main(void) {
    hello_from_cpu();

    // gpu 有成千上万个处理核，对应到 CUDA 编程中就是成千上万个线程（thread），
    // gpu 的线程非常轻量，线程切换 ～1 cycle，而 cpu 的线程切换需要 ～1000 cycle.
    // 调用 CUDA 核函数时，必须指定线程的数量和线程的组织方式，这就是 <<<...>>> 中的参数。
    // 线程先组织成线程块，所有线程块再构成一个网格（grid），网格大小（grid size）就是线程块的个数，线程块大小（block size）就是线程块中线程的个数。
    // 因此，<<<>>> 中的参数就是 <<<grid size, block size>>>,即<<<线程块的数量, 每个线程块中线程的数量>>>。
    // 这里的 <<<1, 1>>> 表示启动一个线程块，线程块中有一个线程。
    hello_from_gpu<<<1, 1>>>();

    // cudaDeviceSynchronize() 是一个 CUDA 的 API 函数，作用是同步主机端（CPU）和设备端（GPU），
    // 没有这个函数，就打印不出来了
    cudaDeviceSynchronize();
    return 0;
}