#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <cstdio>

// thrust 是一个实现了众多基本并行算法的 C++ 模板库，命名空间是 thrust::，类似 STL，包含在 CUDA 工具箱中。
// 由于是模板库，thrust 仅仅由头文件组成，使用时无需链接任何库。

// thrust 的数据结构主要是 vector, 分为两种：
// 1. thrust::host_vector<T>，在主机上分配内存，类似 STL 的 vector。头文件为 #include <thrust/host_vector.h>。
// 2. thrust::device_vector<T>，在设备上分配内存，头文件为 #include <thrust/device_vector.h>。

// thrust 提供了 5 类常用算法：详情见 https://github.com/thrust/thrust/wiki
// 1. 变换（transformation）
// 2. 归约（reduction）, 前面重点讲解了归约算法实现
// 3. 前缀和（prefix sum，也叫 scan）
// 4. 排序（sorting）和搜索（searching）
// 5. 重排（reordering）：选择性复制、替换、移除、分区
int main(void) {
    int N = 10;
    thrust::device_vector<int> x(N, 0);
    thrust::device_vector<int> y(N, 0);
    for (int i = 0; i < x.size(); i++) {
        x[i] = i + 1;
        printf("%d  ", (int)x[i]);
    }
    printf("\n");

    // thrust::inclusive_scan() 函数用于计算前缀和，也叫扫描，
    // 扫描操作将一个序列： x0, x1, x2, ...
    // 转换为一个新的序列： y0 = x0, y1 = x0 + x1, y2 = x0 + x1 + x2, ...
    // 这种就是包含扫描（inclusive scan），扫描结果存在了 y 中。
    // 提示： hrust::inclusive_scan() 还有一种重载形式，可以直接操作指针，参考：11_cuda_stdlib/thrust_scan_pointer.cu
    thrust::inclusive_scan(x.begin(), x.end(), y.begin());
    for (int i = 0; i < y.size(); ++i) {
        printf("%d  ", (int)y[i]);
    }
    printf("\n");

    return 0;
}