#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "cuda_error.cuh"

// typedef float real;
typedef double real;

// 原子个数
// 每个原子最大的邻居数量，这里设置为 3，实际应用中可能会更大
int N;
const int MAX_NUM_NEIGHBORS = 3;

// 原子间的截断半径
// 原子间的截断半径平方，如果两个原子间的距离平方小于 cutoff_square，则认为它们是邻居
const real cutoff = 1.9; 
const real cutoff_square = cutoff * cutoff;

const int NUM_REPEATS = 20;

void read_xy(std::vector<real> &xv, std::vector<real> &yv) {
    std::ifstream infile("/home/momenta/cuda_junior/07_atomic_func/xy.txt");
    std::string line, word;
    if (!infile) {
        std::cout << "Can not open xy.txt" << std::endl;
        exit(1);
    }

    while (std::getline(infile, line)) {
        std::istringstream words(line);
        if (line.length() == 0) {
            continue;
        }

        for (int i = 0; i < 2; i++) {
            if (words >> word) {
                if (i == 0) {
                    xv.push_back(std::stod(word));
                }
                if (i == 1) {
                    yv.push_back(std::stod(word));
                }
            } else {
                std::cout << "Error for reading xy.txt" << std::endl;
                exit(1);
            }
        }
    }
    infile.close();
}

void find_neighbor(int *nn_ptr, int *nl_ptr, const real *x, const real *y) {
    // 将所有原子的邻居个数初始化为零，方便后面累加
    for (int i = 0; i < N; i++) {
        nn_ptr[i] = 0;
    }

    for (int i = 0; i < N; i++) {
        real x1 = x[i];
        real y1 = y[i];
        // 如果一个原子是另一个原子的邻居，那个反之也成立，
        // 因此双层循环的内层不需要从 0 开始，只需要从 i + 1 开始，省去了一半的计算量
        for (int j = i + 1; j < N; j++) {
            real x12 = x[j] - x1;
            real y12 = y[j] - y1;
            real distance_square = x12 * x12 + y12 * y12;
            if (distance_square < cutoff_square) {
                // nl_ptr[i * MAX_NUM_NEIGHBORS + k] 是第 i 个原子的第 k 个邻居的索引
                // 这里的 nn_ptr[i] 是原子 i 的邻居数量，初始为 0，每找到一个邻居就加 1
                nl_ptr[i * MAX_NUM_NEIGHBORS + nn_ptr[i]++] = j;
                nl_ptr[j * MAX_NUM_NEIGHBORS + nn_ptr[j]++] = i;
            }
        }
    }
}

void timing(int *nn_ptr, int *nl_ptr, std::vector<real> &xv, std::vector<real> &yv) {
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat) {
        cudaEvent_t start, stop;
        CHECK_CUDA_CALL(cudaEventCreate(&start));
        CHECK_CUDA_CALL(cudaEventCreate(&stop));
        CHECK_CUDA_CALL(cudaEventRecord(start));
        cudaEventQuery(start);

        find_neighbor(nn_ptr, nl_ptr, xv.data(), yv.data());

        CHECK_CUDA_CALL(cudaEventRecord(stop));
        CHECK_CUDA_CALL(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK_CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms\n", elapsed_time);
        
        CHECK_CUDA_CALL(cudaEventDestroy(start));
        CHECK_CUDA_CALL(cudaEventDestroy(stop));
    }
}

// 打印每个原子的所有邻居原子的索引，注意中心原子本身不算邻居，不在邻居列表中，因此不打印
// 但 neighbor.txt 文件的行号减一就是中心原子的索引，请读者知悉
void print_neighbor(const int *nn_ptr, const int *nl_ptr) {
    std::ofstream outfile("/home/momenta/cuda_junior/07_atomic_func/neighbor.txt");
    if (!outfile) {
        std::cout << "Can not open neighbor.txt" << std::endl;
        exit(1);
    }
    for (int i = 0; i < N; i++) {
        if (nn_ptr[i] > MAX_NUM_NEIGHBORS) {
            std::cout << "Error: MAX_NUM_NEIGHBORS is too small" << std::endl;
            exit(1);
        }
        for (int j = 0; j < MAX_NUM_NEIGHBORS; j++) {
            if (j < nn_ptr[i]) {
                outfile << " " << nl_ptr[i * MAX_NUM_NEIGHBORS + j];
            } else {
                outfile << " NaN";
            }
        }
        outfile << std::endl;
    }
    outfile.close();
}

int main(void) {
    // xy.txt 文件中存储了每个原子的坐标，第一列是 x 坐标，第二列是 y 坐标
    // 读取 xy.txt 文件中的原子坐标，并存储在 x 和 y 向量中
    // xv 的长度就是所有原子的个数，存储在全局变量 N 中，所有原子的索引范围是 [0, N-1]
    std::vector<real> xv, yv;
    read_xy(xv, yv);
    N = xv.size();

    // neighbor_number_ptr 用来存储每个原子的邻居数量，
    // neighbor_list_ptr 用来存储每个原子的邻居列表，列表元素是邻居原子的索引
    int *neighbor_number_ptr = (int *)malloc(N * sizeof(int));
    int *neighbor_list_ptr = (int *)malloc(N * MAX_NUM_NEIGHBORS * sizeof(int));

    timing(neighbor_number_ptr, neighbor_list_ptr, xv, yv);
    print_neighbor(neighbor_number_ptr, neighbor_list_ptr);

    free(neighbor_number_ptr);
    free(neighbor_list_ptr);
    return 0;
}
