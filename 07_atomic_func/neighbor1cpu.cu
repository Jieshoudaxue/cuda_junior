#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "cuda_error.cuh"

// typedef float real;
typedef double real;

int N; // number of atoms
const int NUM_REPEATS = 20;
const int MAX_NUM_NEIGHBORS = 10; // maximum number of neighbors for each atom
const real cutoff = 1.9; // in units of angtrom
const real cutoff_square = cutoff * cutoff;

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
    for (int i = 0; i < N; i++) {
        nn_ptr[i] = 0;
    }

    for (int i = 0; i < N; i++) {
        real x1 = x[i];
        real y1 = y[i];
        for (int j = i + 1; j < N; j++) {
            real x12 = x[j] - x1;
            real y12 = y[j] - y1;
            real distance_square = x12 * x12 + y12 * y12;
            if (distance_square < cutoff_square) {
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
        outfile << nn_ptr[i];
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
    std::vector<real> x, y;
    read_xy(x, y);
    N = x.size();
    int *neighbor_number_ptr = (int *)malloc(N * sizeof(int));
    int *neighbor_list_ptr = (int *)malloc(N * MAX_NUM_NEIGHBORS * sizeof(int));

    timing(neighbor_number_ptr, neighbor_list_ptr, x, y);
    print_neighbor(neighbor_number_ptr, neighbor_list_ptr);

    free(neighbor_number_ptr);
    free(neighbor_list_ptr);
    return 0;
}
