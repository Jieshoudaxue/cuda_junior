#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<EOF >&2
Usage: $0 <source-file> [compiler-options...]

Compile the given source file and run the resulting executable.
The executable is removed automatically after the program exits.
EOF
  exit 1
}

if [ "$#" -lt 1 ]; then
  usage
fi

src="$1"
shift

if [ ! -f "$src" ]; then
  echo "Error: source file '$src' not found." >&2
  exit 2
fi

compiler="g++"
compiler_args=("-std=c++17" "-Wall" "-Wextra" "-O2")

case "$src" in
  *.c)
    compiler="gcc"
    compiler_args=("-std=c11" "-Wall" "-Wextra" "-O2")
    ;;
  *.cc|*.cpp|*.cxx)
    compiler="g++"
    compiler_args=("-std=c++17" "-Wall" "-Wextra" "-O2")
    ;;
  *.cu)
    compiler="nvcc"
    compiler_args=("-arch=sm_75")
    ;;
  *)
    echo "Error: unsupported source extension. Use .c, .cc, .cpp, .cxx, or .cu." >&2
    exit 3
    ;;
esac

workdir=$(mktemp -d)
exe="$workdir/run_demo_exec"

cleanup() {
  rm -rf "$workdir"
}
trap cleanup EXIT

if ! command -v "$compiler" >/dev/null 2>&1; then
  echo "Error: compiler '$compiler' not found. Please install it or add it to PATH." >&2
  exit 5
fi

# 当使用 nvcc 编译器编译 CUDA 程序时，nvcc 会将 .cu 文件分为两部分，一部分是主机端（CPU）代码，另一部分是设备端（GPU）代码。
# nvcc 会调用 gcc/g++ 编译主机代码，设备代码由 nvcc 编译器编译，最终将两部分代码链接成一个可执行程序。
# nvcc 在编译设备代码时，会将设备代码编译成 PTX（Parallel Thread Execution）汇编代码，PTX 是 NVIDIA GPU 的中间表示语言，类似于 CPU 的汇编语言。
# 然后 nvcc 再将 PTX 代码编译成二进制的 cubin 目标代码。
# 通常情况下，使用 nvcc 编译 CUDA 程序时，需要指定目标 GPU 的 CC 版本（Compute Capability），
# 例如 -arch=sm_75 表示编译针对 CC 7.5 的 GPU 架构，-arch=sm_86 表示编译针对 CC 8.6 的 GPU 架构。
# 我的 GPU NVIDIA GeForce MX450, CC 为 7.5，所以在编译时使用 -arch=sm_75 参数。
# 如果不指定 -arch 参数，nvcc 会使用默认的 CC 版本进行编译，CUDA 11.4 默认的 CC 版本为 5.3
echo "--------------------"
echo "$compiler ${compiler_args[*]} -o $exe $src $@"
echo "--------------------"

if ! "$compiler" "${compiler_args[@]}" -o "$exe" "$src" "$@"; then
  echo "Compilation failed." >&2
  exit 4
fi

"$exe"
