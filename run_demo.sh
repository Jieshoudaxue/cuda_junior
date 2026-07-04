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
    compiler_args=("-O2")
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

if ! "$compiler" "${compiler_args[@]}" -o "$exe" "$src" "$@"; then
  echo "Compilation failed." >&2
  exit 4
fi

"$exe"
