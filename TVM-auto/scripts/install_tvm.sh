#!/usr/bin/env bash
# Build TVM (CPU-only) inside the active conda env
set -euo pipefail

TVM_HOME="$(git rev-parse --show-toplevel)/TVM-auto/third_party/tvm"
BUILD_DIR="${TVM_HOME}/build"


set -euo pipefail

# === Conda Environment Setup ===
if ! conda env list | grep -q "OPT-einsum"; then
    echo "[!] Conda environment 'OPT-einsum' not found."
    echo "    Create it with: conda env create -f TVM-auto/environment.yml"
    exit 1
fi
which python
which pip

echo "[setup_tvm] Building TVM in ${BUILD_DIR}"

# Copy default config the first time
[[ -f "${TVM_HOME}/config.cmake" ]] || \
    cp "${TVM_HOME}/cmake/config.cmake" "${TVM_HOME}"

# Enable LLVM and MetaSchedule in config
sed -i 's/set(USE_LLVM .*)/set(USE_LLVM ON)/'   "${TVM_HOME}/config.cmake"
sed -i 's/set(USE_METAL .*)/set(USE_METAL OFF)/' "${TVM_HOME}/config.cmake"
# AutoScheduler/MetaSchedule is ON by default from TVM v0.14+

mkdir -p "${BUILD_DIR}"
pushd "${BUILD_DIR}" >/dev/null

export CC=gcc-11
export CXX=g++-11

cmake .. \
  -DCMAKE_C_COMPILER=gcc-11 \
  -DCMAKE_CXX_COMPILER=g++-11 \
  -DUSE_LLVM=ON \
  -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"
popd >/dev/null

# Install Python bindings
pip install -e "${TVM_HOME}/python" -q

echo "[setup_tvm] TVM build finished"
