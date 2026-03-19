#!/usr/bin/env bash
# =============================================================================
# build_cross.sh  –  Cross-compile cuda-thor-demo for NVIDIA Thor (aarch64)
#                    Run this on an x86_64 Linux host.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configurable paths – override with env vars or edit below
# ---------------------------------------------------------------------------
# Full sysroot of the Thor target.
# Options:
#   1. NVIDIA SDK Manager sysroot (e.g. ~/nvidia/nvidia_sdk/DriveOS_6.0.x_Linux/DRIVEOS/drive-linux/filesystem/targetfs)
#   2. Jetson cross-tool bare sysroot: /usr/aarch64-linux-gnu
# THOR_SYSROOT="${THOR_SYSROOT:-/usr/aarch64-linux-gnu}"
THOR_SYSROOT="${THOR_SYSROOT:-/data/var/lib/docker/overlay2/393ff708f3e9eb0e03800d1e8c6ffe155ab7688c0b7a2975c43190de87e16262/diff/usr/aarch64-linux-gnu}"  # --- IGNORE ---

# CUDA Toolkit with aarch64 target stubs installed.
# SDK Manager installs to: /usr/local/cuda-<ver>
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"

BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${BUILD_DIR:-build-aarch64}"
INSTALL_DIR="${INSTALL_DIR:-install-aarch64}"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
echo "──────────────────────────────────────────────"
echo " NVIDIA Thor cross-compile build"
echo "──────────────────────────────────────────────"
echo " THOR_SYSROOT : ${THOR_SYSROOT}"
echo " CUDA_PATH    : ${CUDA_PATH}"
echo " BUILD_TYPE   : ${BUILD_TYPE}"
echo " BUILD_DIR    : ${BUILD_DIR}"
echo "──────────────────────────────────────────────"

if ! command -v aarch64-linux-gnu-gcc &>/dev/null; then
    echo "[ERROR] aarch64-linux-gnu-gcc not found."
    echo "  Install with: sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu"
    exit 1
fi

if [ ! -f "${CUDA_PATH}/bin/nvcc" ]; then
    echo "[ERROR] nvcc not found at ${CUDA_PATH}/bin/nvcc"
    echo "  Install CUDA Toolkit and set CUDA_PATH."
    exit 1
fi

if [ ! -d "${CUDA_PATH}/targets/aarch64-linux" ]; then
    echo "[WARN] ${CUDA_PATH}/targets/aarch64-linux not found."
    echo "  Make sure cuda-cross-aarch64 package is installed:"
    echo "  sudo apt install cuda-cross-aarch64-<version>"
fi

# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------
    # -G Ninja \
CUDA_STUBS=${CUDA_PATH}/targets/aarch64-linux/lib/stubs

mkdir -p "${BUILD_DIR}"

cmake -S . -B "${BUILD_DIR}" \
    -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_TOOLCHAIN_FILE="./toolchain-aarch64-thor_rev01.cmake" \
    -DCUDA_CUDA_LIBRARY="${CUDA_STUBS}/libcuda.so" \
    -DCUDA_NVTX_LIBRARY="${CUDA_STUBS}/libnvidia-ml.so" \
    -DCUDA_CURAND_LIBRARY="${CUDA_STUBS}/libcurand.so" \
    -DTHOR_SYSROOT="${THOR_SYSROOT}" \
    -DCUDA_PATH="${CUDA_PATH}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
cmake --build "${BUILD_DIR}" --parallel "$(nproc)"

# ---------------------------------------------------------------------------
# Optional install
# ---------------------------------------------------------------------------
cmake --install "${BUILD_DIR}"

echo ""
echo "✓ Build complete. Binaries:"
find "${BUILD_DIR}" -maxdepth 2 -type f -executable | grep -v '\.cmake\|Makefile'

# echo ""
# echo "Deploy to Thor target:"
# echo "  scp ${INSTALL_DIR}/bin/* <user>@<thor-ip>:/usr/local/bin/"
