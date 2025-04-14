#!/bin/bash

# Check if CUDA is properly set up and rebuild if needed
echo "Verifying CUDA support for llama.cpp..."

# Create a permanent environment setting for LD_LIBRARY_PATH
echo "Setting up permanent CUDA library paths"
ENV_FILE="/etc/environment"

# Check if we can write to /etc
if [ -w "$ENV_FILE" ]; then
    # Set up persistent environment variable for library paths
    grep -q "CUDA_LIBRARY_PATH" $ENV_FILE || echo "CUDA_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib" >> $ENV_FILE
    grep -q "LD_LIBRARY_PATH" $ENV_FILE || echo "LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:\$LD_LIBRARY_PATH" >> $ENV_FILE
fi

# Create a system-wide library path config
echo "Creating system-wide library configuration for CUDA"
if [ -d "/etc/ld.so.conf.d" ]; then
    echo "/usr/local/lib" > /etc/ld.so.conf.d/cuda-local.conf
    echo "/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib" >> /etc/ld.so.conf.d/cuda-local.conf
    echo "/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib" >> /etc/ld.so.conf.d/cuda-local.conf
    ldconfig
fi

# Set for current session
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH set to $LD_LIBRARY_PATH"

# Create symbolic links in standard locations
echo "Creating symbolic links for CUDA libraries"
mkdir -p /usr/local/cuda/lib64

# Create symlinks for all relevant CUDA libraries
for lib in /usr/local/lib/python3.12/site-packages/nvidia/cublas/lib/libcublas*.so* /usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart*.so*; do
    if [ -f "$lib" ]; then
        ln -sf "$lib" /usr/local/lib/$(basename "$lib") 2>/dev/null || true
        ln -sf "$lib" /usr/local/cuda/lib64/$(basename "$lib") 2>/dev/null || true
    fi
done

# Check if the llama-server binary exists
if [ ! -f "/app/llama.cpp/build/bin/llama-server" ]; then
    echo "llama-server binary not found, rebuilding llama.cpp with CUDA support..."
    /app/docker/app/rebuild_llama_cuda.sh
    echo "Rebuild complete."
    exit 0
fi

# First, check if nvidia-smi is available (meaning we have GPU access)
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA drivers not detected - no GPU available. Skipping CUDA check."
    exit 0
fi

# Check for missing CUDA runtime libraries
echo "Checking for required CUDA libraries..."
CUDA_LIBS=$(ldd /app/llama.cpp/build/bin/llama-server 2>&1 | grep "not found")
if [ -n "$CUDA_LIBS" ]; then
    echo "Found missing CUDA libraries:"
    echo "$CUDA_LIBS"
    
    # Install missing CUDA runtime libraries
    echo "Installing missing CUDA libraries..."
    apt-get update
    apt-get install -y --no-install-recommends \
        cuda-cudart-12-* \
        cuda-cudart-dev-12-* \
        libcublas-12-* \
        libcublas-dev-12-* \
        libcublaslt-12-* || \
    apt-get install -y --no-install-recommends \
        cuda-cudart-11-* \
        cuda-cudart-dev-11-* \
        libcublas-11-* \
        libcublas-dev-11-* \
        libcublaslt-11-*
    
    echo "Creating symbolic links for any missing libraries..."
    # Try to find the actual library files
    find /usr -name "libcublas.so*" | while read lib; do
        ln -sf "$lib" /usr/local/lib/$(basename "$lib") 2>/dev/null || true
        ln -sf "$lib" /usr/local/cuda/lib64/$(basename "$lib") 2>/dev/null || true
    done
    find /usr -name "libcudart.so*" | while read lib; do
        ln -sf "$lib" /usr/local/lib/$(basename "$lib") 2>/dev/null || true
        ln -sf "$lib" /usr/local/cuda/lib64/$(basename "$lib") 2>/dev/null || true
    done
    find /usr -name "libcublasLt.so*" | while read lib; do
        ln -sf "$lib" /usr/local/lib/$(basename "$lib") 2>/dev/null || true
        ln -sf "$lib" /usr/local/cuda/lib64/$(basename "$lib") 2>/dev/null || true
    done
    
    # Update the library path
    ldconfig
fi

# Actually test loading a small model with GPU layers to verify CUDA support
echo "Testing actual GPU capability with llama-server..."
# Set library path to include common CUDA locations
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
TEST_OUTPUT=$(/app/llama.cpp/build/bin/llama-server --n-gpu-layers 1 --help 2>&1)

# Check for error messages indicating no CUDA support
if echo "$TEST_OUTPUT" | grep -q "llama.cpp was compiled without support for GPU\|no usable GPU found\|--gpu-layers option will be ignored"; then
    echo "CUDA support test failed: llama.cpp was compiled without GPU support"
    echo "Rebuilding llama.cpp with CUDA support..."
    /app/docker/app/rebuild_llama_cuda.sh
    echo "Rebuild complete."
else
    # Perform a more thorough check by checking if CUDA libraries are loaded
    LOADED_LIBS=$(ldd /app/llama.cpp/build/bin/llama-server 2>/dev/null | grep -i "cuda\|nvidia")
    
    if [ -z "$LOADED_LIBS" ]; then
        echo "WARNING: No CUDA libraries found to be linked with llama-server"
        echo "This might indicate llama.cpp was not built with CUDA support correctly"
        echo "Rebuilding llama.cpp with CUDA support..."
        /app/docker/app/rebuild_llama_cuda.sh
    else
        echo "CUDA support is verified!"
        echo "CUDA libraries linked to llama-server:"
        echo "$LOADED_LIBS"
    fi
fi