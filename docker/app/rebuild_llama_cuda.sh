#!/bin/bash
# Script to rebuild llama.cpp with CUDA support at runtime
# This ensures the build happens with full knowledge of the GPU environment

set -ex  # Exit on error and print each command as it's executed (for better debugging)
cd /app

echo "========== STARTING LLAMA.CPP CUDA REBUILD PROCESS =========="
echo "Current directory: $(pwd)"
echo "CUDA environment variables:"
echo "- CUDA_HOME: $CUDA_HOME"
echo "- LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "- PATH: $PATH"
echo "=====================================================\n"

# Get calling script name - if it's directly called from local_llm_service.py
# we should override the SKIP_LLAMA_REBUILD flag
CALLER=$(ps -o comm= $PPID 2>/dev/null || echo "unknown")
FORCE_REBUILD="false"

# Check if we're being called from the detect_cuda mechanism in local_llm_service.py
if [ -n "$FORCE_CUDA_REBUILD" ] || [ "$CALLER" == "python3" ] || [ "$CALLER" == "python" ]; then
    echo "Called from Python script or with FORCE_CUDA_REBUILD flag, forcing rebuild regardless of SKIP_LLAMA_REBUILD setting"
    FORCE_REBUILD="true"
fi

# First check if CUDA is actually available in the container
echo "Verifying NVIDIA drivers and CUDA availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: NVIDIA drivers not found. Cannot build with CUDA support!"
    echo "Make sure the container has access to the GPU and NVIDIA Container Toolkit is installed."
    echo "Consider running Docker with: --gpus all"
    exit 1
fi

# Run nvidia-smi to check GPU access
echo "Detected NVIDIA GPU:"
nvidia-smi || {
    echo "ERROR: nvidia-smi command failed. GPU is not properly accessible from the container."
    echo "Make sure you're running Docker with GPU access enabled (--gpus all)"
    exit 1
}

# Make sure CUDA is available and properly configured
echo "Checking CUDA compiler..."
if ! command -v nvcc &> /dev/null; then
    echo "CUDA compiler (nvcc) not found. Installing CUDA toolkit..."
    
    # Add NVIDIA repositories and install CUDA toolkit
    apt-get update
    apt-get install -y --no-install-recommends wget gnupg

    # Add NVIDIA repository
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb

    # Update package list
    apt-get update
    
    echo "Installing CUDA Toolkit using NVIDIA's recommended approach..."
    # Try the latest CUDA toolkit first, falling back to earlier versions if needed
    apt-get install -y --no-install-recommends cuda-toolkit-12-8 || \
    apt-get install -y --no-install-recommends cuda-toolkit-12-7 || \
    apt-get install -y --no-install-recommends cuda-toolkit-12-6 || \
    apt-get install -y --no-install-recommends cuda-toolkit-12-5 || \
    apt-get install -y --no-install-recommends cuda-toolkit-12-4 || \
    apt-get install -y --no-install-recommends cuda-toolkit-12-3 || \
    apt-get install -y --no-install-recommends cuda-toolkit-12-2 || \
    apt-get install -y --no-install-recommends cuda-toolkit-12-1 || \
    apt-get install -y --no-install-recommends cuda-toolkit-12-0
    
    # If the specific toolkit versions fail, try a generic install 
    if ! command -v nvcc &> /dev/null; then
        echo "Specific CUDA toolkit version not found, trying generic installation..."
        apt-get install -y --no-install-recommends cuda
    fi
else
    echo "CUDA compiler (nvcc) found: $(which nvcc)"
    nvcc --version
fi

# Find the CUDA installation path
CUDA_PATH=$(dirname $(dirname $(which nvcc)))
if [ ! -d "$CUDA_PATH" ]; then
    echo "Could not determine CUDA path from nvcc. Checking default location..."
    CUDA_PATH=/usr/local/cuda
fi

if [ ! -d "$CUDA_PATH" ]; then
    echo "ERROR: CUDA installation not found at $CUDA_PATH!"
    exit 1
fi

echo "Using CUDA toolkit at: $CUDA_PATH"

# Add CUDA to PATH and LD_LIBRARY_PATH if not already there
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$CUDA_PATH

echo "Verifying CUDA environment after setup:"
echo "- CUDA_HOME: $CUDA_HOME"
echo "- LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "- PATH: $PATH"
echo "nvcc path: $(which nvcc)"

# Check if we need the build dependencies
echo "Checking for build dependencies..."
# Make sure we have git and build essentials
apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    libopenblas-dev

# If llama.cpp directory doesn't exist, clone it
if [ ! -d "/app/llama.cpp" ]; then
    echo "llama.cpp directory not found, cloning fresh copy..."
    git clone https://github.com/ggerganov/llama.cpp.git /app/llama.cpp
    cd /app/llama.cpp
else
    # If directory exists, make sure it's clean
    cd /app/llama.cpp
    git reset --hard || echo "Not a git repository, continuing with existing files"
fi

# Make sure build directory exists
mkdir -p build
cd build

# Clean CMake cache completely to ensure fresh build
echo "Cleaning CMake cache..."
rm -rf *

# Configure with CMake
echo "\n\n========== CONFIGURING LLAMA.CPP WITH CMAKE ==========\n\n"
cmake -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=all \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DLLAMA_NATIVE=OFF \
      ..

# Build the code
echo "\n\n========== BUILDING LLAMA.CPP ==========\n\n"
cmake --build . --config Release --target server --clean-first -j $(nproc)

# Make executables executable
echo "Making executables executable"
chmod +x bin/llama-server bin/llama-cli || echo "Some executables might not exist"

# Verify the build worked and has CUDA support
if [ -f "bin/llama-server" ]; then
    echo "\n\n========== VERIFYING CUDA SUPPORT ==========\n\n"
    # Test if CUDA is actually working now
    TEST_RESULT=$(LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH ./bin/llama-server --help 2>&1)
    if echo "$TEST_RESULT" | grep -q "llama.cpp was compiled without support for GPU\|no usable GPU found"; then
        echo "ERROR: Build completed but llama-server still reports no CUDA support!"
        echo "Build appears to have failed to integrate CUDA properly."
        echo "Test output: $TEST_RESULT"
        exit 1
    else
        # Create a simple test script to verify CUDA support
        echo "Creating test script to verify CUDA support"
        cat > /app/test_cuda.sh << 'EOL'
#!/bin/bash
echo "Testing CUDA support in llama.cpp:"
cd /app/llama.cpp/build
echo "Server version:"
./bin/llama-server --version
echo -e "\nChecking for CUDA initialization messages:"
LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH ./bin/llama-server --verbose-prompt --n-gpu-layers 1 2>&1 | grep -i 'cuda\|gpu'
EOL
        chmod +x /app/test_cuda.sh
        
        echo "\n\nBuild SUCCESSFUL! llama-server binary with CUDA support created at /app/llama.cpp/build/bin/llama-server"
        echo "To verify GPU support, run: /app/test_cuda.sh"
        
        # Run the test script to confirm CUDA support
        echo "\n\n========== RUNNING CUDA TEST ==========\n\n"
        /app/test_cuda.sh
        
        echo "\n\n========== REBUILD COMPLETED SUCCESSFULLY ==========\n\n"
    fi
else
    echo "ERROR: Build failed - llama-server executable not found!"
    exit 1
fi