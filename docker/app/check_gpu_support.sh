#!/bin/bash
# Helper script to check if GPU support is available at runtime
# This replaces the runtime rebuild logic with a simple check

# Check for GPU optimization marker
GPU_MARKER_FILE="/app/data/gpu_optimized.json"
if [ -f "$GPU_MARKER_FILE" ]; then
    GPU_OPTIMIZED=$(grep -o '"gpu_optimized": *true' "$GPU_MARKER_FILE" || echo "false")
    OPTIMIZED_DATE=$(grep -o '"optimized_on": *"[^"]*"' "$GPU_MARKER_FILE" | cut -d'"' -f4)
    
    if [[ "$GPU_OPTIMIZED" == *"true"* ]]; then
        echo "GPU-optimized build detected (built on: $OPTIMIZED_DATE)"
        
        # Verify that CUDA is still available
        if nvidia-smi &>/dev/null && [ -f "/app/llama.cpp/build/bin/llama-server" ]; then
            # Check if the binary is linked against CUDA libraries
            CUDA_LINKED=$(ldd /app/llama.cpp/build/bin/llama-server 2>/dev/null | grep -i "cuda\|nvidia")
            if [ -n "$CUDA_LINKED" ]; then
                echo "CUDA support verified - GPU acceleration is available"
                exit 0
            else
                echo "WARNING: GPU-optimized build marker found, but llama-server is not linked to CUDA libraries"
                echo "This may indicate a problem with the build or CUDA configuration"
            fi
        else
            echo "WARNING: GPU-optimized build marker found, but NVIDIA GPU is not accessible"
            echo "Check that Docker is running with GPU access (--gpus all)"
        fi
    else
        echo "CPU-only build detected (built on: $OPTIMIZED_DATE)"
        echo "To enable GPU support, rebuild the container with: scripts/rebuild_with_cuda.sh"
    fi
else
    echo "No GPU optimization marker found"
    echo "This container was likely built without checking for GPU support"
    echo "To enable GPU support, rebuild the container with: scripts/rebuild_with_cuda.sh"
fi

# Return CUDA status for the runtime system
if nvidia-smi &>/dev/null; then
    echo "NVIDIA GPU is available at runtime"
    if [ -f "/app/llama.cpp/build/bin/llama-server" ] && ldd /app/llama.cpp/build/bin/llama-server 2>/dev/null | grep -q "libcuda"; then
        echo "llama-server has CUDA support"
        exit 0
    else
        echo "llama-server does not have CUDA support"
        echo "To enable GPU support, rebuild the container with: scripts/rebuild_with_cuda.sh" 
        exit 1
    fi
else
    echo "No NVIDIA GPU detected at runtime"
    exit 1
fi