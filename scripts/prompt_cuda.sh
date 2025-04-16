#!/bin/bash
# Script to prompt user for CUDA support preference and directly build with the appropriate Dockerfile

echo "=== CUDA Support Selection ==="
echo ""
echo "Do you want to build with NVIDIA GPU (CUDA) support?"
echo "This requires an NVIDIA GPU and proper NVIDIA Docker runtime configuration."
echo ""
read -p "Build with CUDA support? (y/n): " choice

case "$choice" in
  y|Y|yes|YES|Yes )
    echo "Selected: Build WITH CUDA support"
    
    # Create or update .env file with the Dockerfile selection
    if [ -f .env ]; then
      # Update existing file
      grep -q "DOCKER_BACKEND_DOCKERFILE" .env && \
        sed -i 's/^DOCKER_BACKEND_DOCKERFILE=.*/DOCKER_BACKEND_DOCKERFILE=Dockerfile.backend.cuda/' .env || \
        echo "DOCKER_BACKEND_DOCKERFILE=Dockerfile.backend.cuda" >> .env
    else
      # Create new file
      echo "DOCKER_BACKEND_DOCKERFILE=Dockerfile.backend.cuda" > .env
    fi
    
    # Create a new docker-compose override file that explicitly sets the Dockerfile
    cat > docker-compose.override.yml << EOF
services:
  backend:
    build:
      dockerfile: Dockerfile.backend.cuda
EOF
    
    echo "Environment set to build with CUDA support"
    ;;
  * )
    echo "Selected: Build WITHOUT CUDA support (CPU only)"
    
    # Create or update .env file with the Dockerfile selection
    if [ -f .env ]; then
      # Update existing file
      grep -q "DOCKER_BACKEND_DOCKERFILE" .env && \
        sed -i 's/^DOCKER_BACKEND_DOCKERFILE=.*/DOCKER_BACKEND_DOCKERFILE=Dockerfile.backend/' .env || \
        echo "DOCKER_BACKEND_DOCKERFILE=Dockerfile.backend" >> .env
    else
      # Create new file
      echo "DOCKER_BACKEND_DOCKERFILE=Dockerfile.backend" > .env
    fi
    
    # Remove any override file if it exists
    if [ -f docker-compose.override.yml ]; then
      rm docker-compose.override.yml
    fi
    
    echo "Environment set to build without CUDA support"
    ;;
esac

echo "=== CUDA Selection Complete ==="