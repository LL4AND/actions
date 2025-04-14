#!/bin/bash
# Script to restart Docker services and rebuild llama.cpp with CUDA support

echo "==== Stopping Second-Me services ===="
docker-compose down

echo "==== Removing llama-cpp-build volume to force rebuild ===="
docker volume rm second-me_llama-cpp-build || true

echo "==== Starting Second-Me backend with GPU support ===="
# Start only the backend service with GPU enabled
docker-compose up -d backend

echo "==== Following logs to monitor rebuild process ===="
echo "This will show the rebuild process. Wait until you see 'REBUILD COMPLETED SUCCESSFULLY'"
echo "Press Ctrl+C to stop viewing logs when rebuild is complete"
docker-compose logs -f backend

echo ""
echo "==== Starting remaining services ===="
docker-compose up -d

echo ""
echo "==== Rebuild process complete ===="
echo "You can check CUDA status with:"
echo "docker exec second-me-backend /app/test_cuda.sh"