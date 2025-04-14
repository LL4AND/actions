import os
import json
import logging
import psutil
import time
import subprocess
import torch  # Add torch import for CUDA detection
from typing import Iterator, Any, Optional, Generator, Dict
from datetime import datetime
from flask import Response
from openai import OpenAI
from lpm_kernel.api.domains.kernel2.dto.server_dto import ServerStatus, ProcessInfo
from lpm_kernel.configs.config import Config
import uuid

logger = logging.getLogger(__name__)

class LocalLLMService:
    """Service for managing local LLM client and server"""
    
    def __init__(self):
        self._client = None
        self._stopping_server = False
        
    @property
    def client(self) -> OpenAI:
        config = Config.from_env()
        """Get the OpenAI client for local LLM server"""
        if self._client is None:
            base_url = config.get("LOCAL_LLM_SERVICE_URL")
            if not base_url:
                raise ValueError("LOCAL_LLM_SERVICE_URL environment variable is not set")
                
            self._client = OpenAI(
                base_url=base_url,
                api_key="sk-no-key-required"
            )
        return self._client

    def start_server(self, model_path: str, use_gpu: bool = True) -> bool:
        """
        Start the llama-server service with GPU acceleration when available
        
        Args:
            model_path: Path to the GGUF model file
            use_gpu: Whether to use GPU acceleration if available
            
        Returns:
            bool: True if server started successfully, False otherwise
        """
        try:
            # Check if server is already running
            status = self.get_server_status()
            if status.is_running:
                logger.info("LLama server is already running")
                return True

            # Check for CUDA availability if GPU was requested
            cuda_available = torch.cuda.is_available() if use_gpu else False
            gpu_info = ""
            
            if use_gpu and cuda_available:
                gpu_device = torch.cuda.current_device()
                gpu_info = f" using GPU: {torch.cuda.get_device_name(gpu_device)}"
                gpu_memory = torch.cuda.get_device_properties(gpu_device).total_memory / (1024**3)
                
                logger.info(f"CUDA is available. Using GPU acceleration{gpu_info}")
                logger.info(f"CUDA device capabilities: {torch.cuda.get_device_capability(gpu_device)}")
                logger.info(f"CUDA memory: {gpu_memory:.2f} GB")
                
                # Pre-initialize CUDA to speed up first inference
                logger.info("Pre-initializing CUDA context to speed up first inference")
                torch.cuda.init()
                torch.cuda.empty_cache()
            elif use_gpu and not cuda_available:
                logger.warning("CUDA was requested but is not available. Using CPU instead.")
            else:
                logger.info("Using CPU for inference (GPU not requested)")

            # Check for GPU optimization marker
            gpu_optimized = False
            model_dir = os.path.dirname(model_path)
            gpu_marker_path = os.path.join(model_dir, "gpu_optimized.json")
            if os.path.exists(gpu_marker_path):
                try:
                    with open(gpu_marker_path, 'r') as f:
                        gpu_data = json.load(f)
                        if gpu_data.get("gpu_optimized", False):
                            gpu_optimized = True
                            logger.info(f"Found GPU optimization marker created on {gpu_data.get('optimized_on', 'unknown date')}")
                except Exception as e:
                    logger.warning(f"Error reading GPU marker file: {e}")

            # Get the correct path to the llama-server executable
            base_dir = os.getcwd()
            server_path = os.path.join(base_dir, "llama.cpp", "build", "bin", "llama-server")
            
            # For Windows, add .exe extension if needed
            if os.name == 'nt' and not server_path.endswith('.exe'):
                server_path += '.exe'
                
            # Verify executable exists
            if not os.path.exists(server_path):
                logger.error(f"llama-server executable not found at: {server_path}")
                return False
                
            # Start server with optimal parameters for faster startup
            cmd = [
                server_path,
                "-m", model_path,
                "--host", "0.0.0.0",
                "--port", "8080",
                "--ctx-size", "2048",     # Default context size (adjust based on needs)
                "--parallel", "2",        # Enable request parallelism
                "--cont-batching"         # Enable continuous batching
            ]
            
            # Set up environment with CUDA variables to ensure GPU detection
            env = os.environ.copy()
            
            # Add GPU-related parameters if CUDA is available
            if cuda_available and use_gpu:
                # Force GPU usage with optimal parameters for faster loads
                cmd.extend([
                    "--n-gpu-layers", "999",  # Use all layers on GPU
                    "--tensor-split", "0",    # Use the first GPU for all operations
                    "--main-gpu", "0",        # Use GPU 0 as the primary device
                    "--mlock"                 # Lock memory to prevent swapping during inference
                ])
                
                # Set CUDA environment variables to help with GPU detection
                env["CUDA_VISIBLE_DEVICES"] = "0"  # Force using first GPU
                
                # Ensure comprehensive library paths for CUDA
                cuda_lib_paths = [
                    "/usr/local/cuda/lib64",
                    "/usr/lib/cuda/lib64",
                    "/usr/local/lib",
                    "/usr/lib/x86_64-linux-gnu",
                    "/usr/lib/wsl/lib"  # For Windows WSL environments
                ]
                
                # Build a comprehensive LD_LIBRARY_PATH
                current_ld_path = env.get("LD_LIBRARY_PATH", "")
                for path in cuda_lib_paths:
                    if os.path.exists(path) and path not in current_ld_path:
                        current_ld_path = f"{path}:{current_ld_path}" if current_ld_path else path
                
                env["LD_LIBRARY_PATH"] = current_ld_path
                logger.info(f"Setting LD_LIBRARY_PATH to: {current_ld_path}")
                
                # If this is Windows, use different approach for CUDA libraries
                if os.name == 'nt':
                    # Windows typically has CUDA in PATH already if installed
                    logger.info("Windows system detected, using system CUDA libraries")
                else:
                    # On Linux, try to find CUDA libraries in common locations
                    for cuda_path in [
                        # Common CUDA paths
                        "/usr/local/cuda/lib64",
                        "/usr/lib/cuda/lib64",
                        "/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib",
                        "/usr/local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib",
                    ]:
                        if os.path.exists(cuda_path):
                            # Add CUDA path to library path
                            env["LD_LIBRARY_PATH"] = f"{cuda_path}:{env.get('LD_LIBRARY_PATH', '')}"
                            env["CUDA_HOME"] = os.path.dirname(cuda_path)
                            logger.info(f"Found CUDA at {cuda_path}, setting environment variables")
                            break
                            
                # Check if we need to rebuild llama.cpp with CUDA support
                if self._should_rebuild_with_cuda():
                    logger.info("Detected llama.cpp might be compiled without CUDA support, attempting to rebuild...")
                    if self._rebuild_llama_with_cuda(env):
                        logger.info("Successfully rebuilt llama.cpp with CUDA support!")
                    else:
                        logger.warning("Failed to rebuild llama.cpp with CUDA support, proceeding with CPU mode")
                
                # Pre-heat GPU to ensure faster initial response
                if torch.cuda.is_available():
                    logger.info("Pre-warming GPU to reduce initial latency...")
                    dummy_tensor = torch.zeros(1, 1).cuda()
                    del dummy_tensor
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    logger.info("GPU warm-up complete")
                
                logger.info("Using GPU acceleration for inference with optimized settings")
            else:
                # If GPU isn't available or supported, optimize for CPU
                cmd.extend([
                    "--threads", str(max(1, os.cpu_count() - 1)),  # Use all CPU cores except one
                ])
                logger.info(f"Using CPU-only mode with {max(1, os.cpu_count() - 1)} threads")
            
            logger.info(f"Starting llama-server with command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                env=env
            )
            
            # Wait for server to start (longer wait for GPU initialization)
            wait_time = 5 if cuda_available and use_gpu else 3
            logger.info(f"Waiting {wait_time} seconds for server to start...")
            time.sleep(wait_time)
            
            # Check if process is still running
            if process.poll() is None:
                # Log initialization success
                if cuda_available and use_gpu:
                    logger.info(f"✅ LLama server started successfully with GPU acceleration{gpu_info}")
                else:
                    logger.info("✅ LLama server started successfully in CPU-only mode")
                return True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"Failed to start llama-server: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting llama-server: {str(e)}")
            return False

    def stop_server(self) -> ServerStatus:
        """
        Stop the llama-server service.
        Find and forcibly terminate all llama-server processes
        
        Returns:
            ServerStatus: Service status object containing information about whether processes are still running
        """
        try:
            if self._stopping_server:
                logger.info("Server is already in the process of stopping")
                return self.get_server_status()
            
            self._stopping_server = True
        
            try:
                # Find all possible llama-server processes and forcibly terminate them
                terminated_pids = []
                for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                    try:
                        cmdline = proc.cmdline()
                        if any("llama-server" in cmd for cmd in cmdline):
                            pid = proc.pid
                            logger.info(f"Force terminating llama-server process, PID: {pid}")
                            
                            # Directly use kill signal to forcibly terminate
                            proc.kill()
                            
                            # Ensure the process has been terminated
                            try:
                                proc.wait(timeout=0.2)  # Slightly increase wait time to ensure process termination
                                terminated_pids.append(pid)
                                logger.info(f"Successfully terminated llama-server process {pid}")
                            except psutil.TimeoutExpired:
                                # If timeout, try to terminate again
                                logger.warning(f"Process {pid} still running, sending SIGKILL again")
                                try:
                                    import os
                                    import signal
                                    os.kill(pid, signal.SIGKILL)  # Use system-level SIGKILL signal
                                    terminated_pids.append(pid)
                                    logger.info(f"Successfully force killed llama-server process {pid} with SIGKILL")
                                except ProcessLookupError:
                                    # Process no longer exists
                                    terminated_pids.append(pid)
                                    logger.info(f"Process {pid} no longer exists after kill attempt")
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
                
                if terminated_pids:
                    logger.info(f"Terminated llama-server processes: {terminated_pids}")
                else:
                    logger.info("No running llama-server process found")
                
                # Check again if any llama-server processes are still running
                return self.get_server_status()
            
            finally:
                self._stopping_server = False
            
        except Exception as e:
            logger.error(f"Error stopping llama-server: {str(e)}")
            self._stopping_server = False
            return ServerStatus.not_running()

    def get_server_status(self) -> ServerStatus:
        """
        Get the current status of llama-server
        Returns: ServerStatus object
        """
        try:
            base_dir = os.getcwd()
            server_path = os.path.join(base_dir, "llama.cpp", "build", "bin", "llama-server")
            server_exec_name = os.path.basename(server_path)
            
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    cmdline = proc.cmdline()
                    # Check both for the executable name and the full path
                    if any(server_exec_name in cmd for cmd in cmdline) or any("llama-server" in cmd for cmd in cmdline):
                        with proc.oneshot():
                            process_info = ProcessInfo(
                                pid=proc.pid,
                                cpu_percent=proc.cpu_percent(),
                                memory_percent=proc.memory_percent(),
                                create_time=proc.create_time(),
                                cmdline=cmdline,
                            )
                            return ServerStatus.running(process_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                    
            return ServerStatus.not_running()
            
        except Exception as e:
            logger.error(f"Error checking llama-server status: {str(e)}")
            return ServerStatus.not_running()

    def _should_rebuild_with_cuda(self) -> bool:
        """
        Check if llama.cpp needs to be rebuilt with CUDA support
        
        Returns:
            bool: True if we should rebuild, False otherwise
        """
        # First check if a recent rebuild marker exists to avoid unnecessary rebuilds
        base_dir = os.getcwd()
        rebuild_marker = os.path.join(base_dir, ".llama_cuda_rebuild_marker")
        
        # If marker exists and is less than 24 hours old, we've already tried rebuilding recently
        if os.path.exists(rebuild_marker):
            try:
                marker_time = os.path.getmtime(rebuild_marker)
                if (time.time() - marker_time) < 86400:  # 24 hours in seconds
                    logger.info("Found recent rebuild marker, skipping rebuild check")
                    return False
            except Exception as e:
                logger.warning(f"Error checking rebuild marker: {e}")
        
        # Try running llama-server with --help to check if it mentions GPU support
        try:
            server_path = os.path.join(base_dir, "llama.cpp", "build", "bin", "llama-server")
            if os.name == 'nt' and not server_path.endswith('.exe'):
                server_path += '.exe'
                
            if not os.path.exists(server_path):
                logger.warning("Cannot check for CUDA support: llama-server not found")
                return False
                
            # First check if libcuda is properly linked
            try:
                ldd_output = subprocess.run(
                    ["ldd", server_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5
                ).stdout
                
                has_cuda_lib = "libcuda.so" in ldd_output or "libggml-cuda" in ldd_output
                logger.info(f"CUDA library check: {'Found' if has_cuda_lib else 'Not found'} in linked libraries")
                
                # If there's a CUDA library linked, it's likely compiled with CUDA support
                if has_cuda_lib:
                    logger.info("CUDA library is linked to llama-server binary, assuming CUDA support is available")
                    return False
            except Exception as e:
                # ldd might not be available on all platforms, so we'll continue with other checks
                logger.warning(f"Could not check linked libraries: {e}")
            
            # Run with --help to check output
            result = subprocess.run(
                [server_path, "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )
            
            output = result.stdout + result.stderr
            
            # Check for specific indicators that GPU support is NOT compiled in
            no_gpu_indicators = [
                "llama.cpp was compiled without support for GPU",
                "without GPU support",
                "no usable GPU found",
            ]
            
            # Don't include "--gpu-layers option will be ignored" as it might appear 
            # even when GPU support is available but not applicable to the specific command
            
            # Look for positive GPU support indicators - need just ONE of these to be present
            gpu_support_indicators = [
                "--n-gpu-layers",
                "--tensor-split"
            ]
            
            # Check if we have negative indicators
            has_negative = any(indicator in output for indicator in no_gpu_indicators)
            
            # Check if we have at least one positive indicator
            has_positive = any(indicator in output for indicator in gpu_support_indicators)
            
            # Log the detailed check results
            logger.info(f"CUDA support indicators: Negative={has_negative}, Positive={has_positive}")
            
            # If we have any positive indicator, we likely have GPU support
            if has_positive:
                logger.info("CUDA support positive indicators found, no need to rebuild")
                return False
                
            # Only need rebuild if we have negative indicators or NONE of the positive ones
            needs_rebuild = has_negative or not has_positive
            
            if needs_rebuild:
                logger.warning("Detected llama.cpp without proper GPU support")
            
            return needs_rebuild
            
        except Exception as e:
            logger.error(f"Error checking llama.cpp GPU support: {e}")
            return False
            
    def _rebuild_llama_with_cuda(self, env: Dict[str, str] = None) -> bool:
        """
        Rebuild llama.cpp with CUDA support
        
        Args:
            env: Environment variables to use for the build
            
        Returns:
            bool: True if rebuild was successful, False otherwise
        """
        try:
            base_dir = os.getcwd()
            
            # Create rebuild marker to avoid repeated rebuild attempts
            rebuild_marker = os.path.join(base_dir, ".llama_cuda_rebuild_marker")
            with open(rebuild_marker, 'w') as f:
                f.write(f"Rebuild attempted at {datetime.now().isoformat()}")
                
            # Check if we have a rebuild script
            rebuild_script = os.path.join(base_dir, "docker", "app", "rebuild_llama_cuda.sh")
            
            if os.path.exists(rebuild_script):
                logger.info(f"Found rebuild script: {rebuild_script}")
                
                # Execute the rebuild script
                if env is None:
                    env = os.environ.copy()
                    
                # Ensure we have proper CUDA environment variables
                if "CUDA_HOME" not in env and os.path.exists("/usr/local/cuda"):
                    env["CUDA_HOME"] = "/usr/local/cuda"
                
                # Set flag to force rebuild even if SKIP_LLAMA_REBUILD is set
                env["FORCE_CUDA_REBUILD"] = "true"
                
                process = subprocess.Popen(
                    ["bash", rebuild_script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    env=env
                )
                
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    logger.info(f"Successfully rebuilt llama.cpp with CUDA support: {stdout}")
                    return True
                else:
                    logger.error(f"Failed to rebuild llama.cpp: {stderr}")
                    return False
            else:
                # If no script exists, try to rebuild manually
                logger.info("No rebuild script found, attempting manual rebuild")
                
                # Check if we have the llama.cpp directory
                llama_cpp_dir = os.path.join(base_dir, "llama.cpp")
                
                if not os.path.exists(llama_cpp_dir):
                    logger.error("llama.cpp directory not found, cannot rebuild")
                    return False
                
                # First, try to clean up any previous build
                clean_process = subprocess.Popen(
                    ["make", "clean"],
                    cwd=llama_cpp_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    env=env
                )
                clean_process.communicate()
                
                # Now build with CUDA support
                build_cmd = [
                    "make", 
                    "LLAMA_CUBLAS=1", 
                    "-j",
                    "server"
                ]
                
                logger.info(f"Running build command: {' '.join(build_cmd)}")
                build_process = subprocess.Popen(
                    build_cmd,
                    cwd=llama_cpp_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    env=env
                )
                
                stdout, stderr = build_process.communicate()
                
                if build_process.returncode == 0:
                    logger.info(f"Successfully built llama.cpp with CUDA support")
                    return True
                else:
                    logger.error(f"Failed to build llama.cpp with CUDA: {stderr}")
                    return False
                
        except Exception as e:
            logger.error(f"Error during llama.cpp rebuild: {str(e)}")
            return False

    def _parse_response_chunk(self, chunk):
        """Parse different response chunk formats into a standardized format."""
        try:
            if chunk is None:
                logger.warning("Received None chunk")
                return None
                
            # logger.info(f"Parsing response chunk: {chunk}")
            # Handle custom format
            if isinstance(chunk, dict) and "type" in chunk and chunk["type"] == "chat_response":
                logger.info(f"Processing custom format response: {chunk}")
                return {
                    "id": str(uuid.uuid4()),  # Generate a unique ID
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": "models/lpm",
                    "system_fingerprint": None,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": chunk.get("content", "")
                            },
                            "finish_reason": "stop" if chunk.get("done", False) else None
                        }
                    ]
                }
            
            # Handle OpenAI format
            if not hasattr(chunk, 'choices'):
                logger.warning(f"Chunk has no choices attribute: {chunk}")
                return None
                
            choices = getattr(chunk, 'choices', [])
            if not choices:
                logger.warning("Chunk has empty choices")
                return None
                
            # logger.info(f"Processing OpenAI format response: choices={choices}")
            delta = choices[0].delta
            
            # Create standard response structure
            response_data = {
                "id": chunk.id,
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": "models/lpm",
                "system_fingerprint": chunk.system_fingerprint if hasattr(chunk, 'system_fingerprint') else None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            # Keep even if content is None, let the client handle it
                            "content": delta.content if hasattr(delta, 'content') else ""
                        },
                        "finish_reason": choices[0].finish_reason
                    }
                ]
            }
            
            # If there is neither content nor finish_reason, skip
            if not (hasattr(delta, 'content') or choices[0].finish_reason):
                logger.debug("Skipping chunk with no content and no finish_reason")
                return None
                
            return response_data
            
        except Exception as e:
            logger.error(f"Error parsing response chunk: {e}, chunk: {chunk}")
            return None

    def handle_stream_response(self, response_iter: Iterator[Any]) -> Response:
        """Handle streaming response from the LLM server"""
        def generate():
            chunk = None  # Initialize chunk variable
            try:
                for chunk in response_iter:
                    if chunk is None:
                        logger.warning("Received None chunk in stream, skipping")
                        continue
                        
                    # logger.info(f"Received raw chunk: {chunk}")
                    # Check if this is the done marker for custom format
                    if chunk == "[DONE]":
                        logger.info("Received [DONE] marker")
                        yield b"data: [DONE]\n\n"
                        return  # Use return instead of break to ensure [DONE] in finally won't be executed
                    
                    # Handle OpenAI error format directly
                    if isinstance(chunk, dict) and "error" in chunk:
                        logger.warning(f"Received error response: {chunk}")
                        data_str = json.dumps(chunk)
                        yield f"data: {data_str}\n\n".encode('utf-8')
                        # After sending error, send [DONE] marker to close the stream properly
                        yield b"data: [DONE]\n\n"
                        return
                    
                    response_data = self._parse_response_chunk(chunk)
                    if response_data:
                        data_str = json.dumps(response_data)
                        # logger.info(f"Sending response data: {data_str}")
                        yield f"data: {data_str}\n\n".encode('utf-8')
                    else:
                        logger.warning("Parsed response data is None, skipping chunk")
                    
            except Exception as e:
                error_msg = json.dumps({'error': str(e)})
                logger.error(f"Failed to process stream response: {str(e)}", exc_info=True)
                yield f"data: {error_msg}\n\n".encode('utf-8')
            finally:
                if chunk != "[DONE]":  # Only send if [DONE] marker was not received
                    logger.info("Sending final [DONE] marker")
                    yield b"data: [DONE]\n\n"
                logger.info("Stream response completed successfully")

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache, no-transform',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive',
                'Transfer-Encoding': 'chunked'
            }
        )


# Global instance
local_llm_service = LocalLLMService()
