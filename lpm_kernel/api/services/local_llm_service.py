import os
import json
import logging
import psutil
import time
import subprocess
import torch
from typing import Iterator, Any, Optional, Generator, Dict
from datetime import datetime
from flask import Response
from openai import OpenAI
from lpm_kernel.api.domains.kernel2.dto.server_dto import ServerStatus, ProcessInfo
from lpm_kernel.configs.config import Config
import uuid
import platform

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

    def _detect_gpu_availability(self) -> bool:
        """
        Cross-platform GPU detection that works on macOS, Linux, Windows, 
        including Docker and WSL2 environments
        """
        # Method 1: Check using PyTorch
        if torch.cuda.is_available():
            cuda_device = 0
            cuda_name = torch.cuda.get_device_name(cuda_device)
            logger.info(f"CUDA detected via PyTorch. Using device: {cuda_name}")
            return True
            
        # Platform-specific checks
        current_platform = platform.system().lower()
        
        # Method 2: Check for NVIDIA device files (Linux, WSL2)
        if current_platform == "linux" and (
            os.path.exists("/dev/nvidia0") or 
            os.path.exists("/proc/driver/nvidia")
        ):
            logger.info("NVIDIA GPU detected via device files")
            return True
            
        # Method 3: Try running nvidia-smi (cross-platform)
        try:
            # Use different commands based on platform
            if current_platform == "windows":
                # Check for Windows nvidia-smi
                result = subprocess.run(
                    ["where", "nvidia-smi"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=3
                )
                if result.returncode == 0:
                    # nvidia-smi exists, now try to run it
                    nvidia_smi = subprocess.run(
                        ["nvidia-smi"], 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        check=False,
                        timeout=3
                    )
                    if nvidia_smi.returncode == 0:
                        logger.info("NVIDIA GPU detected via nvidia-smi on Windows")
                        return True
            else:
                # Linux/macOS check
                result = subprocess.run(
                    ["which", "nvidia-smi"],
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=3
                )
                if result.returncode == 0:
                    # nvidia-smi exists, now try to run it
                    nvidia_smi = subprocess.run(
                        ["nvidia-smi"], 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        check=False,
                        timeout=3
                    )
                    if nvidia_smi.returncode == 0:
                        logger.info("NVIDIA GPU detected via nvidia-smi")
                        return True
        except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
            logger.debug(f"Error checking for nvidia-smi: {e}")
        
        # Method 4: Check for WSL-specific NVIDIA integration
        if os.path.exists("/usr/lib/wsl/lib/libnvidia-ml.so"):
            logger.info("NVIDIA GPU detected via WSL2 integration")
            return True
            
        # No GPU detected
        logger.info("No NVIDIA GPU detected. Using CPU only.")
        return False

    def start_server(self, model_path: str) -> bool:
        """
        Start the llama-server service with CUDA support if available
        """
        try:
            # Check if server is already running
            status = self.get_server_status()
            if status.is_running:
                logger.info("LLama server is already running")
                return True

            # Base command
            cmd = [
                "llama-server",
                "-m", model_path,
                "--host", "0.0.0.0",
                "--port", "8000"
            ]
            
            # Check for GPU availability using our cross-platform detection
            if self._detect_gpu_availability():
                # Add GPU-specific parameters
                cmd.extend([
                    "--n-gpu-layers", "99",  # Use as many layers on GPU as possible
                    "--parallel", "2"        # Increase parallelization for better performance
                ])
                
                # Set CUDA environment variables
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                
                logger.info("Starting llama-server with GPU acceleration")
            else:
                logger.info("Starting llama-server with CPU only")
            
            logger.info(f"Starting llama-server with command: {cmd}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                env=dict(os.environ)
            )
            
            # Wait for server to start
            time.sleep(2)
            
            # Check if process started successfully
            if process.poll() is None:
                logger.info("LLama server started successfully")
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
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    cmdline = proc.cmdline()
                    if any("llama-server" in cmd for cmd in cmdline):
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
