"""
Local PyTorch Distributed RPC Master Node.

This module connects the local pipeline to the Google Colab GPU worker.
It transparently offloads heavy tensor operations (like LOBERT or FinCast)
over the TCP tunnel.
"""
import os
import logging
import torch
import torch.distributed.rpc as rpc

logger = logging.getLogger(__name__)

class CloudGPUClient:
    """
    Connects to the Colab GPU worker and provides a seamless interface
    to run remote functions.
    """
    def __init__(self, remote_url: str):
        self.remote_url = remote_url
        self.worker_name = "worker0"
        self._init_rpc()

    def _init_rpc(self):
        # The local master binds to an arbitrary open port locally
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29501'

        logger.info(f"Connecting to Cloud GPU at {self.remote_url}...")
        
        try:
            # TensorPipe is only supported on Linux/macOS. 
            # On Windows, this will throw an AttributeError.
            options = rpc.TensorPipeRpcBackendOptions(
                init_method=self.remote_url
            )
            rpc.init_rpc(
                "master",
                rank=0,
                world_size=2,
                rpc_backend_options=options
            )
            logger.info("Successfully connected to Cloud GPU via PyTorch RPC!")
        except Exception as e:
            logger.error(f"Failed to connect to RPC worker (or RPC not supported on this OS): {e}")
            logger.warning("Falling back to local CPU execution.")

    def run_remote(self, func, *args, **kwargs):
        """
        Executes a function on the remote Colab GPU.
        Example: `client.run_remote(lobert_forward_pass, data_tensor)`
        """
        try:
            return rpc.rpc_sync(self.worker_name, func, args=args, kwargs=kwargs)
        except Exception as e:
            logger.error(f"Remote execution failed: {e}")
            # Fallback to local
            return func(*args, **kwargs)
            
    def shutdown(self):
        rpc.shutdown()
