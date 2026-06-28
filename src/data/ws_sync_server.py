"""
WebSocket Synchronization Server.

Listens for incoming binary payloads (PyTorch state_dicts) streamed directly
from the Cloud GPU (Colab) and saves them in real-time to the local SSD.
This completely bypasses HTTP API bottlenecks and ensures you never lose
training progress.
"""
import asyncio
import logging
from pathlib import Path
import websockets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WSSyncServer")

# Ensure checkpoints directory exists
SAVE_DIR = Path("checkpoints")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

async def handle_client(websocket):
    """Handles an incoming WebSocket connection from the Colab Streamer."""
    remote_ip = websocket.remote_address[0]
    logger.info(f"Connected to GPU Worker at {remote_ip}")
    
    try:
        # First message should be the model name (e.g., 'lobert' or 'fincast')
        model_name = await websocket.recv()
        logger.info(f"Receiving sync stream for model: {model_name}")
        
        while True:
            # Wait for binary blob
            blob = await websocket.recv()
            if isinstance(blob, bytes):
                # Save to disk instantly
                save_path = SAVE_DIR / f"{model_name}_live.pt"
                with open(save_path, "wb") as f:
                    f.write(blob)
                
                size_mb = len(blob) / (1024 * 1024)
                logger.info(f"✅ Received and saved {size_mb:.2f} MB to {save_path}")
                
                # Send ACK back to Colab so it knows we got it
                await websocket.send("ACK")
            else:
                logger.warning("Received non-binary data. Ignoring.")
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"GPU Worker disconnected ({remote_ip})")
    except Exception as e:
        logger.error(f"Error handling connection: {e}")

async def main(port=8765):
    logger.info(f"Starting WebSocket Sync Server on port {port}...")
    logger.info("To expose to Colab, run: bore local 8765 --to bore.pub")
    async with websockets.serve(handle_client, "0.0.0.0", port):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shut down manually.")
