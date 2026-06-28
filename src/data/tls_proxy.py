import asyncio
import websockets
import json
import socket
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TLS_PROXY")

# Binance LOB endpoint
BINANCE_WSS = "wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms"

async def handle_zig_client(reader, writer):
    logger.info("Zig Native Engine connected to proxy.")
    try:
        async with websockets.connect(BINANCE_WSS) as ws:
            logger.info("Connected to Binance TLS WebSocket.")
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                
                # Extract bids and asks and flatten into a float array
                bids = [float(p) for p, q in data.get('bids', [])[:16]]
                asks = [float(p) for p, q in data.get('asks', [])[:16]]
                
                # Mock flattening the rest to reach 640 features for LOBERT
                features = bids + asks
                while len(features) < 640:
                    features.append(0.0)
                
                # Pack into binary f32 array (little-endian)
                import struct
                binary_data = struct.pack(f'<{len(features)}f', *features)
                
                # Stream to Zig
                writer.write(binary_data)
                await writer.drain()
                
    except Exception as e:
        logger.error(f"Proxy Connection Lost: {e}")
    finally:
        writer.close()
        await writer.wait_closed()

async def main():
    server = await asyncio.start_server(handle_zig_client, '127.0.0.1', 9000)
    logger.info("TLS Proxy listening on 127.0.0.1:9000")
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
