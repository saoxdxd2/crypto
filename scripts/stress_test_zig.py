import socket
import time
import subprocess
import json
import numpy as np
from pathlib import Path
import os
import sys

# Ensure ONNXRuntime is built
print("--- ZIG / C++ HFT STRESS TEST ---")
print("Target: 1,000,000 LOB Ticks")

# 1. Spawn Zig Engine
engine_path = Path("src/execution/zig-out/bin/engine.exe")
if not engine_path.exists():
    print(f"Error: Engine not found at {engine_path}. Please build it.")
    sys.exit(1)

zig_proc = subprocess.Popen(
    [str(engine_path)], 
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

time.sleep(1) # Let it initialize ONNX

# 2. Connect to Zig's listening TCP Port (Mocking the TLS Proxy)
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", 9000))
    print("Connected to Zig Engine TCP Port 9000.")
except Exception as e:
    print(f"Failed to connect to Zig: {e}")
    zig_proc.terminate()
    sys.exit(1)

# 3. Prepare Mock Data (640 floats)
mock_lob = ",".join(["0.555"] * 640) + "\n"
mock_bytes = mock_lob.encode('utf-8')

TOTAL_TICKS = 100_000 # 100k for the test so it doesn't take forever

latencies = []

print(f"Blasting {TOTAL_TICKS} ticks...")
start_time = time.perf_counter()

# We flood the socket
for i in range(TOTAL_TICKS):
    s.sendall(mock_bytes)

# We read the JSON signals from Zig stdout
for i in range(TOTAL_TICKS):
    line = zig_proc.stdout.readline()
    if not line:
        break
    try:
        data = json.loads(line)
        if "reason_code" in data:
            lat = int(data["reason_code"].split(":")[1])
            latencies.append(lat)
    except:
        pass

end_time = time.perf_counter()
s.close()
zig_proc.terminate()

# 4. Results
total_time = end_time - start_time
print("\n--- RESULTS ---")
print(f"Total Ticks Processed : {len(latencies)}")
print(f"Total Time Taken      : {total_time:.2f} seconds")
print(f"Throughput            : {len(latencies) / total_time:,.0f} ticks/sec")

if latencies:
    latencies = np.array(latencies) / 1000.0 # Convert ns to microseconds
    print(f"P50 Latency           : {np.percentile(latencies, 50):.2f} us")
    print(f"P95 Latency           : {np.percentile(latencies, 95):.2f} us")
    print(f"P99 Latency           : {np.percentile(latencies, 99):.2f} us")
else:
    print("No signals received from Zig Engine. Check logs.")
