# main_client.py

from flask import Flask, request
import torch
import io
import os
import time
from torchvision import models
from topology_client import TopologyProvider
from availability import AvailabilityTrace
import threading
import requests
import socket
import argparse
from typing import Dict, Tuple


def read_proc_self_status() -> Dict[str, int]:
    info = {}
    with open("/proc/self/status", "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            parts = value.strip().split()
            if parts and parts[0].isdigit():
                info[key.strip()] = int(parts[0])
    return info


def read_proc_self_io() -> Dict[str, int]:
    io_stats = {}
    with open("/proc/self/io", "r") as f:
        for line in f:
            key, value = line.split(":", 1)
            io_stats[key.strip()] = int(value.strip())
    return io_stats


def snapshot_process():
    wall = time.perf_counter()
    cpu = time.process_time()
    status = read_proc_self_status()
    io_stats = read_proc_self_io()
    return wall, cpu, status, io_stats


def summarize_process(start, end):
    wall0, cpu0, status0, io0 = start
    wall1, cpu1, status1, io1 = end
    wall_delta = max(wall1 - wall0, 1e-9)
    cpu_pct = ((cpu1 - cpu0) / wall_delta) * 100.0
    rss_kb = status1.get("VmRSS", 0)
    io_read_delta = io1.get("read_bytes", 0) - io0.get("read_bytes", 0)
    io_write_delta = io1.get("write_bytes", 0) - io0.get("write_bytes", 0)
    return cpu_pct, rss_kb, io_read_delta, io_write_delta



# -------------------------------
# 1. INIT APP + GLOBAL OBJECTS
# -------------------------------
app = Flask(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device_id", required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--server_ip", required=True)
args = parser.parse_args()

# Extract values
device_id_str = args.device_id
port = args.port
main_server_url = f"http://{args.server_ip}:8080"

device_index = AvailabilityTrace.extract_device_index(device_id_str)
trace = AvailabilityTrace("traces/traces.txt", device_index)

# Initialize topology (shared across requests)
worker_name = f"h{device_index}"
topology = TopologyProvider(
    device_names=[worker_name],
    num_workers=1,
    workers=[],
    num_epochs=1,
    link_latency=20,
    link_loss=5,
    model_name='resnet'
)
#port = int(os.getenv("FLASK_PORT", "5000"))
#ip = "127.0.0.1"  # ✅ Use localhost for local runs
ip = socket.gethostbyname(socket.gethostname())  # get pod IP
# 🛠 Register this single pod as a worker
topology.register_worker(worker_name, ip = ip, port=port)

# -------------------------------
# 2. FLASK ENDPOINTS
# -------------------------------
@app.route("/train", methods=["POST"])
def train():
#    if not trace.is_available():
#        trace.advance()  # ⬅️ Advance even if unavailable
#        return "Device unavailable", 503

    try:
        train_start = time.perf_counter()
        proc_start = snapshot_process()
        # Load posted weights
        raw_weights = request.files["weights"].read()
        state_dict = torch.load(io.BytesIO(raw_weights), map_location="cpu")


        # Parse sync_only flag
#        sync_only = request.form.get("sync_only", "False") == "True"
        logical_id = request.form.get("logical_id")

#        if sync_only:
#            print("📥 Received weights (sync_only=True), skipping training")
#            updated_weights = state_dict  # just return the received weights
#        else:
        # Perform training
        updated_weights = topology.run_local_training(
                global_weights=state_dict,
                local_epochs=3,
                logical_id=logical_id
            )
        trace.advance()  # ⬅️ Move to next trace after training
        print("✅ Training completed. Advanced trace.")
        proc_end = snapshot_process()
        cpu_pct, rss_kb, io_read_delta, io_write_delta = summarize_process(proc_start, proc_end)
        print(
            "🧮 Client process usage: CPU~{:.1f}% RSS~{:.1f}MB IO(read/write)~{}/{} bytes TrainTime~{:.2f}s"
            .format(cpu_pct, rss_kb / 1024.0, io_read_delta, io_write_delta, time.perf_counter() - train_start)
        )

        
        # Return updated weights
        buffer = io.BytesIO()
        torch.save(updated_weights, buffer)
        buffer.seek(0)
        return buffer.read(), 200

    except Exception as e:
        print("🔥 Error during training:", e)
        trace.advance()  # ⬅️ Advance even if training failed
        return str(e), 500


# Optional health check
@app.route("/health", methods=["GET"])
def health():
    return "OK", 200


# -------------------------------
# 3. APP ENTRY POINT
# -------------------------------
#port = int(os.getenv("FLASK_PORT", "5000"))
#main_server_url = os.getenv("MAIN_SERVER_URL", "http://main-server-service:8080")  # service name in K8s
#main_server_url = os.getenv("MAIN_SERVER_URL", "http://localhost:8080")  # service name in K8s

def register_with_main_server():
    try:
        ip = socket.gethostbyname(socket.gethostname())  # get pod IP
        payload = {
            "device_id": device_id_str,
            "ip": ip,
            "port": port
        }
        requests.post(f"{main_server_url}/register", json=payload)
        print(f"✅ Registered with main server: {payload}")
    except Exception as e:
        print(f"❌ Failed to register with main server: {e}")


def send_status_update():
    try:
#       server_ip = socket.gethostbyname("main_server_service")
       server_ip= args.server_ip
       print("server_ip is:", args.server_ip)
       latency, packet_loss, initial_availability = topology.measure_latency_and_loss(server_ip)
       print("latency: from client", latency)
       requests.post(
          f"{main_server_url}/status_update",
          json={
          "device_id": device_id_str,
          "latency": latency,
          "packet_loss": packet_loss,
          "timestamp": time.time(),
          "availability": initial_availability
       })
    except Exception as e:
       print(f"❌ Failed to status update with the main server: {e}")


def periodic_status_update(interval=10, max_updates=5):
     while True:
#    for _ in range(max_updates):
        send_status_update()
        time.sleep(interval)

def wait_for_topology_ready(delay=2):
    while True:
        try:
            r = requests.get(f"{main_server_url}/ready", timeout=2)
            if r.status_code == 200 and r.text.strip() == "ready":
                return True
        except:
            pass
        print("Waiting for server topology to initialize...")
        time.sleep(delay)

if __name__ == "__main__":
    print("📟 Starting client")
    #print("📟 Starting client {device_id_str} (device index {})".format(device_index))
    #print(f"📟 Client {device_id_str} is listening on port {port}")

    # 1. Register with main server (blocking call)
    register_with_main_server()

    # 2. Wait until server topology is ready
    wait_for_topology_ready()

    # 3. Start periodic status updates in background
    threading.Thread(target=periodic_status_update, daemon=True).start()

    # 4. Start Flask server (this blocks)
    app.run(host="0.0.0.0", port=port)
