# AWPSP for Fair Federated Learning

This repository implements a federated learning simulator with a centralized server coordinator and multiple client workers. The system is designed to study fairness under intermittent client participation using cumulative utility parity.

The server exposes a Flask API for client registration, coordination, and aggregation. Each client runs a lightweight Flask service that registers with the server, periodically reports availability and latency, receives global model parameters, performs local training, and returns updated weights.

The server and clients are intended to run in separate Docker containers (using distinct Dockerfiles). Both sides load CIFAR-10 locally and construct non-IID client-specific subsets, so data is logically partitioned per client rather than globally shared.

---

## System Architecture

- **Server**: Coordinates training rounds, performs client selection, aggregates updates, evaluates models, and logs fairness metrics.
- **Clients**: Register with the server, maintain local datasets, execute local training, and return updates.
- **Communication**: Implemented via REST endpoints using Flask.
- **Deployment**: Server and clients run in isolated Docker containers.

---

## How to Run

### 1. Build Docker Images

Build the server and client images from their respective directories:

```bash
docker build -t cpp-server ./server
docker build -t cpp-client ./client
````

These images run `main_server.py` and `main_client.py`, respectively.

---

### 2. Start the Server

Launch the server container:

```bash
docker run --rm -p 8080:8080 --name cpp-server cpp-server
```

The server listens on port `8080` and waits for clients to register before starting training.

---

### 3. Start Multiple Clients

Each client must be assigned a unique device ID and port and must be configured with the server’s IP address.

Example (run multiple times with different IDs and ports):

```bash
docker run --rm --network host cpp-client \
  --device_id h0 --port 5000 --server_ip <SERVER_IP>

docker run --rm --network host cpp-client \
  --device_id h1 --port 5001 --server_ip <SERVER_IP>
```

Clients register via `/register`, wait for topology initialization, send periodic status updates, and respond to `/train` requests with locally trained model updates.

---

## Notes

* Non-IIDness is induced by assigning each client a fixed subset of CIFAR-10 labels.
* Availability traces and correlated dropout patterns can be configured in the client code.
* All fairness and performance metrics are logged in CSV format on the server side.

---

## License

This project is provided for research and academic use.

```

