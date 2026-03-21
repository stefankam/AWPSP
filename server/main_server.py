# main_server.py

from flask import Flask, request
import torch
import io
import os
import csv
import time
import copy
import math
import random
import json
import numpy as np
from typing import Dict, Tuple
from types import SimpleNamespace
from collections import defaultdict
from torchvision import models
from shared_state import topology
import threading
import requests
import socket
from topology_server import TopologyProvider
import shared_state
from availability import extract_availability_vectors
from oort.oort import create_training_selector


def read_proc_stat() -> Tuple[int, int]:
    with open("/proc/stat", "r") as f:
        cpu_line = f.readline().strip().split()
    values = list(map(int, cpu_line[1:]))
    idle = values[3] + values[4]  # idle + iowait
    total = sum(values)
    return total, idle


def read_meminfo() -> Dict[str, int]:
    info = {}
    with open("/proc/meminfo", "r") as f:
        for line in f:
            key, value = line.split(":", 1)
            info[key.strip()] = int(value.strip().split()[0])
    return info


def read_diskstats() -> Dict[str, int]:
    stats = {}
    with open("/proc/diskstats", "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 14:
                continue
            name = parts[2]
            if name.startswith("loop") or name.startswith("ram"):
                continue
            reads = int(parts[3])
            writes = int(parts[7])
            sectors_read = int(parts[5])
            sectors_written = int(parts[9])
            stats[name] = reads + writes + sectors_read + sectors_written
    return stats


def snapshot_system():
    total, idle = read_proc_stat()
    mem = read_meminfo()
    disk = read_diskstats()
    return total, idle, mem, disk


def summarize_system(start, end):
    total0, idle0, mem0, disk0 = start
    total1, idle1, mem1, disk1 = end
    cpu_delta = total1 - total0
    idle_delta = idle1 - idle0
    cpu_pct = 0.0 if cpu_delta == 0 else (1.0 - idle_delta / cpu_delta) * 100
    mem_used_kb = mem1.get("MemTotal", 0) - mem1.get("MemAvailable", 0)
    disk_delta = sum(disk1.values()) - sum(disk0.values())
    return cpu_pct, mem_used_kb, disk_delta


app = Flask(__name__)
current_round = 0

# Global device registry
device_registry = {}
REGISTERED_CLIENTS_CACHE = os.getenv("REGISTERED_CLIENTS_CACHE", "registered_clients.json")
#topology = None  # ← define globally so update_status() can access it
#current_round = 0  # make current_round global to

# -------------------------------
# 1. REGISTRATION ENDPOINT
# -------------------------------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()

    # 🧠 Use actual sender IP, not what the client claims
#    sender_ip = request.remote_addr

    device_registry[data["device_id"]] = {
        "ip": data["ip"],
        "port": data["port"]
    }

    # Persist registrations so experiment sweeps can reuse the same physical clients
    # across subprocess runs without waiting for fresh re-registration.
    try:
        with open(REGISTERED_CLIENTS_CACHE, "w") as f:
            json.dump(device_registry, f)
    except Exception as e:
        print(f"⚠️ Could not persist registered clients cache: {e}")


    print(f"📥 Registered {data['device_id']} at {data['ip']}:{data['port']}")

    return "OK", 200

# Distributed Hash Table
class DHT:
    def __init__(self, size=100):
        self.table = {}
        self.size = size

@app.route("/ready", methods=["GET"])
def ready():
    if shared_state.topology:
        return "ready", 200
    return "not_ready", 503


def initialize_topology(device_file="devices.txt", num_clients=None):
    if num_clients is None:
        num_clients = int(os.getenv("REGISTERED_CLIENT_COUNT", os.getenv("PHYSICAL_CONTAINER_LIMIT", "2")))

    reuse_registered_clients = os.getenv("REUSE_REGISTERED_CLIENTS", "1").lower() in ("1", "true", "yes", "on")
    if reuse_registered_clients and len(device_registry) < num_clients and os.path.exists(REGISTERED_CLIENTS_CACHE):
        try:
            with open(REGISTERED_CLIENTS_CACHE, "r") as f:
                cached = json.load(f)
            if isinstance(cached, dict):
                for k, v in cached.items():
                    if isinstance(v, dict) and "ip" in v and "port" in v:
                        device_registry[k] = v
                print(f"♻️ Loaded {len(device_registry)} cached registered clients from {REGISTERED_CLIENTS_CACHE}")
        except Exception as e:
            print(f"⚠️ Could not load registered clients cache: {e}")

    while len(device_registry) < num_clients:
        print(f"🕒 Registered devices: {len(device_registry)} / {num_clients}")
        time.sleep(2)

    print("✅ All clients registered. Initializing topology.")

#    print("📄 Loading devices from file...")

#    device_registry = {}
#    with open(device_file, "r") as f:
#        for i, line in enumerate(f):
#            if i >= num_clients:
#                break
#            if not line.strip():
#                continue
#            device_id, ip, port = line.strip().split()
#            device_registry[device_id] = {
#                "ip": ip,
#                "port": int(port)
#            }

#    print(f"✅ Loaded {len(device_registry)} devices.")

    device_ids = list(device_registry.keys())

    shared_state.topology = TopologyProvider(
        device_names=device_ids,
        num_epochs=1,
        link_latency=20, 
        link_loss=5,
        model_name='resnet',
        device_registry=device_registry 
    )
    shared_state.topology.dht = DHT(size=100)  # Initialize the DHT
    for device_id in device_ids:
        shared_state.topology.dht.table[device_id] = {
          "latency": None,
          "packet_loss": None,
          "last_seen": None,
          "availability": None,
          "freshness": None,
          "correlation": None
        }
    print("✅ Topology initialized.")
    wait_for_latency_data(num_clients=num_clients)


@app.route("/status_update", methods=["POST"])
def update_status():
    global current_round
    data = request.get_json()
    node = data["device_id"]

    if shared_state.topology is None:
        print("⚠️ Topology not initialized yet. Ignoring status update.")
        return "ERROR: Topology not initialized", 503

    if node not in shared_state.topology.dht.table:
        print(f"⚠️ Node {node} not found in DHT.")
        return "ERROR: Node not found", 404

    shared_state.topology.dht.table[node]["latency"] = data["latency"]
    shared_state.topology.dht.table[node]["packet_loss"] = data["packet_loss"]
    shared_state.topology.dht.table[node]["last_seen"] = time.time()
    shared_state.topology.dht.table[node]["availability"] = data["availability"]
    shared_state.topology.dht.table[node]["freshness"] = shared_state.topology.get_freshness(node, current_round)
    shared_state.topology.dht.table[node]["correlation"] = shared_state.topology.failure_correlation.get(node, {})
    print(f"📶 Updated status for {node}: latency={data['latency']}, loss={data['packet_loss']}")


    return "OK", 200

# -------------------------------
# 2. HEALTH CHECK (optional)
# -------------------------------
@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

# -------------------------------
# 3. FEDERATED COORDINATOR
# -------------------------------

import torch.nn as nn
from torchvision import models, datasets

def init_resnet(train_last_n_blocks=1):
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)

    # Freeze everything first
    for param in base_model.parameters():
        param.requires_grad = False

    # Unfreeze last N blocks + FC
    if train_last_n_blocks >= 1:
        for param in base_model.layer4.parameters():
            param.requires_grad = True
    if train_last_n_blocks >= 2:
        for param in base_model.layer3.parameters():
            param.requires_grad = True
    if train_last_n_blocks >= 3:
        for param in base_model.layer2.parameters():
            param.requires_grad = True

    # Always unfreeze fc
    for param in base_model.fc.parameters():
        param.requires_grad = True

    return base_model







def _env_int(name, default):
    return int(os.getenv(name, str(default)))


def _env_float(name, default):
    return float(os.getenv(name, str(default)))


def _env_bool(name, default):
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes", "on")


def build_logical_label_map(logical_client_count, labels_per_client, split_mode="extreme", dirichlet_alpha=0.5, seed=0):
    rng = np.random.default_rng(seed)
    split_mode = (split_mode or "extreme").lower()
    label_map = {}

    if split_mode == "dirichlet":
        for i in range(logical_client_count):
            probs = rng.dirichlet(np.full(10, max(dirichlet_alpha, 1e-3)))
            labels = list(np.argsort(probs)[-labels_per_client:])
            label_map[f"h{i}"] = sorted(int(x) for x in labels)
        return label_map

    overlap_shift = 1 if split_mode == "overlap" else labels_per_client
    for i in range(logical_client_count):
        start = (i * overlap_shift) % 10
        labels = [((start + j) % 10) for j in range(labels_per_client)]
        label_map[f"h{i}"] = labels
    return label_map


def apply_correlation_noise(correlated_failures, node_pool, noise_pct=0.0, seed=0):
    if noise_pct <= 0 or not node_pool:
        return correlated_failures
    rng = random.Random(seed)
    noisy = list(correlated_failures)
    k = max(1, int(len(noisy) * noise_pct / 100.0)) if noisy else 0
    for _ in range(k):
        if noisy and rng.random() < 0.5:
            noisy.pop(rng.randrange(len(noisy)))
        a = rng.choice(node_pool)
        b = rng.choice(node_pool)
        if a != b:
            noisy.append((a, b))
    return noisy



def build_oort_args():
    return SimpleNamespace(
        exploration_factor=float(os.getenv("OORT_EXPLORATION_FACTOR", "0.9")),
        exploration_decay=float(os.getenv("OORT_EXPLORATION_DECAY", "0.98")),
        exploration_min=float(os.getenv("OORT_EXPLORATION_MIN", "0.1")),
        exploration_alpha=float(os.getenv("OORT_EXPLORATION_ALPHA", "0.3")),
        round_threshold=float(os.getenv("OORT_ROUND_THRESHOLD", "50")),
        sample_window=float(os.getenv("OORT_SAMPLE_WINDOW", "5.0")),
        pacer_step=int(os.getenv("OORT_PACER_STEP", "20")),
        pacer_delta=float(os.getenv("OORT_PACER_DELTA", "5")),
        blacklist_rounds=int(os.getenv("OORT_BLACKLIST_ROUNDS", "-1")),
        blacklist_max_len=float(os.getenv("OORT_BLACKLIST_MAX_LEN", "0.5")),
        clip_bound=float(os.getenv("OORT_CLIP_BOUND", "0.95")),
        round_penalty=float(os.getenv("OORT_ROUND_PENALTY", "2.0")),
        cut_off_util=float(os.getenv("OORT_CUTOFF_UTIL", "0.95")),
    )



def run_federated_training():
    global current_round
    if shared_state.topology is None:
        raise RuntimeError("❌ Topology has not been initialized.")


    seed = _env_int("EXPERIMENT_SEED", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    logical_client_count = _env_int("LOGICAL_CLIENT_COUNT", 100)
    physical_container_limit = _env_int("PHYSICAL_CONTAINER_LIMIT", 10)
    logical_selected_per_round = _env_int("LOGICAL_SELECTED_PER_ROUND", 10)
    use_logical_scheduling = _env_bool("USE_LOGICAL_SCHEDULING", True)
    logical_labels_per_client = _env_int("LOGICAL_LABELS_PER_CLIENT", 2)
    split_mode = os.getenv("LOGICAL_SPLIT_MODE", "extreme")
    dirichlet_alpha = _env_float("DIRICHLET_ALPHA", 0.5)
    selector_mode = os.getenv("SELECTOR_MODE", "awpsp").lower()
    corr_noise_pct = _env_float("CORRELATION_NOISE_PCT", 0.0)

    experiment_context = {
        "logical_population": logical_client_count,
        "selected_per_round": logical_selected_per_round,
        "physical_clients": physical_container_limit,
        "split_mode": split_mode,
        "dirichlet_alpha": dirichlet_alpha,
        "selector_mode": selector_mode,
        "correlation_noise_pct": corr_noise_pct,
        "seed": seed,
    }
    
    metrics_log_path = os.getenv("METRICS_LOG_PATH", "metrics_log.csv")
    final_metrics_path = os.getenv("FINAL_METRICS_PATH", "final_metrics.csv")
    logical_participation_awpsp = defaultdict(int)
    logical_participation_psp = defaultdict(int)
    logical_participation_oort = defaultdict(int)
    logical_participation_log = defaultdict(int)
    # Oort-style selector state (utility/reward + exploration + duration penalty).
    oort_pull_count = defaultdict(int)
    oort_utility_ema = defaultdict(float)
    oort_duration_ema = defaultdict(lambda: 1.0)
    logical_loss_history_awpsp = defaultdict(list)
    logical_loss_history_psp = defaultdict(list)
    logical_loss_history_oort = defaultdict(list)

    label_map = shared_state.topology.label_map

    # Load availability vectors
    availability_vectors = extract_availability_vectors("traces/traces.txt")

    awpsp_accuracy_log = []
    psp_accuracy_log = [] 
    oort_accuracy_log = []
    awpsp_instant_fairness_log = []
    psp_instant_fairness_log = []
    oort_instant_fairness_log = []
    awpsp_cumul_fairness_log = []
    psp_cumul_fairness_log = []  
    oort_cumul_fairness_log = []
    corr_failure_log = []
    awpsp_covered_labels_log = []
    psp_covered_labels_log = []
    oort_covered_labels_log = []
    selected_awpsp_log = []
    selected_psp_log = []
    selected_oort_log = []
    awpsp_avg_score_log = []
    psp_avg_score_log = []
    oort_avg_score_log = []
    awpsp_labels_log =[]
    psp_labels_log =[]
    oort_labels_log =[]
    awpsp_KL_log =[]
    psp_KL_log =[]
    oort_KL_log =[]
    awpsp_unseen_log =[]
    psp_unseen_log =[]
    oort_unseen_log =[]
    awpsp_gini_log =[]
    psp_gini_log =[]
    oort_gini_log =[]
    accuracy_log = []
    var_u_log = []
    surrogate_log = []
    awpsp_avg_within_class_log = []
    psp_avg_within_class_log = []
    oort_avg_within_class_log = []
    awpsp_fairness_inter_class_log = []
    psp_fairness_inter_class_log = []
    oort_fairness_inter_class_log = []

    # ---------------- Initialize models ----------------
#    base_model = models.resnet18(weights=None)
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    base_model.fc = torch.nn.Linear(base_model.fc.in_features, 10)
#    base_model = init_resnet(train_last_n_blocks=2)  # train layer3 + layer4 + fc

    # Two independent weight states
    current_weights_fair = base_model.state_dict()
    current_weights_awpsp = copy.deepcopy(current_weights_fair)
    current_weights_psp = copy.deepcopy(current_weights_fair)
    current_weights_oort = copy.deepcopy(current_weights_fair)


    logical_label_map = build_logical_label_map(
        logical_client_count,
        logical_labels_per_client,
        split_mode=split_mode,
        dirichlet_alpha=dirichlet_alpha,
        seed=seed,
    )



    logical_oort_selector = create_training_selector(build_oort_args())
    logical_oort_selector_id_to_client = {}
    logical_oort_client_to_selector_id = {}
    for idx in range(logical_client_count):
        logical_id = f"h{idx}"
        selector_id = str(idx)
        logical_oort_selector_id_to_client[selector_id] = logical_id
        logical_oort_client_to_selector_id[logical_id] = selector_id
        logical_oort_selector.register_client(selector_id, {"reward": 0.0, "duration": 1.0, "status": True})


    def compute_logical_label_metrics(selected_ids):
        total_labels = 10
        label_counts = [0] * total_labels
        covered_labels = set()
        for logical_id in selected_ids:
            labels = logical_label_map.get(logical_id, [])
            for label in labels:
                label_counts[label] += 1
                covered_labels.add(label)
        total = sum(label_counts)
        if total > 0:
            p = [count / total for count in label_counts]
            u = [1 / total_labels] * total_labels
            eps = 1e-12
            kl = sum(pi * (math.log(pi + eps) - math.log(ui + eps)) for pi, ui in zip(p, u) if pi > 0)
            unseen = sum(1 for c in label_counts if c == 0) / total_labels
        else:
            kl = 0.0
            unseen = 0.0
        mean_count = total / total_labels if total_labels else 0.0
        variance = (
            sum((c - mean_count) ** 2 for c in label_counts) / total_labels
            if total_labels else 0.0
        )
        return {
            "covered_labels": sorted(covered_labels),
            "covered_count": len(covered_labels),
            "kl": kl,
            "unseen": unseen,
            "variance": variance,
        }



    def compute_participation_stats(participation_log, all_ids):
        counts = [participation_log.get(cid, 0) for cid in all_ids]
        total_counts = sum(counts)
        if not counts:
            return 0.0, 0.0
        mean_count = total_counts / len(counts)
        variance = sum((c - mean_count) ** 2 for c in counts) / len(counts)
        if total_counts == 0:
            return variance, 0.0
        diffs = 0.0
        for i in counts:
            for j in counts:
                diffs += abs(i - j)
        gini = diffs / (2 * len(counts) * total_counts)
        return variance, gini



    def compute_logical_model_loss_fairness(model, selected_ids, history_map, all_ids):
        eval_dataset = datasets.CIFAR10(
            root='data/',
            train=False,
            download=False,
            transform=shared_state.topology.transform,
        )
        targets = np.array(eval_dataset.targets)
        criterion = torch.nn.CrossEntropyLoss(reduction="none")

        model.eval()
        current_losses = {}
        class_losses = defaultdict(list)
        with torch.no_grad():
            for logical_id in selected_ids:
                labels = logical_label_map.get(logical_id, [])
                if not labels:
                    continue
                indices = np.where(np.isin(targets, labels))[0].tolist()
                if not indices:
                    continue
                subset = torch.utils.data.Subset(eval_dataset, indices)
                loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=False)
                losses = []
                for images, y in loader:
                    outputs = model(images)
                    batch_losses = criterion(outputs, y.long())
                    losses.extend(batch_losses.tolist())
                    for lbl, loss in zip(y.tolist(), batch_losses.tolist()):
                        class_losses[int(lbl)].append(float(loss))
                if losses:
                    avg_loss = float(sum(losses) / len(losses))
                    current_losses[logical_id] = avg_loss
                    history_map[logical_id].append(avg_loss)

        # Match topology-side fairness computation semantics.
        # 1) Per-class sample variance (ddof=1) when class has >1 samples, else 0.0
        # 2) Inter-class fairness computed on class means with a minimum sample threshold
        #    to avoid noisy single-sample classes.
        MIN_SAMPLES_PER_CLASS = 2

        per_class_means = {}
        per_class_vars = {}
        per_class_counts = {}
        for lbl, losses in class_losses.items():
            cnt = len(losses)
            per_class_counts[int(lbl)] = cnt
            if cnt == 0:
                continue
            per_class_means[int(lbl)] = float(np.mean(losses))
            if cnt > 1:
                per_class_vars[int(lbl)] = float(np.var(losses, ddof=1))
            else:
                per_class_vars[int(lbl)] = 0.0

        valid_class_means = [
            mean
            for lbl, mean in per_class_means.items()
            if per_class_counts.get(lbl, 0) >= MIN_SAMPLES_PER_CLASS
        ]

        avg_within_class_var = float(np.mean(list(per_class_vars.values()))) if per_class_vars else 0.0
        fairness_inter_class = float(np.var(valid_class_means, ddof=0)) if valid_class_means else 0.0

        return avg_within_class_var, fairness_inter_class


    def compute_logical_oort_utilities(model, selected_ids):
        """
        Oort-style statistical utility proxy per logical client:
        utility_k ~= sqrt(mean(loss^2)) on the logical client's label-conditioned slice.
        """
        eval_dataset = datasets.CIFAR10(
            root='data/',
            train=False,
            download=False,
            transform=shared_state.topology.transform,
        )
        targets = np.array(eval_dataset.targets)
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        model.eval()
        utility_map = {}

        with torch.no_grad():
            for logical_id in selected_ids:
                labels = logical_label_map.get(logical_id, [])
                if not labels:
                    continue
                indices = np.where(np.isin(targets, labels))[0].tolist()
                if not indices:
                    continue

                subset = torch.utils.data.Subset(eval_dataset, indices)
                loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=False)
                sq_losses = []

                for images, y in loader:
                    outputs = model(images)
                    losses = criterion(outputs, y.long())
                    sq_losses.extend((losses ** 2).tolist())

                if sq_losses:
                    utility_map[logical_id] = float(math.sqrt(sum(sq_losses) / len(sq_losses)))

        return utility_map




    def compute_final_metrics(model=None, round_index=None):
        if model is None:
            model = base_model
        if round_index is None:
            round_index = current_round
        nodes = list(shared_state.topology.dht.table.keys())
        print("nodes before calculating evaluate_per_client_accuracy",nodes)
        per_client_acc = shared_state.topology.evaluate_per_client_accuracy(model, nodes)
        acc_values = [val for val in per_client_acc.values() if val is not None]
        if not acc_values:
            return None

        avg_acc = sum(acc_values) / len(acc_values)
        acc_variance = sum((val - avg_acc) ** 2 for val in acc_values) / len(acc_values)
        acc_squared_sum = sum(val ** 2 for val in acc_values)
        jain_acc = (sum(acc_values) ** 2) / (len(acc_values) * acc_squared_sum) if acc_squared_sum else 0.0

        u_tilde_values = []
        u_tilde_with_surrogate = []
        for node in nodes:
            if shared_state.topology.total_rounds_elapsed > 0:
                pi_k = shared_state.topology.availability_counts[node] / shared_state.topology.total_rounds_elapsed
                if pi_k > 0:
                    u_k = shared_state.topology.utility_log[node]
                    u_tilde_values.append(u_k / pi_k)
                    surrogate_k = shared_state.topology.surrogate_contributions.get(node, 0.0)
                    u_tilde_with_surrogate.append((u_k + surrogate_k) / pi_k)

        def compute_utility_metrics(values):
            if not values:
                return None, None
            mean_u = sum(values) / len(values)
            std_u = math.sqrt(sum((val - mean_u) ** 2 for val in values) / len(values))
            utility_cv = (std_u / mean_u) if mean_u != 0 else 0.0
            squared_sum = sum(val ** 2 for val in values)
            jain_utility = (sum(values) ** 2) / (len(values) * squared_sum) if squared_sum else 0.0
            return utility_cv, jain_utility

        if use_logical_scheduling:
            selected_counts = [logical_participation_log[node] for node in logical_client_ids]
        else:
            selected_counts = [len(shared_state.topology.participation_log.get(node, [])) for node in nodes]
        sel_gap = max(selected_counts) - min(selected_counts) if selected_counts else 0.0
        if selected_counts and sum(selected_counts) > 0:
            diffs = 0.0
            for i in selected_counts:
                for j in selected_counts:
                    diffs += abs(i - j)
            gini = diffs / (2 * len(selected_counts) * sum(selected_counts))
        else:
            gini = 0.0

        utility_cv_no, jain_utility_no = compute_utility_metrics(u_tilde_values)
        utility_cv_with, jain_utility_with = compute_utility_metrics(u_tilde_with_surrogate)

        return {
            "Round": round_index + 1,
            "logical_population": experiment_context["logical_population"],
            "selected_per_round": experiment_context["selected_per_round"],
            "physical_clients": experiment_context["physical_clients"],
            "split_mode": experiment_context["split_mode"],
            "dirichlet_alpha": experiment_context["dirichlet_alpha"],
            "selector_mode": experiment_context["selector_mode"],
            "correlation_noise_pct": experiment_context["correlation_noise_pct"],
            "seed": experiment_context["seed"],
            "Avg Acc (No Surrogate)": avg_acc,
            "Jain (Acc) (No Surrogate)": jain_acc,
            "Utility CV (No Surrogate)": utility_cv_no,
            "Jain (Utility) (No Surrogate)": jain_utility_no,
            "Sel. Gap (No Surrogate)": sel_gap,
            "Gini (No Surrogate)": gini,
            "Avg Acc (With Surrogate)": avg_acc,
            "Jain (Acc) (With Surrogate)": jain_acc,
            "Utility CV (With Surrogate)": utility_cv_with,
            "Jain (Utility) (With Surrogate)": jain_utility_with,
            "Sel. Gap (With Surrogate)": sel_gap,
            "Gini (With Surrogate)": gini,
            "Acc Variance": acc_variance,
        }


    num_rounds = _env_int("NUM_ROUNDS", 50)

    for current_round in range(num_rounds):
        print(f"\n🌐 Federated Round {current_round + 1}")
        round_start = time.perf_counter()
        sys_start = snapshot_system()


        # Detect correlated failures
        correlated_failures = shared_state.topology.get_correlated_failure(
            current_round, availability_vectors, corr_threshold=0.35, num_neighbors=4
        )


        correlated_failures = apply_correlation_noise(
            correlated_failures,
            list(shared_state.topology.dht.table.keys()),
            noise_pct=corr_noise_pct,
            seed=seed + current_round,
        )

        logical_client_ids = [f"h{i}" for i in range(logical_client_count)]
        physical_ids = list(device_registry.keys())[:physical_container_limit]
        if use_logical_scheduling:
            # Reuse AW-PSP physical prioritization, driven by AW-PSP model state.
            awpsp_priority_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            awpsp_priority_model.fc = torch.nn.Linear(awpsp_priority_model.fc.in_features, 10)
            awpsp_priority_model.load_state_dict(current_weights_awpsp)

            priority_nodes, *_ = shared_state.topology.prioritize_available_nodes(
                awpsp_priority_model,
                current_round,
                correlated_failures,
                num_clients=len(physical_ids),
                label_map=label_map,
            )
            prioritized = [n for n in priority_nodes if n in physical_ids]
            physical_ids = prioritized + [n for n in physical_ids if n not in prioritized]

            logical_selected_awpsp, *_ = shared_state.topology.prioritize_available_nodes(
                awpsp_priority_model,
                current_round,
                correlated_failures,
                num_clients=logical_selected_per_round,
                label_map=logical_label_map,
                candidate_ids=logical_client_ids,
            )

            feasible_logical_selector_ids = set()
            for logical_id in logical_client_ids:
                selector_id = logical_oort_client_to_selector_id[logical_id]
                logical_idx = int(str(logical_id).replace("h", "")) if str(logical_id).startswith("h") else 0
                mapped_latency = 1.0
                if physical_ids:
                    mapped_pid = physical_ids[logical_idx % len(physical_ids)]
                    mapped_latency = shared_state.topology.dht.table.get(mapped_pid, {}).get("latency", 1.0) or 1.0
                logical_oort_selector.update_duration(selector_id, float(mapped_latency))
                feasible_logical_selector_ids.add(int(selector_id))

            selected_oort_selector_ids = logical_oort_selector.select_participant(
                min(logical_selected_per_round, len(feasible_logical_selector_ids)),
                feasible_clients=feasible_logical_selector_ids,
            ) if feasible_logical_selector_ids else []
            logical_selected_oort = [
                logical_oort_selector_id_to_client[sid]
                for sid in selected_oort_selector_ids
                if sid in logical_oort_selector_id_to_client
            ]

            # PSP logical baseline
            logical_selected_psp = random.sample(
                logical_client_ids,
                min(logical_selected_per_round, len(logical_client_ids)),
            )

            logical_selected_fair = logical_selected_awpsp
            logical_metrics_awpsp = compute_logical_label_metrics(logical_selected_awpsp)
            logical_metrics_psp = compute_logical_label_metrics(logical_selected_psp)
            logical_metrics_oort = compute_logical_label_metrics(logical_selected_oort)

        num_corr_failed = sum(1 for _, failed_neighbors in correlated_failures if failed_neighbors)
        corr_failure_log.append((current_round, num_corr_failed))
        print(f"🌩️ Correlated failure count: {num_corr_failed}")


        if use_logical_scheduling:
            selected = logical_selected_fair
            # Keep metrics lightweight in logical mode without changing core AWPSP/PSP implementation
            var_u = logical_metrics_awpsp["variance"]
            total_bias_bound = 0.0
        else:
            selected, var_u, total_bias_bound = shared_state.topology.select_fair_nodes(
                base_model,
                current_round,
                correlated_failures,
                num_clients=5,
                corr_threshold=0.35,
                label_map=label_map,
                lambda_=0.5,
                epsilon=1e-5,
            )
        selection_end = time.perf_counter()
        if use_logical_scheduling:
            weights_fair = shared_state.topology.run_logical_federated_round(
                selected, physical_ids, current_weights_fair
            )
        else:
            weights_fair = shared_state.topology.run_federated_round(selected, current_weights_fair, base_model)
        if weights_fair is not None:
           base_model.load_state_dict(weights_fair)
           current_weights_awpsp = weights_fair
           accuracy = shared_state.topology.evaluate_global_model(base_model, selected_nodes=selected, use_selected_nodes=True, physical_ids=physical_ids if use_logical_scheduling else None)
           accuracy_log.append((current_round, accuracy))
           var_u_log.append((current_round, var_u))
           surrogate_log.append((current_round, total_bias_bound))
           print(f"🔁 Round {current_round + 1}: Fair-Select Acc = {accuracy:.2f}%")
        else:
           print("⚠️ No updates received from clients. Skipping model update this round.")
        fair_end = time.perf_counter()


        # ---------------- AW-PSP branch ----------------
        # Node selection based on AW-PSP

#        awpsp_model = models.resnet18(weights=None)
        awpsp_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        awpsp_model.fc = torch.nn.Linear(awpsp_model.fc.in_features, 10)
        awpsp_model.load_state_dict(current_weights_awpsp)

        if use_logical_scheduling:
            selected_awpsp = logical_selected_awpsp
            for logical_id in selected_awpsp:
                logical_participation_awpsp[logical_id] += 1
            awpsp_instant_var = logical_metrics_awpsp["variance"]
            awpsp_cumul_var, awpsp_gini = compute_participation_stats(logical_participation_awpsp, logical_client_ids)

            avg_within_class_var, fairness_inter_class = compute_logical_model_loss_fairness(
                awpsp_model,
                selected_awpsp,
                logical_loss_history_awpsp,
                logical_client_ids,
            )

            awpsp_covered_labels = logical_metrics_awpsp["covered_labels"]
            awpsp_avg_score = float(len(selected_awpsp)) / float(max(1, logical_client_count))
            awpsp_labels = logical_metrics_awpsp["covered_labels"]
            awpsp_KL = logical_metrics_awpsp["kl"]
            awpsp_unseen = logical_metrics_awpsp["unseen"]
        else:

            selected_awpsp, awpsp_instant_var, awpsp_cumul_var, avg_within_class_var, fairness_inter_class, awpsp_covered_labels, awpsp_avg_score, awpsp_labels, awpsp_KL, awpsp_unseen, awpsp_gini = \
                shared_state.topology.prioritize_available_nodes(
                    awpsp_model, current_round, correlated_failures, num_clients=5, label_map=label_map
                )
        awpsp_covered_labels_log.append((current_round, len(awpsp_covered_labels)))
        selected_awpsp_log.append((current_round, selected_awpsp))

        if use_logical_scheduling:
            weights_awpsp = shared_state.topology.run_logical_federated_round(
                selected_awpsp, physical_ids, current_weights_awpsp
            )
        else:
            weights_awpsp = shared_state.topology.run_federated_round(selected_awpsp, current_weights_awpsp, awpsp_model)

        if weights_awpsp is not None:
           current_weights_awpsp = weights_awpsp
           awpsp_model.load_state_dict(current_weights_awpsp)
           accuracy_awpsp = shared_state.topology.evaluate_global_model(awpsp_model, selected_nodes=selected_awpsp, use_selected_nodes=False, physical_ids=physical_ids if use_logical_scheduling else None)           
           awpsp_accuracy_log.append((current_round, accuracy_awpsp))
           awpsp_instant_fairness_log.append((current_round, awpsp_instant_var))
           awpsp_cumul_fairness_log.append((current_round, awpsp_cumul_var))
           print(f"🔁 Round {current_round + 1}: AW-PSP Acc = {accuracy_awpsp:.2f}%")
           awpsp_avg_score_log.append((current_round, awpsp_avg_score))
           awpsp_labels_log.append((current_round, awpsp_labels))
           awpsp_KL_log.append((current_round, awpsp_KL))
           awpsp_unseen_log.append((current_round, awpsp_unseen))
           awpsp_gini_log.append((current_round, awpsp_gini))
           awpsp_avg_within_class_log.append(avg_within_class_var)
           awpsp_fairness_inter_class_log.append(fairness_inter_class)
        else:
           print("⚠️ No updates received from clients. Skipping model update this round.")
        awpsp_end = time.perf_counter()




        # ---------------- OORT branch ----------------
        oort_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        oort_model.fc = torch.nn.Linear(oort_model.fc.in_features, 10)
        oort_model.load_state_dict(current_weights_oort)

        if use_logical_scheduling:
            selected_oort = logical_selected_oort
            observed_oort_utilities = compute_logical_oort_utilities(oort_model, selected_oort)
            for logical_id in selected_oort:
                logical_idx = int(str(logical_id).replace("h", "")) if str(logical_id).startswith("h") else 0
                if physical_ids:
                    mapped_pid = physical_ids[logical_idx % len(physical_ids)]
                    mapped_latency = shared_state.topology.dht.table.get(mapped_pid, {}).get("latency", 1.0) or 1.0
                else:
                    mapped_latency = 1.0
                beta = 0.8
                observed_utility = float(observed_oort_utilities.get(logical_id, 0.0))
                oort_utility_ema[logical_id] = beta * oort_utility_ema[logical_id] + (1.0 - beta) * observed_utility
                oort_duration_ema[logical_id] = beta * oort_duration_ema[logical_id] + (1.0 - beta) * float(mapped_latency)
                oort_pull_count[logical_id] += 1
                selector_id = logical_oort_client_to_selector_id.get(logical_id)
                if selector_id is not None:
                    logical_oort_selector.update_client_util(selector_id, {
                        "reward": observed_utility,
                        "duration": float(mapped_latency),
                        "time_stamp": current_round + 1,
                        "status": True,
                    })
            for logical_id in selected_oort:
                logical_participation_oort[logical_id] += 1
            avg_within_class_var, fairness_inter_class = compute_logical_model_loss_fairness(
                oort_model, selected_oort, logical_loss_history_oort, logical_client_ids
            )
            oort_instant_var = logical_metrics_oort["variance"]
            oort_cumul_var, oort_gini = compute_participation_stats(logical_participation_oort, logical_client_ids)
            oort_covered_labels = logical_metrics_oort["covered_labels"]
            oort_avg_score = float(len(selected_oort)) / float(max(1, logical_client_count))
            oort_labels = logical_metrics_oort["covered_labels"]
            oort_KL = logical_metrics_oort["kl"]
            oort_unseen = logical_metrics_oort["unseen"]
        else:
            selected_oort, oort_instant_var, oort_cumul_var, avg_within_class_var, fairness_inter_class, oort_covered_labels, oort_avg_score, oort_labels, oort_KL, oort_unseen, oort_gini = shared_state.topology.select_oort_nodes(
                    oort_model, current_round, correlated_failures, num_clients=5, label_map=label_map
                )
        oort_covered_labels_log.append((current_round, len(oort_covered_labels)))
        selected_oort_log.append((current_round, selected_oort))

        if use_logical_scheduling:
            weights_oort = shared_state.topology.run_logical_federated_round(
                selected_oort, physical_ids, current_weights_oort
            )
        else:
            weights_oort = shared_state.topology.run_federated_round(selected_oort, current_weights_oort, oort_model)

        if weights_oort is not None:
           current_weights_oort = weights_oort
           oort_model.load_state_dict(current_weights_oort)
           accuracy_oort = shared_state.topology.evaluate_global_model(oort_model, selected_nodes=selected_oort, use_selected_nodes=False, physical_ids=physical_ids if use_logical_scheduling else None)
           oort_accuracy_log.append((current_round, accuracy_oort))
           oort_instant_fairness_log.append((current_round, oort_instant_var))
           oort_cumul_fairness_log.append((current_round, oort_cumul_var))
           print(f"🔁 Round {current_round + 1}: OORT Acc = {accuracy_oort:.2f}%")
           oort_avg_score_log.append((current_round, oort_avg_score))
           oort_labels_log.append((current_round, oort_labels))
           oort_KL_log.append((current_round, oort_KL))
           oort_unseen_log.append((current_round, oort_unseen))
           oort_gini_log.append((current_round, oort_gini))
           oort_avg_within_class_log.append(avg_within_class_var)
           oort_fairness_inter_class_log.append(fairness_inter_class)
        else:
           print("⚠️ No updates received from clients. Skipping model update this round.")


        # ---------------- PSP branch ----------------
        # Use a fresh model copy so AW-PSP doesn’t pollute PSP results
        #psp_model = models.resnet18(weights=None)
        psp_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        psp_model.fc = torch.nn.Linear(psp_model.fc.in_features, 10)
        psp_model.load_state_dict(current_weights_psp)

        if use_logical_scheduling:
            selected_psp = logical_selected_psp
            for logical_id in selected_psp:
                logical_participation_psp[logical_id] += 1

            psp_instant_var = logical_metrics_psp["variance"]
            psp_cumul_var, psp_gini = compute_participation_stats(logical_participation_psp, logical_client_ids)

            avg_within_class_var, fairness_inter_class = compute_logical_model_loss_fairness(
                psp_model,
                selected_psp,
                logical_loss_history_psp,
                logical_client_ids,
            )
            _, psp_gini = compute_participation_stats(logical_participation_psp, logical_client_ids)
            psp_covered_labels = logical_metrics_psp["covered_labels"]
            psp_avg_score = float(len(selected_psp)) / float(max(1, logical_client_count))
            psp_labels = logical_metrics_psp["covered_labels"]
            psp_KL = logical_metrics_psp["kl"]
            psp_unseen = logical_metrics_psp["unseen"]
        else:
            selected_psp, psp_instant_var, psp_cumul_var, avg_within_class_var, fairness_inter_class, psp_covered_labels, psp_avg_score, psp_labels, psp_KL, psp_unseen, psp_gini = \
                shared_state.topology.psp_random_selection(
                    psp_model, correlated_failures, num_clients=5, label_map=label_map
                )
        psp_covered_labels_log.append((current_round, len(psp_covered_labels)))
        selected_psp_log.append((current_round, selected_psp))

        if use_logical_scheduling:
            weights_psp = shared_state.topology.run_logical_federated_round(
                selected_psp, physical_ids, current_weights_psp
            )
        else:
            weights_psp = shared_state.topology.run_federated_round(selected_psp, current_weights_psp,psp_model)

        if weights_psp is not None:
           current_weights_psp = weights_psp
           psp_model.load_state_dict(current_weights_psp)
           accuracy_psp = shared_state.topology.evaluate_global_model(psp_model, selected_nodes=selected_psp, use_selected_nodes=False, physical_ids=physical_ids  if use_logical_scheduling else None)           
           psp_accuracy_log.append((current_round, accuracy_psp))
           psp_instant_fairness_log.append((current_round, psp_instant_var))
           psp_cumul_fairness_log.append((current_round, psp_cumul_var))
           print(f"🔁 Round {current_round + 1}: PSP Acc = {accuracy_psp:.2f}%")
           psp_avg_score_log.append((current_round, psp_avg_score))
           psp_labels_log.append((current_round, psp_labels))
           psp_KL_log.append((current_round, psp_KL))
           psp_unseen_log.append((current_round, psp_unseen))
           psp_gini_log.append((current_round, psp_gini))
           psp_avg_within_class_log.append(avg_within_class_var)
           psp_fairness_inter_class_log.append(fairness_inter_class)
        else:
           print("⚠️ No updates received from clients. Skipping model update this round.")

        psp_end = time.perf_counter()

        # Save per-round logs to CSV (append one row per round)
        metrics_header = [
            "Round",
            "logical_population",
            "selected_per_round",
            "physical_clients",
            "split_mode",
            "dirichlet_alpha",
            "selector_mode",
            "correlation_noise_pct",
            "seed",
            "Select_Fair_Accuracy",
            "Select_Fair_variance",
            "Select_Fair_Surrogate",
            "AWPSP_Accuracy",
            "AWPSP_instant_fairness",
            "AWPSP_cumul_fairness",
            "AWPSP_avg_within_class",
            "AWPSP_fairness_inter_class",
            "CorrelatedFailureCount",
            "AWPSP_CoveredLabelsCount",
            "PSP_Accuracy",
            "PSP_instant_fairness",
            "PSP_cumul_fairness",
            "PSP_avg_within_class",
            "PSP_fairness_inter_class",
            "PSP_CoveredLAbelsCount",
            "OORT_Accuracy",
            "OORT_instant_fairness",
            "OORT_cumul_fairness",
            "OORT_avg_within_class",
            "OORT_fairness_inter_class",
            "OORT_CoveredLabelsCount",
            "selected_awpsp",
            "selected_psp",
            "selected_oort",
            "AWPSP Avg Score",
            "PSP Avg Score",
            "OORT Avg Score",
            "AWPSP labels",
            "PSP labels",
            "OORT labels",
            "AWPSP KL",
            "PSP KL",
            "OORT KL",
            "AWPSP unseen",
            "PSP unseen",
            "OORT unseen",
            "AWPSP gini",
            "PSP gini",
            "OORT gini",
        ]
        metrics_mode = "w" if current_round == 0 else "a"

        with open(metrics_log_path, metrics_mode, newline="") as f:
            writer = csv.writer(f)
            if current_round == 0:
                writer.writerow(metrics_header)

            for i in range(num_rounds):
                print(i, accuracy_log[i][1] if i < len(accuracy_log) else None, var_u_log[i][1] if i < len(var_u_log) else None, surrogate_log[i][1] if i < len(surrogate_log) else None, awpsp_accuracy_log[i][1] if i < len(awpsp_accuracy_log) else None, awpsp_instant_fairness_log[i][1] if i < len(awpsp_instant_fairness_log) else None,  awpsp_cumul_fairness_log[i][1] if i < len(awpsp_cumul_fairness_log) else None, corr_failure_log[i][1] if i < len(corr_failure_log) else None, awpsp_covered_labels_log[i][1] if i < len(awpsp_covered_labels_log) else None, psp_accuracy_log[i][1] if i < len(psp_accuracy_log) else None, psp_instant_fairness_log[i][1] if i < len(psp_instant_fairness_log) else None, psp_cumul_fairness_log[i][1] if i < len(psp_cumul_fairness_log) else None, psp_covered_labels_log[i][1] if i < len(psp_covered_labels_log) else None, selected_awpsp_log[i][1] if i < len(selected_awpsp_log) else None, selected_psp_log[i][1] if i < len(selected_psp_log) else None, awpsp_avg_score_log[i][1] if i < len(awpsp_avg_score_log) else None, psp_avg_score_log[i][1] if i < len(psp_avg_score_log) else None, awpsp_labels_log[i][1] if i < len(awpsp_labels_log) else None, psp_labels_log[i][1] if i < len(psp_labels_log) else None, awpsp_KL_log[i][1] if i < len(awpsp_KL_log) else None, psp_KL_log[i][1] if i < len(psp_KL_log) else None, awpsp_unseen_log[i][1] if i < len(awpsp_unseen_log) else None, psp_unseen_log[i][1] if i < len(psp_unseen_log) else None, awpsp_gini_log[i][1] if i < len(awpsp_gini_log) else None, psp_gini_log[i][1] if i < len(psp_gini_log) else None)

            writer.writerow([
                current_round,
                experiment_context["logical_population"],
                experiment_context["selected_per_round"],
                experiment_context["physical_clients"],
                experiment_context["split_mode"],
                experiment_context["dirichlet_alpha"],
                experiment_context["selector_mode"],
                experiment_context["correlation_noise_pct"],
                experiment_context["seed"],
                accuracy_log[-1][1] if accuracy_log else None,
                var_u_log[-1][1] if var_u_log else None,
                surrogate_log[-1][1] if surrogate_log else None,
                awpsp_accuracy_log[-1][1] if awpsp_accuracy_log else None,
                awpsp_instant_fairness_log[-1][1] if awpsp_instant_fairness_log else None,
                awpsp_cumul_fairness_log[-1][1] if awpsp_cumul_fairness_log else None,
                awpsp_avg_within_class_log[-1] if awpsp_avg_within_class_log else None,
                awpsp_fairness_inter_class_log[-1] if awpsp_fairness_inter_class_log else None,
                corr_failure_log[-1][1] if corr_failure_log else None,
                awpsp_covered_labels_log[-1][1] if awpsp_covered_labels_log else None,
                psp_accuracy_log[-1][1] if psp_accuracy_log else None,
                psp_instant_fairness_log[-1][1] if psp_instant_fairness_log else None,
                psp_cumul_fairness_log[-1][1] if psp_cumul_fairness_log else None,
                psp_avg_within_class_log[-1] if psp_avg_within_class_log else None,
                psp_fairness_inter_class_log[-1] if psp_fairness_inter_class_log else None,
                psp_covered_labels_log[-1][1] if psp_covered_labels_log else None,
                oort_accuracy_log[-1][1] if oort_accuracy_log else None,
                oort_instant_fairness_log[-1][1] if oort_instant_fairness_log else None,
                oort_cumul_fairness_log[-1][1] if oort_cumul_fairness_log else None,
                oort_avg_within_class_log[-1] if oort_avg_within_class_log else None,
                oort_fairness_inter_class_log[-1] if oort_fairness_inter_class_log else None,
                oort_covered_labels_log[-1][1] if oort_covered_labels_log else None,
                selected_awpsp_log[-1][1] if selected_awpsp_log else None,
                selected_psp_log[-1][1] if selected_psp_log else None,
                selected_oort_log[-1][1] if selected_oort_log else None,
                awpsp_avg_score_log[-1][1] if awpsp_avg_score_log else None,
                psp_avg_score_log[-1][1] if psp_avg_score_log else None,
                oort_avg_score_log[-1][1] if oort_avg_score_log else None,
                awpsp_labels_log[-1][1] if awpsp_labels_log else None,
                psp_labels_log[-1][1] if psp_labels_log else None,
                oort_labels_log[-1][1] if oort_labels_log else None,
                awpsp_KL_log[-1][1] if awpsp_KL_log else None,
                psp_KL_log[-1][1] if psp_KL_log else None,
                oort_KL_log[-1][1] if oort_KL_log else None,
                awpsp_unseen_log[-1][1] if awpsp_unseen_log else None,
                psp_unseen_log[-1][1] if psp_unseen_log else None,
                oort_unseen_log[-1][1] if oort_unseen_log else None,
                awpsp_gini_log[-1][1] if awpsp_gini_log else None,
                psp_gini_log[-1][1] if psp_gini_log else None,
                oort_gini_log[-1][1] if oort_gini_log else None,
            ])

        summary = compute_final_metrics(base_model, current_round)
        if summary:
            print("Final metrics summary:")
            for key, value in summary.items():
                print(f"{key}: {value}")
            write_header = not os.path.exists(final_metrics_path)
            with open(final_metrics_path, "a") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(list(summary.keys()))
                writer.writerow(list(summary.values()))

        sys_end = snapshot_system()
        cpu_pct, mem_used_kb, disk_delta = summarize_system(sys_start, sys_end)
        print(
            "📊 Round timing: selection={:.2f}s fair={:.2f}s awpsp={:.2f}s psp={:.2f}s total={:.2f}s".format(
                selection_end - round_start,
                fair_end - selection_end,
                awpsp_end - fair_end,
                psp_end - awpsp_end,
                psp_end - round_start,
            )
        )
        print(
            "🧮 System usage: CPU~{:.1f}% MemUsed~{:.1f}MB DiskDelta~{}"
            .format(cpu_pct, mem_used_kb / 1024.0, disk_delta)
        )


def wait_for_latency_data(num_clients=2):
    print("⏳ Waiting for latency updates from clients...")
    while True:
        ready = 0
        for node, metadata in shared_state.topology.dht.table.items():
            if metadata.get("latency") is not None and metadata.get("packet_loss") is not None:
                ready += 1
        print(f"✅ Clients with latency info: {ready} / {num_clients}")
        if ready >= num_clients:
            break
        time.sleep(2)

    print("🚀 Sufficient clients reported latency. Starting training.")
    run_federated_training()



if __name__ == "__main__":
    print("🚀 Starting main server HTTP API...")


    # Step 1: Start the server in a separate thread
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8080), daemon=True).start()

    # Step 2: Start topology initialization in the background
    threading.Thread(target=initialize_topology).start()

    # Step 3: Wait for latency info from clients, then start training
    #threading.Thread(target=wait_for_latency_data).start()
