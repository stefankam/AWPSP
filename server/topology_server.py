
# topology_server.py

import sys
from flask import Flask, request
import io
import re
import requests
import threading
import copy
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import hashlib
from collections import defaultdict
import time
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
import random
import json
import time
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

class TopologyProvider:
    def __init__(self, device_names, num_epochs, link_latency=None, link_loss=None, model_name='resnet', device_registry=None):
        self.devices = device_names 
        self.link_latency = f"{link_latency / 2}ms" if link_latency else None
        self.link_loss = (1 - np.sqrt(1 - link_loss / 100)) * 100 if link_loss else None
        self.num_epochs = num_epochs
        self.model_name = model_name
        self.utility_log = defaultdict(float)  # Tracks cumulative utility u_k(T)
        self.previous_losses = {}  # Stores last loss per client
        self.transform = self.get_transform() 
        self.fixed_indices = {}
        self.cifar_loader = self.load_cifar_data()
        self.dht = DHT(size=100)  # Initialize the DHT
        self.availability_predictor = AvailabilityPredictor(node_count=len(device_names) * len(self.devices))
        self.availability_predictor.load_history()
        self.participation_log = {}
        self.failed_nodes = []
        self.recovery_log = {worker: None for worker in self.devices}  # When a node is reintegrated
        self.node_neighbors = {} 
        self.node_losses = {}
        self.total_rounds_elapsed = 0
        self.availability_counts = defaultdict(int)
        self.psp_availability_counts = defaultdict(int)
        self.awpsp_availability_counts = defaultdict(int)
        self.last_model_states = {}  # node_id → (model_state_dict, round_number)
        self.surrogate_contributions = defaultdict(int)
        self.surrogate_staleness = defaultdict(int)
        self.device_registry = device_registry or {}
        self.failure_correlation = defaultdict(lambda: defaultdict(set))
        self.recovery_counters = {}   # how many healthy rounds since failure
        self.probation_rounds = {}    # when node first recovered
        self.recovery_threshold = 4   # rounds to wait before marking as recovered
        self.probation_duration = 3   # rounds to weight updates less

    def get_subset_indices(self, worker_name, dataset, subset_size=1000, seed=42):
        """
        Return non-IID training data indices per worker (by label).
        """
        from torchvision.datasets import CIFAR10
        import numpy as np

        # Extract numeric index from device name, works for 'hX' or 'Device_X'
        index = int(worker_name.replace("Device_", ""))

        # get all indices
#        all_indices = list(range(total_size))

        # Assign 2 labels per client (you can change this)
        num_labels_per_worker = 2
        total_labels = 10

        start = (index * num_labels_per_worker) % total_labels
        worker_labels = list(range(total_labels))[start:start + num_labels_per_worker]

        # Get all indices that belong to the assigned labels
#        label_indices = np.where(np.isin(all_indices, worker_labels))[0]
        labels_array = np.array(dataset.targets)
        label_indices = np.where(np.isin(labels_array, worker_labels))[0]

        # deterministic shuffle
        rng = np.random.RandomState(seed + index)
        rng.shuffle(label_indices)

        print(f"[DEBUG] {worker_name} first indices:", label_indices[:5])
        print(f"[DEBUG] {worker_name} labels in subset:", [dataset.targets[i] for i in label_indices[:20]])

        self.fixed_indices[worker_name] = label_indices[:subset_size]

        return label_indices[:subset_size]



    def get_subset_indices1(self, worker_name, total_size, subset_size=1000, seed=42):
        # Make subset selection deterministic per worker
        index = int(worker_name.replace("Device_", ""))
        random.seed(seed + index)
        all_indices = list(range(total_size))
        random.shuffle(all_indices)
        return all_indices[:subset_size]


    def load_dnn_model(self, train_loader, model = None, model_weights=None):
        """Load the DNN model for failure prediction."""

        # Only train the final fully connected layer
        model.fc.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, 10)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # If weights were passed from coordinator
        if model_weights is not None:
           model.load_state_dict(model_weights)

        # Use CrossEntropyLoss for classification
        criterion = torch.nn.CrossEntropyLoss()
        # Reinitialize optimizer for only the classifier layer
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

        # Training loop
        for epoch in range(self.num_epochs):  # Number of epochs
            model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for i, (inputs, labels) in enumerate(train_loader):  # Assuming train_loader is your DataLoader
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)  # Get predicted class
                correct_predictions += (predicted == labels).sum().item()  # Count correct predictions
                total_predictions += labels.size(0)  # Total predictions

            # Calculate average loss and accuracy for the epoch
            avg_loss = running_loss / len(train_loader)
            accuracy = (correct_predictions / total_predictions) * 100

            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        return model        


    def get_trained_model(self):
        """Return the trained model for use by each host."""
        return self.dnn_model

    def get_transform(self):
        """Get the transform needed for CIFAR-10."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2430, 0.2610])
        ])
    
    def load_cifar_data(self):
        """Load the CIFAR-10 dataset."""
        full_dataset = datasets.CIFAR10(root='data/', train=True, download=True, transform=self.transform)

        self.label_map = {}
        self.dataloaders = {}

        for worker_name in self.devices:
            indices = self.get_subset_indices(worker_name, full_dataset, subset_size=1000)
            filtered_dataset = torch.utils.data.Subset(full_dataset, indices)

            # Store dataloader per worker
            self.dataloaders[worker_name] = DataLoader(filtered_dataset, batch_size=32, shuffle=False)

            # Record which labels are present in this worker's dataset
            label_set = set()
            for i in indices:
                _, label = full_dataset[i]
                label_set.add(label)

            self.label_map[worker_name] = sorted(label_set)
            print(f"📦 {worker_name} assigned labels: {sorted(label_set)}")

        return self.dataloaders  # Optionally return if you want



    def send_weights_to_client(self, device_id, global_weights, max_retries=50, sync_only=False, logical_id=None):
        # Get IP and port of the device
        entry = self.device_registry[device_id]
        ip = entry["ip"]
        port = entry["port"]

        # POST to client
        url = f"http://{ip}:{port}/train"

        for attempt in range(max_retries):
            try:
               print(f"📤 Sending weights to {device_id} at {url} (attempt {attempt + 1})")

               # 💡 Recreate the buffer and files inside the loop!
               buffer = io.BytesIO()
               torch.save(global_weights, buffer)
               buffer.seek(0)
               files = {"weights": ("model.pth", buffer)}

               # Send sync_only flag so client can decide whether to train
               data = {"sync_only": sync_only}
               if logical_id is not None:
                  data["logical_id"] = logical_id

               response = requests.post(url, data=data, files=files, timeout=5000)

               if response.status_code == 200:
                  return torch.load(io.BytesIO(response.content), map_location="cpu")
               else:
                  print(f"⚠️  {device_id} unavailable (status {response.status_code}), retrying...")
            except Exception as e:
               print(f"❌ Error contacting {device_id}: {e}, retrying...")
            time.sleep(2)

        print(f"❌ Client {device_id} at {ip}:{port} failed after {max_retries} retries. Skipping.")
        return None


    def run_logical_federated_round(self, logical_ids, physical_ids, global_weights, per_client_timeout=30):
        updated_weights = []
        if not physical_ids:
            return None

        def train_logical_on_physical(logical_id, physical_id):
            print(f"Sending logical client {logical_id} to physical {physical_id}")
            start_time = time.time()
            client_weights = None
            while time.time() - start_time < per_client_timeout:
                client_weights = self.send_weights_to_client(
                    physical_id,
                    global_weights,
                    sync_only=False,
                    logical_id=logical_id,
                )
                if client_weights is not None:
                    return (client_weights, 1.0)
                time.sleep(0.5)
            print(f"   ^z^o Logical client {logical_id} via {physical_id} did not respond.")
            return None

        wave_size = len(physical_ids)
        for wave_start in range(0, len(logical_ids), wave_size):
            wave = logical_ids[wave_start:wave_start + wave_size]
            with ThreadPoolExecutor(max_workers=wave_size) as executor:
                futures = []
                for idx, logical_id in enumerate(wave):
                    physical_id = physical_ids[idx % wave_size]
                    futures.append(executor.submit(train_logical_on_physical, logical_id, physical_id))

                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        updated_weights.append(result)

        return self.aggregate_weights(updated_weights)



    def resolve_pod_url(self, node_name):
        namespace = "fl-simulation"
        pod_dns = f"http://{node_name}.{namespace}.svc.cluster.local:5000/train"
        return pod_dns


    def run_federated_round(self, selected_hosts, global_weights, model=None):
        updated_weights = []
        print("selected_hosts: ", selected_hosts)
        for host in selected_hosts:
            client_weights = self.send_weights_to_client(host, global_weights)
            if client_weights is not None:
                updated_weights.append(client_weights)

        return self.aggregate_weights(updated_weights)


    def aggregate_weights(self, weighted_list):
        # Filter out None entries
        weighted_list = [w for w in weighted_list if w is not None]
        if not weighted_list:
           print("   ^z         ^o No weights to aggregate (no clients participated this round).")
           return None  # or return last global weights, or reinitialize
        normalized = []
        for item in weighted_list:
            if isinstance(item, tuple) and len(item) == 2:
                state, factor = item
                if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], dict):
                    state, factor = state
                normalized.append((state, factor))
            else:
                normalized.append((item, 1.0))
        # normalized is [(state_dict, weight_factor), ...]
        new_state = copy.deepcopy(normalized[0][0])

        for key in new_state:
           if torch.is_floating_point(new_state[key]):
              total = 0.0
              total_weight = 0.0
              for w, factor in normalized:
                 total += w[key] * factor
                 total_weight += factor
              new_state[key] = total / total_weight if total_weight > 0 else total
           else:
              new_state[key] = normalized[0][0][key]
        return new_state  


    def get_node_labels(self, node, labels_per_client=2, total_labels=10):
        labels = self.label_map.get(node)
        if labels:
            return labels
        digits = ''.join(filter(str.isdigit, str(node)))
        if digits:
            idx = int(digits)
            start = (idx * labels_per_client) % total_labels
            return [((start + j) % total_labels) for j in range(labels_per_client)]
        return []



    def evaluate_global_model(self, model, selected_nodes=None, subset_size=1000, use_selected_nodes=True, physical_ids=None):
        correct = total = 0
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print("selected_nodes before calculating accuracy: ", selected_nodes)

        if use_selected_nodes and selected_nodes:
            # Keep mapping aligned with run_logical_federated_round wave scheduling:
            # wave index i maps to physical_ids[i % len(physical_ids)].
            mapped_nodes = []
            candidates = physical_ids if physical_ids else self.devices
            for i, node in enumerate(selected_nodes):
                node_str = str(node)
                if node_str in self.label_map or node_str in self.fixed_indices:
                    mapped = node_str
                elif candidates:
                    mapped = candidates[i % len(candidates)]
                else:
                    mapped = node_str
                mapped_nodes.append(mapped)

            per_client = self.evaluate_per_client_accuracy(model, mapped_nodes, use_unseen_test=True)
            values = [v for v in per_client.values() if v is not None]
            print("average per-client accuracy: ", values)
            return (sum(values) / len(values)) if values else 0.0
        else:
            eval_dataset = datasets.CIFAR10(
                root='data/',
                train=False,
                download=False,
                transform=self.transform,
            )
            test_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            accuracy = 100 * correct / total
            return accuracy


    def evaluate_per_client_accuracy(self, model, nodes, use_unseen_test=True):
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        full_dataset = datasets.CIFAR10(
            root='data/',
            train=(not use_unseen_test),
            download=False,
            transform=self.transform,
        )
        dataset_targets = np.array(full_dataset.targets)
        per_client_acc = {}

        with torch.no_grad():
            for node in nodes:
                if use_unseen_test:
                    node_labels = self.label_map.get(node, [])
                    if not node_labels:
                        per_client_acc[node] = None
                        continue
                    indices = np.where(np.isin(dataset_targets, node_labels))[0].tolist()
                else:
                    indices = self.fixed_indices.get(node, [])

                if len(indices) == 0:
                    per_client_acc[node] = None
                    continue

                subset = torch.utils.data.Subset(full_dataset, indices)
                loader = DataLoader(subset, batch_size=32, shuffle=False)
                correct = total = 0
                for images, labels in loader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                per_client_acc[node] = (100 * correct / total) if total else None

        return per_client_acc


    def get_freshness(self, node, current_round):
        """Returns how long since this node was last selected."""
        last_selected_round = self.participation_log.get(node)
        if last_selected_round is None or not last_selected_round:
           freshness = current_round  # Never participated
        else:
           latest_round = max(last_selected_round)
           freshness = current_round - latest_round

        print(f"{node}: last_selected_round={last_selected_round}, current_round={current_round}, freshness={freshness}")
        return freshness


    def update_participation_log(self, selected_nodes, current_round):
        """Update participation log for fairness tracking."""
        for node in selected_nodes:
            print("node: ", node)
            if node not in self.participation_log:
                self.participation_log[node] = []
            self.participation_log[node].append(current_round)  # Log current round



    def get_correlated_failure(self, current_round, availability_vectors, corr_threshold=0.6, num_neighbors=4):
        print("corr_threshold: ", corr_threshold)
   
        # 1. Compute trace-based correlation matrix
        device_ids = list(availability_vectors.keys())
        matrix = np.corrcoef([availability_vectors[d] for d in device_ids])
        device_idx = {f"h{int(device[1:]) - 1}": idx for idx, device in enumerate(device_ids)}
        print("device_idx :", device_idx)
        # 2. Compute proximity-based neighbors
        node_latencies = {
           node: metadata["latency"]
           for node, metadata in self.dht.table.items()
           if "latency" in metadata
        }


        self.node_neighbors = {}
        for node, latency in node_latencies.items():
            filtered = [(other_node, other_latency)
                    for other_node, other_latency in node_latencies.items()
                    if other_node != node]
            print(f"node {node} has latency: ", latency)
            print("filtered: ", filtered)
            sorted_neighbors = sorted(filtered, key=lambda x: abs(x[1] - latency))
            self.node_neighbors[node] = [n for n, _ in sorted_neighbors[:num_neighbors]]
 
        # 3. Update failure_correlation structure
        for node in self.failed_nodes:
            neighbors = self.node_neighbors.get(node, [])
            for neighbor in neighbors:
                if neighbor in self.failed_nodes:
                   if current_round not in self.failure_correlation[node][neighbor]:
                      self.failure_correlation[node][neighbor].add(current_round)
                   if current_round not in self.failure_correlation[neighbor][node]:
                      self.failure_correlation[neighbor][node].add(current_round)
        
        # 4. Detect current failed nodes
        print("🔎 Starting Failure Detection using latency & packet loss")
        for node, metadata in self.dht.table.items():
            latency = metadata.get("latency")
            loss = metadata.get("packet_loss")
            if latency is not None and loss is not None:
               if latency > 90 or loss >= 40:
                  if node not in self.failed_nodes:
                     print(f"❌ Node {node} failed: latency={latency}, loss={loss}")
                     self.failed_nodes.append(node)
               else:
                  # increment recovery counter
                  self.recovery_counters[node] = self.recovery_counters.get(node, 0) + 1
                  if node in self.failed_nodes and self.recovery_counters[node] >= self.recovery_threshold:
                     print(f"   ^|^e Node {node} recovered after {self.recovery_counters[node]} healthy rounds")
                     self.failed_nodes.remove(node)

                     # put node into probation phase
#                     self.probation_rounds[node] = 0

        # 5. Use trace correlation and failure correlation to detect groups
        correlated_failures = []
        for node in self.failed_nodes:
            neighbors = self.node_neighbors.get(node, [])
            for neighbor in neighbors:
#                if neighbor in self.failed_nodes:
                   normalized_node = node.lower().replace("device_", "h")
                   normalized_neighbor = neighbor.lower().replace("device_", "h")
                   print(f"device_idx[{normalized_node}]: ", device_idx[normalized_node])
                   trace_score = matrix[device_idx[normalized_node]][device_idx[normalized_neighbor]] if normalized_node in device_idx and normalized_neighbor in device_idx else 0                   
                   fail_score = len(self.failure_correlation[node][neighbor]) / (current_round + 1) if neighbor in self.failure_correlation[node] else 0
                   print(f"for node {node} and neighbor {neighbor}, trace_score is {trace_score} and fail_score is {fail_score}")
                   if trace_score >= corr_threshold or fail_score >= corr_threshold:
                       correlated_failures.append((node, neighbor))

        if correlated_failures:
           print("⚠️ Correlated failures detected:", correlated_failures)
        else:
           print("✅ No correlated failures detected.")


        return correlated_failures



    def select_fair_nodes(self, model, current_round, correlated_failures,  label_map, num_clients,  corr_threshold=0.35, lambda_=0.5, epsilon=1e-5):        
        """
        - Select a subset of active nodes with:
        - Inverse availability × reactive reweighting (fairness-aware)
        - Class (label) coverage
        """
        eta_0 = 1.0         # base surrogate weight
        lambda_ = 0.5       # decay rate (can be tuned or swept)

        # 1. Get active nodes
        correlated_nodes = {n for pair in correlated_failures for n in pair}
        active_hosts = [node for node, metadata in self.dht.table.items() if node not in self.failed_nodes and node not in correlated_nodes]
        all_hosts = [node for node, metadata in self.dht.table.items()]
        print(f"❌ Failed Hosts: {self.failed_nodes}")
        print("✅ Active Hosts:", active_hosts)

        # Save the latest model state after training
        for host in active_hosts:
            self.last_model_states[host] = (copy.deepcopy(model.state_dict()), current_round)

        #Load custom test subsets per node
        full_test_dataset = datasets.CIFAR10(
            root='data/', train=False, download=False, transform=self.transform)

        node_test_loaders = {}

        for node in all_hosts:
            subset = torch.utils.data.Subset(full_test_dataset, self.fixed_indices[node])
            node_test_loaders[node] = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False)


        all_clients = list(self.dht.table.keys())
        missing_clients = [n for n in all_clients if n not in active_hosts]
        loss_fn = torch.nn.CrossEntropyLoss()
        for node in missing_clients:
            if node in self.last_model_states:
               past_model_state, last_seen = self.last_model_states[node]
               delta = current_round - last_seen
               eta_k = eta_0 * math.exp(-lambda_ * delta)

               # Load old model to surrogate
               surrogate_model = models.resnet18(weights=None)
               surrogate_model.fc = nn.Linear(surrogate_model.fc.in_features, 10)
               surrogate_model.load_state_dict(past_model_state)
               surrogate_model.eval()

               # Evaluate on current batch for simulation
        #       for image, label in self.sample_images:
         #          if label.view(-1)[0].item() not in label_map.get(node, []):
          #            continue

               for image, label in node_test_loaders[node]:
                   labels = label.long()

                   with torch.no_grad():
                      output = surrogate_model(image)
                      loss = loss_fn(output, label).item()

                   # Store surrogate contribution and staleness
                   self.surrogate_contributions[node] = eta_k * loss
                   self.surrogate_staleness[node] = delta
                   break  # Only evaluate once 

        total_bias_bound = sum(self.surrogate_contributions.values())
        print(f"📉 Total surrogate contribution (bias bound): {total_bias_bound:.4f}")


        # 2. Compute inverse-availability × missed-round score
        scores = []
        for node in active_hosts:
            # Estimate π_k
            self.availability_counts[node] += 1
            if self.total_rounds_elapsed > 0:
               pi_k = self.availability_counts[node] / (self.total_rounds_elapsed)
            else:
               pi_k = 0

            # Missed rounds since last participation
            last_rounds = self.participation_log.get(node, [])
            last_selected = max(last_rounds) if last_rounds else 0
            print("last_selected: ",last_selected)
            missed = current_round - last_selected
            print("missed: ",missed)

            # Final score: inverse availability × reactive boost
            if pi_k > 0:
               score = (1 / pi_k ) * (1 + lambda_ * missed)
            else:
               score = 0
            scores.append((node, score))
            print("scores: ", (node, score))

        # 3. Sort nodes by fairness-aware score
        scores.sort(key=lambda x: x[1], reverse=True)

        # 4. Greedily select nodes with label coverage
        selected = []
        covered_labels = set()

        for node, _ in scores:
            node_labels = set(label_map.get(node, []))
            if not node_labels.issubset(covered_labels) or len(selected) < num_clients:
               selected.add(node)
               covered_labels.update(node_labels)
            if len(selected) >= num_clients:
               break 

        print("📊 Selected nodes:", selected)
        print("🏷️ Covered labels:", covered_labels)

        for node in self.surrogate_contributions:
            labels = label_map.get(node, [])
            covered_labels.update(labels)

        print(f"📚 Surrogate coverage — Labels retained via surrogates: {sorted(covered_labels)}")

        # 5. Update participation_log
        self.total_rounds_elapsed += 1
        print("total_rounds_elapsed: ", self.total_rounds_elapsed)
        self.update_participation_log(selected, current_round)


        # 6. Evaluate ΔF_k(t) and update u_k
        loss_fn = torch.nn.CrossEntropyLoss()
        for node in selected:
            host = self.dht.table[node]
            relevant_losses = []

            for image, label in node_test_loaders[node]:
 
              # Ensure label is LongTensor for CrossEntropyLoss
              label = label.long()

              # Get prediction and compute loss
              with torch.no_grad():
                 output = model(image)
                 loss = loss_fn(output, label).item()
                 relevant_losses.append(loss)
 
            if relevant_losses:
              avg_loss = sum(relevant_losses) / len(relevant_losses)

              # Compute utility as positive delta from previous loss
              prev = self.previous_losses.get(node, None)
              if prev is not None:
                 delta_f = max(0, prev - avg_loss)
                 self.utility_log[node] += delta_f

              # Save loss for next round
              self.previous_losses[node] = avg_loss


        # 7. Compute and report fairness variance
        normalized_utilities = []
        for node in self.utility_log:
            if self.total_rounds_elapsed > 0:
               pi_k = self.availability_counts[node] / self.total_rounds_elapsed
               u_k = self.utility_log[node]
               print(f"for node {node}, pi_k = {pi_k} and u_k = {u_k}") 
               if pi_k > 0:
                  avg_u_k = u_k / self.total_rounds_elapsed
                  u_tilde_k = avg_u_k / pi_k
                  normalized_utilities.append(u_tilde_k)
        var_u = 0.0
        if normalized_utilities:
            mean_u = sum(normalized_utilities) / len(normalized_utilities)
            var_u = sum((u - mean_u) ** 2 for u in normalized_utilities) / len(normalized_utilities)
            print(f"🎯 Fairness Variance (Normalized Utility): {var_u:.4f}")

        else:
            var_u = 0


        return selected, var_u, total_bias_bound 



    def psp_random_selection(self, model, correlated_failures, num_clients, label_map):
        """
        Classical PSP: Randomly sample a subset of available nodes.
        """
        available_nodes = [node for node in self.dht.table if node not in self.failed_nodes]
        selected = random.sample(available_nodes, min(num_clients, len(available_nodes)))

        # 4. Covered_labels
        covered_labels = set()
    
        for node in selected:
            node_labels = set(label_map.get(node, []))
            covered_labels.update(node_labels)

        # 6. Load custom test subsets per node
        full_test_dataset = datasets.CIFAR10(
            root='data/', train=False, download=False, transform=self.transform)

        node_test_loaders = {}

        for node in selected:
            self.psp_availability_counts[node] += 1
            subset = torch.utils.data.Subset(full_test_dataset, self.fixed_indices[node])
            node_test_loaders[node] = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False)


        # 7. Evaluate    ^tF_k(t) and Calculate Fairness Variance
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        # Store per-round losses
        current_round_losses = {}
        class_losses = defaultdict(list)

        for node in selected:
           relevant_losses = []

           for image, label in node_test_loaders[node]:
               labels = label.long()

               with torch.no_grad():
                    output = model(image)
                    losses = loss_fn(output, labels)       

               # iterate per sample to assign loss to correct class
               for  lbl, loss in zip(labels, losses):
                    class_losses[int(lbl.item())].append(loss.item())
                    relevant_losses.append(loss.item())


           if relevant_losses:
               avg_loss = sum(relevant_losses) / len(relevant_losses)
               current_round_losses[node] = avg_loss
               print(f"   ^=^s^i Node {node} - Avg loss this round: {avg_loss:.4f}")

           # Append to historical record for cumulative fairness
           self.node_losses.setdefault(node, []).append(avg_loss)


        # Pseudo-code for per-class variance
        per_class_variance = {c: np.var(losses) for c, losses in class_losses.items()}
        #weights = {c: 1 / label_counts[c] for c in per_class_variance}  # rarer labels weigh more
        #fairness_variance = np.average(list(per_class_variance.values()), weights=list(weights.values()))
        within_class_variance = np.mean(list(per_class_variance.values()))

        # after populating `class_losses` where keys are ints (class labels) and values are lists of per-sample losses

        import math

        # minimum samples per class to include it in the inter-class fairness metric
        MIN_SAMPLES_PER_CLASS = 2

        # 1) per-class statistics
        per_class_means = {}
        per_class_vars  = {}
        per_class_counts = {}

        for c, losses in class_losses.items():
               cnt = len(losses)
               per_class_counts[c] = cnt 
               if cnt == 0:
                  continue
               # use population mean; for within-class variance you may use ddof=1 if cnt>1
               per_class_means[c] = float(np.mean(losses))
               if cnt > 1:
                  per_class_vars[c] = float(np.var(losses, ddof=1))  # sample variance 
               else:
                  per_class_vars[c] = 0.0

        # 2) Inter-class fairness: variance of class means (this is the recommended fairness metric)
        valid_class_means = [m for c, m in per_class_means.items() if per_class_counts.get(c,0) >= MIN_SAMPLES_PER_CLASS]

        if len(valid_class_means) >= 1:
              # variance across class means (population variance)
              fairness_inter_class = float(np.var(valid_class_means, ddof=0))
              # optional: normalized (coefficient of variation) to compare across rounds with different loss scale
              mean_of_means = float(np.mean(valid_class_means)) 
              fairness_inter_class_rel = fairness_inter_class / (mean_of_means**2 + 1e-12)  # relative squared-CV 
        else:
              fairness_inter_class = 0.0
              fairness_inter_class_rel = 0.0

        # 3) (Optional) average within-class variance
        if per_class_vars:
              avg_within_class_var = float(np.mean(list(per_class_vars.values())))
        else:
              avg_within_class_var = 0.0

        # Logging
        print(f"[FAIRNESS] inter-class var = {fairness_inter_class:.6f}  (rel {fairness_inter_class_rel:.6f}), "
              f"avg-within-class-var = {avg_within_class_var:.6f}, classes-used = {len(valid_class_means)}")


        # ---- Fairness Metric 1: Instant (Current Round) ----
        instant_fairness_variance = 0.0
        if current_round_losses: 
           mean_loss_round = sum(current_round_losses.values()) / len(current_round_losses)
           instant_fairness_variance = sum(
              (loss - mean_loss_round) ** 2 for loss in current_round_losses.values()
           ) / len(current_round_losses)
           print(f"🎯 Instant Fairness Variance (this round): {instant_fairness_variance:.4f}")

        # ---- Fairness Metric 2: Cumulative (All Rounds So Far) ----
        cumulative_fairness_variance = 0.0
        avg_losses_over_time = {
           node: sum(losses) / len(losses)
           for node, losses in self.node_losses.items()
           if len(losses) > 0
        } 

        avg_availability_count = {}
        for node in self.psp_availability_counts:
            if self.total_rounds_elapsed > 0:
               avg_availability_count[node] = self.psp_availability_counts[node] / self.total_rounds_elapsed
               print(f"avg_availability_count[{node}]: {avg_availability_count[node]} for total rounds of {self.total_rounds_elapsed}")

        if avg_losses_over_time and sum(self.psp_availability_counts[node] for node in avg_losses_over_time) > 0:
           mean_loss_all = sum(avg_losses_over_time.values()) / len(avg_losses_over_time)
           cumulative_fairness_variance = sum( 
           self.psp_availability_counts[node] * ((avg_losses_over_time[node] - mean_loss_all) ** 2)
           for node in avg_losses_over_time
           ) / sum(self.psp_availability_counts[node] for node in avg_losses_over_time)
           print(f"Cumulative Fairness Variance (all rounds): {cumulative_fairness_variance:.4f}")


        # metrics for demonstration ---
        # Avg availability x freshness
        sel_scores = [self.dht.table[node]["availability"] * self.get_freshness(node, self.total_rounds_elapsed) 
                  for node in selected]
        avg_sel_score = float(np.mean(sel_scores)) if sel_scores else 0.0

        # Class coverage stats
        class_counts = np.zeros(10, dtype=np.int64)
        for node in selected:
          for lbl in label_map.get(node, []):
             class_counts[int(lbl)] += 1
        total = class_counts.sum()
        if total > 0:
          p = class_counts / total
          u = np.full_like(p, 1/len(p), dtype=np.float64)
          eps = 1e-12
          kl = float(np.sum(p*(np.log(p+eps)-np.log(u+eps))))
        else:
          kl = 0.0
        unseen_rate = float(np.mean(class_counts==0))

        # Participation Gini
        all_nodes = list(self.dht.table.keys())
        counts = np.array([len(self.participation_log.get(n, [])) for n in all_nodes], dtype=np.float64)
        if counts.sum()>0:
           diffs = np.abs(counts.reshape(-1,1)-counts.reshape(1,-1))
           gini = float(diffs.sum() / (2*counts.size*counts.sum()))
        else:
           gini = 0.0

        print(f"[RANDOM PSP] avg_score={avg_sel_score:.3f} |labels|={np.sum(class_counts>0)} KL={kl:.4f} unseen={unseen_rate:.2f} Gini={gini:.3f}")



       # 6. Evaluate    ^tF_k(t) and update u_k
        loss_fn = torch.nn.CrossEntropyLoss()
        for node in selected:
            host = self.dht.table[node]
            relevant_losses = []

            for image, label in node_test_loaders[node]:
 
              # Ensure label is LongTensor for CrossEntropyLoss
              label = label.long()

              # Get prediction and compute loss
              with torch.no_grad():
                 output = model(image)
                 loss = loss_fn(output, label).item()
                 relevant_losses.append(loss)
 
            if relevant_losses:
              avg_loss = sum(relevant_losses) / len(relevant_losses)

              # Compute utility as positive delta from previous loss
              prev = self.previous_losses.get(node, None)
              if prev is not None:
                 delta_f = max(0, prev - avg_loss)
                 self.utility_log[node] += delta_f

              # Save loss for next round
              self.previous_losses[node] = avg_loss


        # 7. Compute and report fairness variance
        normalized_utilities = []
        for node in self.utility_log:
            if self.total_rounds_elapsed > 0:
               pi_k = self.awpsp_availability_counts[node] / self.total_rounds_elapsed
               u_k = self.utility_log[node]
               print(f"for node {node}, pi_k = {pi_k} and u_k = {u_k}")
               if pi_k > 0:
                  u_tilde_k = u_k / pi_k
                  normalized_utilities.append(u_tilde_k)

        if normalized_utilities:
            mean_u = sum(normalized_utilities) / len(normalized_utilities)
            var_u = sum((u - mean_u) ** 2 for u in normalized_utilities) / len(normalized_utilities)
            print(f"   ^=^n    Fairness Variance (Normalized Utility): {var_u:.4f}")

        else:
            var_u = 0



        return selected, instant_fairness_variance, cumulative_fairness_variance, avg_within_class_var, fairness_inter_class, covered_labels, avg_sel_score, np.sum(class_counts>0), kl, unseen_rate, gini


    def prioritize_available_nodes(self, model, current_round, correlated_failures, num_clients, label_map):
        """
        Select a subset of active nodes prioritizing:
        - DHT availability and freshness
        - Class (label) coverage
        """
        # 1. Get active nodes
        correlated_nodes = {n for pair in correlated_failures for n in pair}
        active_hosts = [node for node, metadata in self.dht.table.items() if node not in self.failed_nodes and node not in correlated_nodes and node not in self.probation_rounds]
        print(f"❌ Failed Hosts: {self.failed_nodes}")
        print("✅ Active Hosts:", active_hosts)

        # 2. Get DHT availability × freshness scores for active nodes
        scores = []
        for node, meta in self.dht.table.items():
            if node in active_hosts:
               availability = meta["availability"]
               print("availability: ", availability)
               freshness = self.get_freshness(node, current_round)
               print("freshness: ", freshness)
               score = availability * freshness
               scores.append((node, score))
    
        # 3. Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        print("nodes and scores", scores)
    
        # 4. Greedily pick nodes that maximize class coverage from top-ranked
        selected = []
        covered_labels = set()
        sel_scores = []
        for node, score in scores:
            if node in selected:
                continue
            selected.append(node)
            sel_scores.append(score)   # <-- record score for avg
            covered_labels.update(label_map.get(node, []))
            if len(selected) >= num_clients:
               break  # all classes covered


        # 5. Update participation_log
        print("📊 Selected nodes:", selected)
        print("🏷️ Covered labels:", covered_labels)
        self.total_rounds_elapsed += 1
        print("total_rounds_elapsed: ", self.total_rounds_elapsed)
        self.update_participation_log(selected, current_round)


        # 6. Load custom test subsets per node

        full_test_dataset = datasets.CIFAR10(
            root='data/', train=False, download=False, transform=self.transform)

        test_targets = np.array(full_test_dataset.targets)
        node_test_loaders = {}

        for node in selected:
            self.awpsp_availability_counts[node] += 1
            # fixed_indices are built from train split (size 50k). When evaluating on
            # test split (size 10k), guard against out-of-bounds indices.
            candidate_indices = self.fixed_indices.get(node, [])
            valid_indices = [i for i in candidate_indices if 0 <= i < len(full_test_dataset)]

            if valid_indices:
                indices = [i for i in valid_indices if int(test_targets[i]) in covered_labels]
            else:
                # Fallback when no valid per-node fixed indices exist on test split.
                indices = np.where(np.isin(test_targets, list(covered_labels)))[0].tolist()

            subset = torch.utils.data.Subset(full_test_dataset, indices)
            node_test_loaders[node] = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False)


        # 7. Evaluate    ^tF_k(t) and Calculate Fairness Variance
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        # Store per-round losses
        current_round_losses = {}
        class_losses = defaultdict(list)

        for node in selected:
           relevant_losses = []

           for image, label in node_test_loaders[node]:
               labels = label.long()

               with torch.no_grad():
                    output = model(image)
                    losses = loss_fn(output, labels)       

               # iterate per sample to assign loss to correct class
               for  lbl, loss in zip(labels, losses):
                    class_losses[int(lbl.item())].append(loss.item())
                    relevant_losses.append(loss.item())


           if relevant_losses:
               avg_loss = sum(relevant_losses) / len(relevant_losses)
               current_round_losses[node] = avg_loss
               print(f"   ^=^s^i Node {node} - Avg loss this round: {avg_loss:.4f}")

           # Append to historical record for cumulative fairness
           self.node_losses.setdefault(node, []).append(avg_loss)

        # Pseudo-code for per-class variance
        per_class_variance = {c: np.var(losses) for c, losses in class_losses.items()}
        #weights = {c: 1 / label_counts[c] for c in per_class_variance}  # rarer labels weigh more
        #fairness_variance = np.average(list(per_class_variance.values()), weights=list(weights.values()))
        within_class_variance = np.mean(list(per_class_variance.values()))

        # after populating `class_losses` where keys are ints (class labels) and values are lists of per-sample losses

        import math

        # minimum samples per class to include it in the inter-class fairness metric
        MIN_SAMPLES_PER_CLASS = 2

        # 1) per-class statistics
        per_class_means = {}
        per_class_vars  = {}
        per_class_counts = {}

        for c, losses in class_losses.items():
               cnt = len(losses)
               per_class_counts[c] = cnt 
               if cnt == 0:
                  continue
               # use population mean; for within-class variance you may use ddof=1 if cnt>1
               per_class_means[c] = float(np.mean(losses))
               if cnt > 1:
                  per_class_vars[c] = float(np.var(losses, ddof=1))  # sample variance 
               else:
                  per_class_vars[c] = 0.0

        # 2) Inter-class fairness: variance of class means (this is the recommended fairness metric)
        valid_class_means = [m for c, m in per_class_means.items() if per_class_counts.get(c,0) >= MIN_SAMPLES_PER_CLASS]

        if len(valid_class_means) >= 1:
              # variance across class means (population variance)
              fairness_inter_class = float(np.var(valid_class_means, ddof=0))
              # optional: normalized (coefficient of variation) to compare across rounds with different loss scale
              mean_of_means = float(np.mean(valid_class_means)) 
              fairness_inter_class_rel = fairness_inter_class / (mean_of_means**2 + 1e-12)  # relative squared-CV 
        else:
              fairness_inter_class = 0.0
              fairness_inter_class_rel = 0.0

        # 3) (Optional) average within-class variance (what you had before)
        if per_class_vars:
              avg_within_class_var = float(np.mean(list(per_class_vars.values())))
        else:
              avg_within_class_var = 0.0

        # Logging
        print(f"[FAIRNESS] inter-class var = {fairness_inter_class:.6f}  (rel {fairness_inter_class_rel:.6f}), "
              f"wavg-within-class-var = {avg_within_class_var:.6f}, classes-used = {len(valid_class_means)}")


        # ---- Fairness Metric 1: Instant (Current Round) ----
        instant_fairness_variance = 0.0
        if current_round_losses:
           mean_loss_round = sum(current_round_losses.values()) / len(current_round_losses)
           instant_fairness_variance = sum(
              (loss - mean_loss_round) ** 2 for loss in current_round_losses.values()
           ) / len(current_round_losses)
           print(f"🎯 Instant Fairness Variance (this round): {instant_fairness_variance:.4f}")

        # ---- Fairness Metric 2: Cumulative (All Rounds So Far) ----
        cumulative_fairness_variance = 0.0
        avg_losses_over_time = {
           node: sum(losses) / len(losses)
           for node, losses in self.node_losses.items()
           if len(losses) > 0
        } 


        avg_availability_count = {}
        for node in self.awpsp_availability_counts:
            if self.total_rounds_elapsed > 0:
               avg_availability_count[node] = self.awpsp_availability_counts[node] / self.total_rounds_elapsed
               print(f"availability_counts[{node}] is {self.awpsp_availability_counts[node]}")

        if avg_losses_over_time and sum(self.awpsp_availability_counts[node] for node in avg_losses_over_time) > 0:
           mean_loss_all = sum(avg_losses_over_time.values()) / len(avg_losses_over_time)
           cumulative_fairness_variance = sum( 
           self.awpsp_availability_counts[node] * ((avg_losses_over_time[node] - mean_loss_all) ** 2)
           for node in avg_losses_over_time
           ) / sum(self.awpsp_availability_counts[node] for node in avg_losses_over_time)
           print(f"Cumulative Fairness Variance (all rounds): {cumulative_fairness_variance:.4f}")


        # metrics for demonstration ---
        # Avg availability x freshness
        avg_sel_score = float(np.mean(sel_scores)) if sel_scores else 0.0

        # Class coverage stats
        class_counts = np.zeros(10, dtype=np.int64)
        for node in selected:
          for lbl in label_map.get(node, []):
             class_counts[int(lbl)] += 1
        total = class_counts.sum()
        if total > 0:
          p = class_counts / total
          u = np.full_like(p, 1/len(p), dtype=np.float64)
          eps = 1e-12
          kl = float(np.sum(p*(np.log(p+eps)-np.log(u+eps))))
        else:
          kl = 0.0
        unseen_rate = float(np.mean(class_counts==0))

        # Participation Gini
        all_nodes = list(self.dht.table.keys())
        counts = np.array([len(self.participation_log.get(n, [])) for n in all_nodes], dtype=np.float64)
        if counts.sum()>0:
           diffs = np.abs(counts.reshape(-1,1)-counts.reshape(1,-1))
           gini = float(diffs.sum() / (2*counts.size*counts.sum()))
        else:
           gini = 0.0

        print(f"[PRIORITIZED PSP] avg_score={avg_sel_score:.3f} |labels|={np.sum(class_counts>0)} KL={kl:.4f} unseen={unseen_rate:.2f} Gini={gini:.3f}")


       # 6. Evaluate    ^tF_k(t) and update u_k
        loss_fn = torch.nn.CrossEntropyLoss()
        for node in selected:
            host = self.dht.table[node]
            relevant_losses = []

            for image, label in node_test_loaders[node]:
 
              # Ensure label is LongTensor for CrossEntropyLoss
              label = label.long()

              # Get prediction and compute loss
              with torch.no_grad():
                 output = model(image)
                 loss = loss_fn(output, label).item()
                 relevant_losses.append(loss)
 
            if relevant_losses:
              avg_loss = sum(relevant_losses) / len(relevant_losses)

              # Compute utility as positive delta from previous loss
              prev = self.previous_losses.get(node, None)
              if prev is not None:
                 delta_f = max(0, prev - avg_loss)
                 self.utility_log[node] += delta_f

              # Save loss for next round
              self.previous_losses[node] = avg_loss


        # 7. Compute and report fairness variance
        normalized_utilities = []
        for node in self.utility_log:
            if self.total_rounds_elapsed > 0:
               pi_k = self.awpsp_availability_counts[node] / self.total_rounds_elapsed
               u_k = self.utility_log[node]
               print(f"for node {node}, pi_k = {pi_k} and u_k = {u_k}")
               if pi_k > 0:
                  u_tilde_k = u_k / pi_k
                  normalized_utilities.append(u_tilde_k)

        if normalized_utilities:
            mean_u = sum(normalized_utilities) / len(normalized_utilities)
            var_u = sum((u - mean_u) ** 2 for u in normalized_utilities) / len(normalized_utilities)
            print(f"   ^=^n    Fairness Variance (Normalized Utility): {var_u:.4f}")

        else:
            var_u = 0

        return selected, instant_fairness_variance, cumulative_fairness_variance, avg_within_class_var,  fairness_inter_class, covered_labels, avg_sel_score, np.sum(class_counts>0), kl, unseen_rate, gini


# Distributed Hash Table
class DHT:
    def __init__(self, size=100):
        self.table = {}
        self.size = size

    def _hash(self, key):
        key = str(key)  # Ensure key is always string
        return int(hashlib.sha1(key.encode()).hexdigest(), 16) % self.size

    def store(self, key, value):
        h = self._hash(key)
        self.table[h] = value

    def lookup(self, key):
        h = self._hash(key)
        return self.table.get(h, None)

    def all_nodes(self):
        return list(self.table.keys())

# Availability Predictor
class AvailabilityPredictor:
    def __init__(self, node_count, window_size=5, history_file="/tmp/availability_history.json"):
        self.window_size = window_size
        self.history = {
            f'node_{i}': {"comp": [], "comm": []} for i in range(node_count)
        }
        self.beta = {f'node_{i}': 0.5 for i in range(node_count)}  # Default return probability
        self.history_file = history_file
        self.load_history()

    def update(self, node, success_comp, success_comm):
        """Predict node availability using historical data and return probability."""
        if node not in self.history:
            self.history[node] = {"comp": [], "comm": []}

        # Append new values **before** writing to file
        self.history[node]["comp"].append(success_comp)
        self.history[node]["comm"].append(success_comm)
        
        # Keep only last T rounds
        if len(self.history[node]["comp"]) > self.window_size:
            self.history[node]["comp"].pop(0)
        if len(self.history[node]["comm"]) > self.window_size:
            self.history[node]["comm"].pop(0)

        with open(self.history_file, "w") as f:
            json.dump(self.history, f)


    def predict(self, node):
        """Predict node availability using historical data and return probability."""
        if node not in self.history:
            self.history[node] = {"comp": [], "comm": []}
        history_comp = self.history[node]["comp"]
        history_comm = self.history[node]["comm"]
        if not history_comp or not history_comm:
            return 0.5

        # Compute availability from historical data
        a_comp = sum(self.history[node]["comp"]) / len(self.history[node]["comp"]) if self.history[node]["comp"] else 0
        a_comm = sum(self.history[node]["comm"]) / len(self.history[node]["comm"]) if self.history[node]["comm"] else 0
        a_i = a_comp * a_comm  # Overall availability
        print(" a_i: ", a_i)
        # Compute future availability
        future_a_i = a_i + (1 - a_i) * self.beta.get(node, 0.5)
        print("future_a_i : ", future_a_i)
        return future_a_i

    def calculate_neighbor_availability(node, topology):
        neighbors = topology.get_neighbors(node)
        availabilities = []

        for neighbor_id in neighbors:
             neighbor_data = topology.dht.lookup(neighbor_id)
             if not neighbor_data:
                   continue

             availabilities.append(neighbor_data['availability'])

        if not availabilities:
             return 0  # No neighbors = assume 0 (or safe default like 0.5)

        return sum(availabilities) / len(availabilities)

    def save_history(self):
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                 with open(self.history_file, "r") as f:
                      self.history = json.load(f)
            except json.JSONDecodeError:
                 print(f"⚠️ History file corrupted. Starting fresh.")
                 self.history = {}
        else:
            self.history = {}
