
import copy
import math
import os
import re
import requests
import subprocess
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
from flask import Flask, request


class CustomHost:
    def __init__(self, name, ip, port, transform, cifar_loader, dnn_model, *args, **kwargs):
        self.name = name
        self.ip = ip
        self.port = port
        self.client_cache = {}
        self.dht = None
        self.availability_predictor = None
        self.node = name
        self.transform = transform
        self.cifar_loader = cifar_loader
        self.latency = None  # Will store latency of the host dynamically
        self.packet_loss = None
        self.dnn_model = dnn_model
        print(f"✅ CustomHost registered: {self.name} at {self.port}")
        assert self.ip is not None and self.port is not None, "Missing IP or port for CustomHost"

    def predict_failure(self, image):
        """Process the image through the DNN model."""
        # Transform and normalize the CIFAR-10 image
        self.dnn_model.eval()  # Set to evaluation mode
        output = self.dnn_model(image)
        _, predicted = torch.max(output.data, 1)
        return output, predicted



class TopologyProvider:
    def __init__(self, device_names, num_workers, workers, num_epochs, link_latency=None, link_loss=None, model_name='resnet'):
        self.device_names = device_names
        self.num_workers = num_workers
        self.link_latency = f"{link_latency / 2}ms" if link_latency else None
        self.link_loss = (1 - np.sqrt(1 - link_loss / 100)) * 100 if link_loss else None
        self.num_epochs = num_epochs
        self.model_name = model_name
        self.sample_images = []
        self.utility_log = defaultdict(float)  # Tracks cumulative utility u_k(T)
        self.previous_losses = {}  # Stores last loss per client
        self.transform = self.get_transform() 
        self.cifar_loader = self.load_cifar_data()
        self.dht = DHT(size=100)  # Initialize the DHT
        self.availability_predictor = AvailabilityPredictor(node_count=len(device_names) * num_workers)
        self.availability_predictor.load_history()
        self.participation_log = {}
        self.current_round = 0
        self.failed_nodes = {}
        self.recovery_log = {worker.name: None for worker in workers}  # When a node is reintegrated
        self.node_neighbors = {} 
        self.total_rounds_elapsed = 0
        self.availability_counts = defaultdict(int)
        self.last_model_states = {}  # node_id → (model_state_dict, round_number)
        self.surrogate_contributions = defaultdict(int)
        self.surrogate_staleness = defaultdict(int)
        self.registered_hosts = {}
        # persistent model + optimizer inside each client
        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)  # CIFAR-10
        self.dnn_model = self.model
        self.criterion = torch.nn.CrossEntropyLoss()
#        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=0.001)
        for name, param in self.model.named_parameters():
            param.requires_grad = name.startswith("layer4") or name.startswith("fc")
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.001,
        )

    def get_subset_indices(self, worker_name, dataset, subset_size=1000,  seed=42, labels_per_client=None):
        """
        Return non-IID training data indices per worker (by label).
        """
        from torchvision.datasets import CIFAR10
        import numpy as np

        # Extract numeric index from device name, works for 'hX' or 'Device_X'
        index = int(''.join(filter(str.isdigit, worker_name)))

        # get all indices
#        all_indices = list(range(total_size))

        
        # Keep logical-client label cardinality aligned with the coordinator request.
        num_labels_per_worker = int(labels_per_client if labels_per_client is not None else os.getenv("LOGICAL_LABELS_PER_CLIENT", "2"))
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

        return label_indices[:subset_size]


    def get_subset_indices1(self, worker_name, total_size, subset_size=1000, seed=42):
        # Make subset selection deterministic per worker
        index = int(worker_name.replace("h", ""))
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
    
    def load_cifar_data(self, subset_size=1000):
        """Load the CIFAR-10 dataset."""
        full_dataset = datasets.CIFAR10(root='data/', train=True, download=True, transform=self.transform)

        worker_name = self.device_names[0]

        return self.get_or_create_loader(worker_name, subset_size=subset_size, dataset=full_dataset)


    def get_or_create_loader(self, logical_id, subset_size=1000, dataset=None, labels_per_client=None):
        if not hasattr(self, "loader_cache"):
            self.loader_cache = {}
        cache_key = (logical_id, labels_per_client)
        if cache_key in self.loader_cache:
            return self.loader_cache[cache_key]
        full_dataset = dataset or datasets.CIFAR10(
            root='data/', train=True, download=True, transform=self.transform
        )
        indices = self.get_subset_indices(logical_id, full_dataset, subset_size, labels_per_client=labels_per_client)
        print(f"client is: {logical_id}")
        print(f"client label indices are: {indices}")
        indices = np.array(indices)
        full_dataset.data = full_dataset.data[indices]
        full_dataset.targets = [full_dataset.targets[i] for i in indices]
        train_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)
        self.loader_cache[cache_key] = train_loader
        return train_loader

    def run_local_training(self, global_weights, local_epochs=5, logical_id=None, logical_labels_per_client=None):
        """Train locally using persistent model/optimizer."""
        if global_weights is not None:
           # Merge global weights with local model instead of full overwrite
           current_state = self.model.state_dict()
           for key in current_state.keys():
              if key in global_weights:
                 current_state[key] = 0.5 * current_state[key] + 0.5 * global_weights[key]
           self.model.load_state_dict(current_state)

        loader = self.cifar_loader
        if logical_id:
            loader = self.get_or_create_loader(logical_id, labels_per_client=logical_labels_per_client)

        # Training loop
        for epoch in range(local_epochs): 
            self.model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for i, (inputs, labels) in enumerate(loader):  # Assuming train_loader is your DataLoader
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

                running_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)  # Get predicted class
                correct_predictions += (predicted == labels).sum().item()  # Count correct predictions
                total_predictions += labels.size(0)  # Total predictions

            # Calculate average loss and accuracy for the epoch
            avg_loss = running_loss / len(self.cifar_loader)
            accuracy = (correct_predictions / total_predictions) * 100

            print(f"epoch [{epoch+1}/{local_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Return updated weights to the server
        return self.model.state_dict()


    def run_federated_round(self, selected_hosts, global_weights, model=None):
        # Step 0: Send global weights to each selected host and train locally via TopologyProvider
        for node in selected_hosts:
            trained_model = self.load_dnn_model(
            self.cifar_loader,
            model=copy.deepcopy(model),  # one model per client
            model_weights=copy.deepcopy(global_weights)
            )
            #  Save the updated model back to the host
            self.dht.table[node]['host'].dnn_model = trained_model

        # Step 2: Collect updated weights from each host
        updated_weights = [
            self.dht.table[node]['host'].dnn_model.state_dict()
            for node in selected_hosts
        ]

        # Step 3: Aggregate updated weights
        new_global_weights = self.aggregate_weights(updated_weights)
        return new_global_weights

    def aggregate_weights(self, weight_list):
        new_state = copy.deepcopy(weight_list[0])
        for key in new_state:
            if torch.is_floating_point(new_state[key]):
               new_state[key] = sum(w[key] for w in weight_list) / len(weight_list)
            else:
               # Non-float types (e.g. LongTensor): just copy the first one
               new_state[key] = weight_list[0][key]
        return new_state  


    def register_worker(self, name, ip, port):
        """Register a new worker pod with its metadata."""
        print(f"📥 Registering worker: {name}")
        host = CustomHost(
            name=name,
            ip=ip,
            port=port,
            dnn_model=self.dnn_model,
            transform = self.transform,
            cifar_loader = self.cifar_loader
        )
        self.registered_hosts[name] = host
        self.dht.table[name] = {
            "host": self.registered_hosts[name],
            "ip": ip,
            "node": name
        }


    def measure_latency_and_loss(self, server_ip):
        """Ping the server host to measure latency and packet loss."""
        try:
            output1 = subprocess.check_output(["ping", "-c", "5", server_ip], stderr=subprocess.STDOUT).decode()
            # Extract latency (min/avg/max/mdev)
            latency_match1 = re.search(r'rtt min/avg/max/mdev = ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)', output1)
            latency1 = float(latency_match1.group(2)) * 1000  # Extract avg latency
            # Extract packet loss
            loss_match1 = re.search(r'(\d+)% packet loss', output1)
            packet_loss1 = float(loss_match1.group(1)) if loss_match1 else None
            print(f"Latency: {latency1} ms, Packet Loss: {packet_loss1}%")

            self.current_round += 1
            worker_name = self.device_names[0] 
            self.update_participation_log(worker_name, self.current_round)
            # Compute success rates
            success_comp = 1.0 if len(self.participation_log.get(worker_name)) / self.current_round > 0.7 else 0.0 
            print(f"self.participation_log.get({worker_name}) is {self.participation_log.get(worker_name)} and self.current_round is {self.current_round}")
            success_comm = 1.0 if latency1 < 90 and packet_loss1 < 40 else 0.0  

            # Update availability history
            worker_name = self.device_names[0]
            print("worker_name: ", worker_name)
            initial_availability = self.availability_predictor.predict(worker_name)
            self.availability_predictor.update(worker_name, success_comp, success_comm)

            return latency1, packet_loss1, initial_availability

        except Exception as e:
            print(f"❌ Failed to ping the  server: {e}")



    def get_freshness(self, node, current_round):
        """Returns how long since this node was last selected."""
        #host = self.dht.table.get(node)  # node is a string like 'h10'
        last_selected_round = self.participation_log.get(node)
        if last_selected_round is None or not last_selected_round:
           freshness = current_round  # Never participated
        else:
           latest_round = max(last_selected_round)
           freshness = current_round - latest_round

        print(f"{node}: last_selected_round={last_selected_round}, current_round={current_round}, freshness={freshness}")
        return freshness


    def update_participation_log(self, node, current_round):
        """Update participation log for fairness tracking."""
        print("node: ", node)
        if node not in self.participation_log:
           self.participation_log[node] = []  # initialize list
        self.participation_log[node].append(current_round)  # log current round




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
