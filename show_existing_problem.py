import sys
sys.path.append('/home/skb67/.local/lib/python3.10/site-packages/')
import re
import time
from collections import defaultdict
import torch
import torchvision
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
import random
from mininet.net import Mininet
from mininet.node import Host, OVSSwitch, OVSController
from mininet.link import TCLink
import time
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class CustomHost(Host):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name,  *args, **kwargs)
        self.client_cache = {}
        self.dht = None
        self.availability_predictor = None
        self.node_id = name
        self.latency = None  # Will store latency of the host dynamically
        self.failed = False  # Added to track if the host is available
        self.topology_provider = topology_provider
        self.dnn_model = self.topology_provider.get_trained_model()
        self.transform = self.topology_provider.transform
        self.cifar_loader = self.topology_provider.cifar_loader

    def update_latency(self, net, host_to_switch, switch_name):
        """Ping a host to measure latency and packet loss."""
        try:
            if self.failed:
                print(f"⚠️ {self.name} is marked as failed, skipping latency update.")
                return None

            # Find other hosts connected to the same switch
            same_switch_hosts = [h for h, s in host_to_switch.items() if s == switch_name and h != self.name]
            other_switch_hosts = [h for h, s in host_to_switch.items() if s != switch_name and h != self.name]
            print("same_switch_hosts: ", same_switch_hosts)
            print("other_switch_hosts: ", other_switch_hosts)
            if not same_switch_hosts:
               print(f"⚠️ No other hosts connected to {switch_name} for {self.name} to ping.")
               return
    
            # Select a target host to ping
            target_host_name_1 = random.choice(same_switch_hosts)
            target_host_name_2 = random.choice(other_switch_hosts)
            target_host_1 = net.get(target_host_name_1)
            target_host_2 = net.get(target_host_name_2)
            target_switch_1 = host_to_switch.get(target_host_name_1)
            target_switch_2 = host_to_switch.get(target_host_name_2)
#                net.addLink(switch_name, target_switch)
#                net.addLink(self.name, target_host_name)
            print(f"🔎 {self.name} with switch {switch_name} is pinging {target_host_1} at {target_host_1.IP()} with SAME switch {target_switch_1}")
            output1 = self.cmd(f"ping -c 5 {target_host_1.IP()}")
            print(f"🔎 {self.name} with switch {switch_name} is pinging {target_host_2} at {target_host_2.IP()} with OTHER switch {target_switch_2}")
            output2 = self.cmd(f"ping -c 5 {target_host_2.IP()}")

            # Extract latency (min/avg/max/mdev)
            latency_match1 = re.search(r'rtt min/avg/max/mdev = ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)', output1)
            latency1 = float(latency_match1.group(2))  # Extract avg latency
            # Extract packet loss
            loss_match1 = re.search(r'(\d+)% packet loss', output1)
            packet_loss1 = float(loss_match1.group(1)) if loss_match1 else None
            print(f"Host: {self.name}, Latency: {latency1} ms, Packet Loss: {packet_loss1}%")

            latency_match2 = re.search(r'rtt min/avg/max/mdev = ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)', output2)
            latency2 = float(latency_match2.group(2))  # Extract avg latency
            # Extract packet loss
            loss_match2 = re.search(r'(\d+)% packet loss', output2)
            packet_loss2 = float(loss_match2.group(1)) if loss_match2 else None
            print(f"Host: {self.name}, Latency: {latency2} ms, Packet Loss: {packet_loss2}%")

            self.latency = max(latency1, latency2)
            return self.latency

        except Exception as e:
            print(f"Error monitoring latency: {e}")


    def predict_failure(self, image):
        """Process the image through the DNN model."""
        # Convert tensor to PIL if needed
#        if image.ndim == 4 and image.shape[0] > 1:
#            image = image[0]  # pick the first image
#        while image.ndim > 3:
#            image = image.squeeze(0)
#        image = transforms.ToPILImage()(image) 
#        # Apply the exact same transforms used during training
        self.dnn_model.eval()  # Set to evaluation mode
        output = self.dnn_model(image)
        _, predicted = torch.max(output.data, 1)
        return predicted


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # Modify the final fully connected layer to match CIFAR-10 (10 classes)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)
        self.topology_provider = topology_provider
        self.cifar_loader = self.topology_provider.cifar_loader   
        self.failed = False  # Track model failure state

    def forward(self, x):
        # Define your forward pass here
        return self.model(x)

    def evaluate_classification_performance(self, dataloader, max_batches=5):
        """Evaluate classification accuracy on a given dataloader."""
        if  self.failed:
            print(f"Returning degraded accuracy.")
            return 0.0  # Or return degraded value, e.g., 30% of baseline

        correct = 0
        total = 0
        self.eval()  # Set model to evaluation mode

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                if i >= max_batches:
                    break
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total if total > 0 else 0
        return accuracy


    def calculate_swcd(self, baseline_accuracy):
        """Compute System-Wide Classification Degradation (SWCD)."""
        self.mark_failed()
        new_accuracy = self.evaluate_classification_performance(self.cifar_loader)
        self.mark_recovered()
        degradation = max(0, baseline_accuracy - new_accuracy)  # Ensure non-negative
        print(f"📉 SWCD : {degradation}%")
        return degradation


    def calculate_fis(self, baseline_accuracy):
        """Compute the Failure Impact Score (FIS) when this host fails."""
        self.mark_failed()  # Simulate node failure
        new_accuracy = self.evaluate_classification_performance(self.cifar_loader)
        self.mark_recovered()  # Restore the node

        impact_score = max(0, baseline_accuracy - new_accuracy) / baseline_accuracy if baseline_accuracy > 0 else 0
        print(f"⚠️ Failure Impact Score (FIS) : {impact_score:.4f}")
        return impact_score

    def mark_failed(self):
        """Mark this host as failed."""
        self.failed = True

    def mark_recovered(self):
        """Mark this host as recovered."""
        self.failed = False


class TopologyProvider:
    def __init__(
        self, 
        device_names,                   # required
        num_workers,                    # required
        model_name,                     # required
        num_epochs,                     # required
        link_latency=None,              # optional, default no latency
        link_loss=None,                 # optional, default no packet loss
        failure_probability=0.0,        # optional, default no failure
        corr_failure_probability=0.0,   # optional, default no failure
        topology_provider=None          # optional, advanced use
    ):
        self.device_names = device_names
        self.num_workers = num_workers
        self.link_latency = f"{link_latency / 2}ms" if link_latency else None
        self.link_loss = (1 - np.sqrt(1 - link_loss / 100)) * 100 if link_loss else None
        self.net = None
        self.switch_num = 0
        self.host_num = 0
        self.num_epochs = num_epochs
        self.model_name = model_name
        self.sample_images = []
        self.transform = self.get_transform() 
        self.cifar_loader = self.load_cifar_data()
        self.dnn_model = self.load_dnn_model(self.cifar_loader) 
        self.failed_nodes_per_switch =  {}
        self.host_to_switch = {}
        self.failure_probability = failure_probability  # Probability of worker failure
        self.corr_failure_probability = corr_failure_probability  # Probability of worker failure
        self.topology_provider = topology_provider
        self.failed_nodes = []  # Track failed nodes

    def assign_labels_to_workers(self, labels_per_worker):
       all_labels = list(range(10))  # CIFAR-10 has 10 classes
       assignments = {}
       for i in range(self.num_workers):
         start = (i * labels_per_worker) % len(all_labels)
         assigned = all_labels[start:start + labels_per_worker]
         assigned = [label + 1 for label in assigned]  # shift labels from [0-9] → [1-10]
         worker_name = f"h{i+1}"
         assignments[worker_name] = assigned
         print(f"📦 Worker {worker_name} assigned labels: {assigned}")
       return assignments

    def load_dnn_model(self, train_loader, max_samples=20):
        """Load the DNN model for failure prediction."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data loading (make sure the transform matches for both training and evaluation)
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        full_dataset = datasets.CIFAR10(root='/home/skb67/sim_fed_dht/data', train=True, download=True, transform=transform)

        # Assign labels to workers and build a combined subset of all assigned labels
        label_assignments = self.assign_labels_to_workers(labels_per_worker=1)
        all_assigned_labels = set(l for labels in label_assignments.values() for l in labels)

        print(f"Training only on labels: {sorted(all_assigned_labels)}")

        # Filter dataset to only include images with these labels
        filtered_indices = [i for i, (_, label) in enumerate(full_dataset)
                        if label in all_assigned_labels]
        filtered_dataset = torch.utils.data.Subset(full_dataset, filtered_indices)
        train_loader = DataLoader(filtered_dataset, batch_size=32, shuffle=True)

        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, 10)
        model.to(device)

        # Only train the final fully connected layer
        model.fc.requires_grad = True

        # Use CrossEntropyLoss for classification
        criterion = torch.nn.CrossEntropyLoss()
        # Reinitialize optimizer for only the classifier layer
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)


       # Collect  CIFAR samples to evaluate
        for i, (image, label) in enumerate(self.cifar_loader):
            self.sample_images.append((image, label))
            if len(self.sample_images) >= max_samples:
               break


        # Training loop
        for epoch in range(self.num_epochs):  # Number of epochs
            model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for i, (inputs, labels) in enumerate(train_loader):  # Assuming train_loader is your DataLoader
                inputs, labels = inputs.to(device), labels.to(device)
                if i >= max_samples:
                    break
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

            print(f"Epoch [{epoch+1}/10], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Save the model after training
        torch.save(model.state_dict(), '/tmp/trained_model.pth')

        return model

    def get_trained_model(self):
        """Return the trained model for use by each host."""
        return self.dnn_model

    def get_transform(self):
        """Get the transform needed for CIFAR-10, including resizing and normalization."""
        return transforms.Compose([
             transforms.Resize(224),  # Resize to 224x224 for ResNet
             transforms.ToTensor(),  # Convert image to tensor
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet
        ])

    def load_cifar_data(self):
        """Load the CIFAR-10 dataset."""
        cifar_data = datasets.CIFAR10(root='/home/skb67/sim_fed_dht/data', train=True, download=True, transform=self.get_transform())
        return DataLoader(cifar_data, batch_size=4, shuffle=True)

    def add_switch(self):
        """Add a new switch to the network."""
        name = f"s{self.switch_num + 1}"
        self.switch_num += 1
        return self.net.addSwitch(name)

    def add_worker(self, device, index, switch):
        """Add a worker node to the network."""
        if not self.net:
            raise ValueError("Mininet instance is not initialized. Call setup() first.")
        worker_name = f"{device}{index + 1}"
        self.host_num += 1
        ip_address = f"10.0.0.{self.host_num}"
        host = self.net.addHost(worker_name, 
                                cls=CustomHost, 
                                ip=ip_address, 
                                dnn_model=self.dnn_model, 
                                transform=self.transform, 
                                cifar_loader=self.cifar_loader)
        self.net.addLink(host, switch, delay=self.link_latency, loss=self.link_loss)
        return host


    def setup(self):
        """Set up the Mininet network and hosts."""
        print("🚀 Starting Mininet Setup...")
        self.net = Mininet(controller=OVSController, switch=OVSSwitch, link=TCLink, autoSetMacs=True)
        self.net.addController("c0")

        # Create multiple switches for logical groups (e.g., 3 groups)
        num_switches = min(5, self.num_workers)  # Ensure not more switches than workers
        switches = []
        for i in range(num_switches):
            switch = self.net.addSwitch(f's{i+1}')
            switches.append(switch)
            print(f"✅ Switch created: {switch}")

        # Add workers and connect to switches
        group_size = max(1, self.num_workers // num_switches)  # Prevent zero division

        for device in self.device_names:
            for i in range(self.num_workers):
            # Divide into switch groups
                switch_index = i // group_size
                switch_index = min(switch_index, num_switches - 1)  # Prevent overflow
                switch = switches[switch_index]
            
                # Add worker node using add_worker
                worker = self.add_worker(device, i, switch)
                print(f"🔗 Linked {worker} to switch {switch}")

                # Track the host-to-switch mapping
                self.host_to_switch[worker.name] = switch.name

                loss = '10%'
                delay = 500
                interface = f"{worker.name}-eth0"
#                self.net.addLink(worker, switch)
#                worker.cmd(f'tc qdisc add dev {interface}  root netem delay {delay}ms loss {loss}')
#                print(f"⏱️ Applied delay of {delay}ms and loss of {loss} to {worker} and {switch}")

#            for i in range(num_switches-1):
#                self.net.addLink(switches[i], switches[i+1])      
            
#            self.net.addLink(host, switch, delay, loss=self.link_loss)
#            print(f"⏱️ Added link between {host} and {switch} with {delay}ms delay")
#        host.cmd(f'tc qdisc add dev h1-eth0 root netem delay {delay} loss {loss}')
#        output =  host.cmd(f'tc qdisc show dev h1-eth0')
#        print("qdisc output: ", output)


        self.net.start()
        print("✅ Mininet started successfully!")
        for i in range(num_switches):
            self.net.addLink(switches[i-1], switches[i])      
        # Initialize
        self.failed_nodes_per_switch = {switch.name: [] for switch in self.net.switches}


    def simulate_unavailability(self, failure_rate=0):
        """Simulate unavailability of workers."""
        for host in self.net.hosts:
            if random.random() < failure_rate:
                host.mark_failed()
                print(f"⚠️ {host.name} is now marked as unavailable.")
            else:
                host.mark_recovered()

    def simulate_correlated_failures(self, correlation_rate=0): 
        """Simulate correlated failures for workers in the same switch.""" 
        for switch, failures in self.failed_nodes_per_switch.items():
            if random.random() < correlation_rate:
                # Mark all hosts in this switch as failed
                for host in self.net.hosts:
                    if self.host_to_switch.get(host.name) == switch:
                        host.mark_failed()
                        print(f"⚠️ {host.name} in switch {switch} is now marked as unavailable.")


    def evaluate_classification(self, host):
        """Evaluate image classification performance (accuracy & precision) for a single host."""
        if host.failed:
            print(f"Skipping evaluation for {host.name} (Failed)")
            return None

        try:
            image, label = next(iter(host.cifar_loader))  # Get a batch of CIFAR-10 data
        
            # Convert tensor to PIL Image
            image_pil = transforms.ToPILImage()(image.squeeze(0).detach())  # Remove batch dimension
            print("image_pil : ", image_pil)
            # Pass PIL image to prediction function
            predicted_class, prediction_probs = host.predict_failure(image_pil)
            print(f"🔍 Host: {host.name} | Predicted: {predicted_class} | Probabilities: {prediction_probs}")
            # Get true class label
            true_class = label[0].item()
            print("([label.item()]: ", [true_class])
            print("[predicted_class]: ", [predicted_class])
    
            # Calculate accuracy
            accuracy = float(predicted_class == true_class)
            print(f"📊 Host: {host.name} | Accuracy: {accuracy:.2f}")
    
            # Calculate precision (simplified example)
            tp = (predicted_class == true_class)
            fp = (predicted_class != true_class)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            print(f"📊 Host: {host.name} | Precision: {precision:.2f}")
    
            return accuracy, precision

        except Exception as e:
            print(f"⚠️ Error evaluating {host.name}: {e}")
            return None


    def assess_network_impact(self):
        """Assess SWCD and FIS for the entire network."""
        print("🧪 Measuring baseline classification accuracy...")
        model = ResNet()
        baseline_accuracy = model.evaluate_classification_performance(self.cifar_loader)
        print(f"✅ Baseline Accuracy: {baseline_accuracy:.2f}%")

        total_swcd = 0
        total_fis = 0
        num_hosts = len(self.host_to_switch)

        for host_name in self.host_to_switch.keys():
            host = self.net.get(host_name)

            swcd = model.calculate_swcd(baseline_accuracy)
            fis = model.calculate_fis(baseline_accuracy)
        
            total_swcd += swcd
            total_fis += fis

        avg_swcd = total_swcd / num_hosts if num_hosts > 0 else 0
        avg_fis = total_fis / num_hosts if num_hosts > 0 else 0

        print(f"📊 System-Wide Classification Degradation (SWCD): {avg_swcd:.2f}%")
        print(f"⚠️ Average Failure Impact Score (FIS): {avg_fis:.4f}")
        return avg_swcd, avg_fis


    def load_availability_traces(self, path):
        """
        Read availability traces from file.
        Format expected:
        device_0:
        """
        traces = {}
        current_device = None
        current_trace = []

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                   continue

                if line.startswith("device_"):
                   # Save the previous device trace
                   if current_device is not None:
                        # Convert device_0 → h1, device_1 → h2, etc.
                        idx = int(current_device.split("_")[1])
                        host_name = f"h{idx+1}"
                        traces[host_name] = current_trace

                   # Start new trace
                   current_device = line[:-1]  # remove colon
                   current_trace = []
                else:
                   current_trace.append(line)

            # Don't forget to save the last trace
            if current_device is not None:
               idx = int(current_device.split("_")[1])
               host_name = f"h{idx+1}"
               traces[host_name] = current_trace

        return traces


    def extract_availability_vectors(self):
         traces = self.load_availability_traces("./traces/traces.txt")
         def extract_vector(trace, length=100):
           wifi, charging = False, False
           vector = []
           for event in trace:
               if "wifi" in event:
                  wifi = "off" not in event
               elif "battery_charged" in event:
                  charging = "off" not in event
               # After each relevant event, record availability status
               availability = wifi and charging
               vector.append(int(availability))
           return vector[:length] + [0] * max(0, length - len(vector))

         availability_vectors = {
           device: extract_vector(trace) for device, trace in traces.items()
         }

         #print("availability_vectors: ", availability_vectors)
         return availability_vectors


    def simulate_active_hosts(self, availability_vectors,  corr_threshold=0.6, fail_prob=0.1, failure_mode= "random"):
       """Simulate failures using trace-based unavailability and correlated behavior."""
       print(f"\n🔎 [Simulating failures with mode={failure_mode}...")

       self.failed_nodes = []
       self.failed_nodes_per_switch = {s.name: [] for s in self.net.switches}
       print("corr_threshold: ", corr_threshold)
       print("fail_prob: ", fail_prob)

       # Compute correlation matrix
       device_ids = list(availability_vectors.keys())
       matrix = np.corrcoef([availability_vectors[d] for d in device_ids])

       # Simulate failures: consider both own availability and correlations
       for i, device_i in enumerate(device_ids):
        timestep = 0
        while timestep <  len(availability_vectors[device_i]):
         host_i = self.net.get(device_i)
         switch_name_i = self.host_to_switch.get(host_i.name, "Unknown")
         #print(f"switch_name_{i}: ", switch_name_i)
         status_i = availability_vectors[device_i][timestep] if timestep < len(availability_vectors[device_i]) else 0

         # 1️⃣ Random failures
         if failure_mode in ["random"] and timestep  == 0:
            if i == 0:
               print("failure mode: ", failure_mode)
            if random.random() < fail_prob and device_i not in self.failed_nodes:
                self.failed_nodes.append(device_i)
                self.failed_nodes_per_switch[switch_name_i].append((device_i, time.time()))
                print(f"❌ [Random] {device_i} failed randomly at step {timestep}")

         # 2️⃣ Correlated failures
         if failure_mode in ["correlated"]:
            if i == 0:
               print("failure mode: ", failure_mode)
            for j, device_j in enumerate(device_ids):
                if i == j or device_j in self.failed_nodes:
                    continue
                corr = matrix[i][j]
                status_j = availability_vectors[device_j][timestep] if timestep < len(availability_vectors[device_j]) else 0
                if corr >= corr_threshold and status_j == 0:
                    host_j = self.net.get(device_j)
                    switch_name_j = self.host_to_switch.get(host_j.name, "Unknown")
                    #print(f"switch_name_{j}: ", switch_name_j)
                    self.failed_nodes.append(device_j)
                    self.failed_nodes_per_switch[switch_name_j].append((device_j, time.time()))
                    print(f"❌ [Correlated] {device_j} failed due to correlation with {device_i} (ρ={corr:.2f})")
         timestep += 1
          
       # ✅ Active hosts = not failed
       active_hosts = [h for h in self.net.hosts if h.name not in self.failed_nodes]
       print(f"\n✅ Active Hosts at timestep {timestep}: {[h.name for h in active_hosts]}")
       print(f"❌ Failed Hosts: {self.failed_nodes}")


       return active_hosts


    def simulate_failures(self, availability_vectors, corr_threshold=0.8, fail_prob=0.3, failure_mode = "random", label_map = None):
       """Evaluate classification accuracy using available hosts."""
       print("🔎 Starting Failure Simulation: Mode =", failure_mode)

       # Evaluate classification accuracy over available (non-failed) hosts
       active_hosts = self.simulate_active_hosts(availability_vectors, corr_threshold=corr_threshold, fail_prob=fail_prob, failure_mode = failure_mode)
       print(f"Active Hosts: {[h.name for h in active_hosts]}")
       print(f"Failed Hosts: {self.failed_nodes}")

       if not active_hosts:
         print("⚠️ All nodes failed. Cannot evaluate.")
         return 0.0

       correct_predictions = 0
       total_predictions = 0

       # Evaluate model accuracy with respect to class presence
       accuracy = self.evaluate_accuracy(active_hosts, self.sample_images, label_map)

      # with torch.no_grad():
       #  for host, (image, label) in zip(active_hosts, self.sample_images):
        #    predicted = host.predict_failure(image)
         #   correct_predictions += (predicted == label).sum().item()
          #  total_predictions += label.size(0)

      # accuracy = (correct_predictions / total_predictions) * 100    
      # print(f"📊 System Accuracy (excluding failed nodes): {accuracy:.2f}%")
       return accuracy


    def evaluate_accuracy(self, active_hosts, sample_images, label_map):
      if not active_hosts:
        print("⚠️ All nodes failed. Cannot evaluate.")
        return 0.0

      correct_predictions = 0
      total_predictions = 0

      class_coverage = set()  # Classes available in current round
      class_counts = defaultdict(int)  # How many times each class appears
      class_correct = defaultdict(int)  # How many times each class is predicted correctly
      class_missed_by_unavailable = set()

      with torch.no_grad():
#        for host, (image, label) in zip(active_hosts, self.sample_images):
#          predicted = host.predict_failure(image)

        for image, label in self.sample_images:
          label_val = label.view(-1)[0].item()

          # Find active hosts trained on this label
          suitable_hosts = [h for h in active_hosts if label_val in label_map.get(h.name, [])]
          if not suitable_hosts:
             print(f"⚠️ No available host trained for label {label_val}. Skipping...")
             continue

          # Pick a random host among suitable ones
          host = random.choice(suitable_hosts)
          predicted = host.predict_failure(image)
          # Flatten tensors to scalars
          label_val = label.view(-1)[0].item()
          pred_val = predicted.view(-1)[0].item()

          # Track class coverage and accuracy
          assigned_classes = label_map.get(host.name, [])
          class_coverage.update(assigned_classes)

          if label_val in assigned_classes:
            class_counts[label_val] += 1
            if pred_val == label_val:
              correct_predictions += 1
              class_correct[label_val] += 1
            total_predictions += 1
          else:
            print(f"⚠️ Host {host.name} evaluated label {label_val} not in its assigned classes {assigned_classes}")

      # Calculate and print accuracy
      accuracy = (correct_predictions / total_predictions) * 100 if total_predictions else 0
      print(f"📊 System Accuracy (excluding failed nodes): {accuracy:.2f}%")

      # Check for unrepresented classes in current active host set
      all_classes = set(range(10))
      missing_classes = all_classes - class_coverage
      print(f"❌ Missing label classes due to failure: {sorted(missing_classes)}")

      # Report per-class accuracy
      print("\n📈 Per-Class Accuracy (for present classes):")
      for c in sorted(class_counts.keys()):
        acc = 100 * class_correct[c] / class_counts[c]
        print(f"  Class {c}: {acc:.2f}% ({class_correct[c]}/{class_counts[c]})")

      return accuracy


    def cleanup(self):
        """Clean up the network."""
        self.net.stop()
        print("🧹 Network stopped and cleaned up.")

if __name__ == "__main__":
    topology_provider = TopologyProvider(
        device_names=['h'],
        model_name='resnet',
        num_workers=10,
        num_epochs=5)
    topology = TopologyProvider(
        device_names=['h'],
        num_workers=10,
        link_latency=5,
        link_loss=5,
        model_name='resnet',
        num_epochs=0,
        failure_probability=0,
        corr_failure_probability=0.3,
        topology_provider = topology_provider
    )  
    topology.setup()

    corr_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    fail_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    availability_vectors = topology.extract_availability_vectors()
    label_map = topology.assign_labels_to_workers(labels_per_worker=1)
    row1 = []
    row2 = []

    for fp in fail_probs:
        acc = topology.simulate_failures(
            availability_vectors,
            corr_threshold=0.6,
            fail_prob=fp,
            failure_mode="random",  # or "random" or "correlated"
            label_map = label_map
        )
        row1.append(acc)
    print("Running for fail_prob=[0.1, 0.3, 0.5, 0.7]", row1)
     
    for corr_th in corr_thresholds:
        acc = topology.simulate_failures(
            availability_vectors,
            corr_threshold=corr_th,
            fail_prob=0.1,
            failure_mode="correlated",  # or "random" or "correlated"
            label_map = label_map
        )
        row2.append(acc)
    print("Running for corr_thresholds = [0.6, 0.7, 0.8, 0.9]", row2)

    topology.cleanup()
