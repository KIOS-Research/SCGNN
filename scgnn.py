import os
import pickle
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wntr
import h5py
from typing import Optional, List, Tuple, Dict
import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from pathlib import Path


class AttentionBasedGraph(nn.Module):
    """
    Learn adjacency via scaled dot-product attention with top-k edge selection
    Computes similarity-based adjacency matrix between sensors
    Only keeps top-k strongest connections per node for sparsity
    """
    def __init__(self, num_sensors: int, feature_dim: int, k_neighbors: int = 10):
        super().__init__()
        self.num_sensors = num_sensors
        self.feature_dim = feature_dim
        self.k_neighbors = k_neighbors
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.last_mask = None
    
    def forward(self, sensor_features):
        """
        Args:
            sensor_features: [batch_size, num_sensors, feature_dim]
        Returns:
            adjacency: [batch_size, num_sensors, num_sensors]
        """
        batch_size, N, _ = sensor_features.shape
        
        # Compute Q, K
        # [batch_size, num_sensors, feature_dim]
        Q = self.query(sensor_features)
        K = self.key(sensor_features)
        
        # Compute attention scores
        # [batch_size, num_sensors, num_sensors]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.feature_dim, dtype=torch.float32, device=Q.device)
        )
        
        self_loop_weight = 1.0
        eye_mask = torch.eye(N, device=scores.device).unsqueeze(0).expand(batch_size, -1, -1)
        scores = scores + eye_mask * self_loop_weight
        
        k = min(self.k_neighbors, N - 1)
        
        # Get top-k values and indices for each row (each node's neighbors)
        topk_values, topk_indices = torch.topk(scores, k=k, dim=-1)
        
        # Create mask: only top-k edges are kept
        mask = torch.zeros_like(scores, dtype=torch.bool)
        batch_idx = torch.arange(batch_size, device=scores.device).view(-1, 1, 1).expand(-1, N, k)
        node_idx = torch.arange(N, device=scores.device).view(1, -1, 1).expand(batch_size, -1, k)
        mask[batch_idx, node_idx, topk_indices] = True
        
        # Ensure self-loops are always included
        mask = mask | eye_mask.bool()
        
        # Make adjacency symmetric (undirected graph)
        # If either i→j or j→i is in top-k, include both directions
        mask_symmetric = mask | mask.transpose(-2, -1)
        
        # Apply symmetric mask to scores
        masked_scores = scores.clone()
        masked_scores[~mask_symmetric] = -1e9

        # Store binary mask for external analysis (before softmax)
        self.last_mask = mask_symmetric.detach()

        # Apply softmax row-wise to get adjacency matrix
        # [batch_size, num_sensors, num_sensors]
        adjacency = torch.softmax(masked_scores, dim=-1)
        
        return adjacency


class GCNLayer(nn.Module):
    """
    Graph Convolutional Layer (GCN)
    Simple message passing with adjacency-based aggregation
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        
        # Transformation matrix
        self.weight = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, h, adj):
        """
        Args:
            h: Node features [batch_size, num_nodes, in_features]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
        Returns:
            out: Updated features [batch_size, num_nodes, out_features]
        """
        batch_size, N, _ = h.shape
        
        # Linear transformation: h @ W
        # [batch_size, num_nodes, in_features] @ [in_features, out_features]
        # -> [batch_size, num_nodes, out_features]
        h_transformed = torch.matmul(h, self.weight) + self.bias
        
        # Aggregate from neighbors: adj @ h_transformed
        # [batch_size, num_nodes, num_nodes] @ [batch_size, num_nodes, out_features]
        # -> [batch_size, num_nodes, out_features]
        h_aggregated = torch.matmul(adj, h_transformed)
        
        # Apply dropout and activation
        h_aggregated = self.dropout_layer(h_aggregated)
        h_aggregated = F.relu(h_aggregated)
        
        # Layer normalization with residual connection
        if self.in_features == self.out_features:
            out = self.layer_norm(h_aggregated + h)
        else:
            out = self.layer_norm(h_aggregated)
        
        return out


class SCGNN(nn.Module):
    def __init__(self,
                 num_sensors: int,
                 time_steps: int,
                 num_classes: int,
                 hidden_dim: int = 128,
                 num_gcn_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 learn_adjacency: bool = True,
                 top_k: int = 10):
        super().__init__()
        
        self.num_sensors = num_sensors
        self.time_steps = time_steps
        self.num_classes = num_classes
        self.learn_adjacency = learn_adjacency
        
        # Feature extraction from time series using 1D CNN (process each sensor independently)
        # Input: [batch_size * num_sensors, 1, time_steps]
        # Output: [batch_size * num_sensors, hidden_dim]
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling to get [batch, hidden_dim, 1]
        )
        
        # Learn adjacency matrix
        if learn_adjacency:
            self.graph_learner = AttentionBasedGraph(
                num_sensors=num_sensors,
                feature_dim=hidden_dim,
                k_neighbors=top_k
            )
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        
        # First GCN layer
        self.gcn_layers.append(
            GCNLayer(hidden_dim, hidden_dim, dropout=dropout)
        )
        
        # Middle GCN layers
        for _ in range(num_gcn_layers - 2):
            self.gcn_layers.append(
                GCNLayer(hidden_dim, hidden_dim, dropout=dropout)
            )
        
        # Last GCN layer
        self.gcn_layers.append(
            GCNLayer(hidden_dim, hidden_dim, dropout=dropout)
        )
        
        # Graph pooling and classification
        self.global_pool = nn.Sequential(
            nn.Linear(num_sensors * hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, adj=None):
        """
        Args:
            x: Sensor readings [batch_size, num_sensors, time_steps]
            adj: Pre-defined adjacency [batch_size, num_sensors, num_sensors] (optional)
        Returns:
            logits: [batch_size, num_classes]
            learned_adj: Learned adjacency matrix (if learn_adjacency=True)
        """
        batch_size = x.shape[0]
        
        # Extract features from time series independently for each sensor using 1D CNN
        # Reshape: [batch_size, num_sensors, time_steps] -> [batch_size * num_sensors, 1, time_steps]
        x_reshaped = x.reshape(batch_size * self.num_sensors, 1, -1)
        
        # Process each sensor's time series independently through 1D CNN
        # Output: [batch_size * num_sensors, hidden_dim, 1]
        features = self.temporal_encoder(x_reshaped)
        
        # Remove temporal dimension and reshape back to separate sensors
        # [batch_size * num_sensors, hidden_dim, 1] -> [batch_size * num_sensors, hidden_dim]
        features = features.squeeze(-1)
        
        # [batch_size * num_sensors, hidden_dim] -> [batch_size, num_sensors, hidden_dim]
        sensor_features = features.reshape(batch_size, self.num_sensors, -1)
        
        # Learn or use adjacency
        if self.learn_adjacency:
            adj = self.graph_learner(sensor_features)
        elif adj is None:
            # Use normalized fully connected graph (each row sums to 1)
            adj = torch.ones(batch_size, self.num_sensors, self.num_sensors,
                           device=x.device) / self.num_sensors
            # Use random adjacency
            # adj = torch.rand(batch_size, self.num_sensors, self.num_sensors, device=x.device)
            # adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Apply GCN layers
        h = sensor_features
        for gcn_layer in self.gcn_layers:
            h = gcn_layer(h, adj)
        
        # Global pooling
        h_flat = h.reshape(batch_size, -1)
        h_pooled = self.global_pool(h_flat)
        
        # Classification
        logits = self.classifier(h_pooled)
        
        return logits, adj


class ContrastiveAdjacencyLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, adjacency, labels):
        """
        Args:
            features: Sensor features [batch_size, num_sensors, hidden_dim]
            adjacency: Learned adjacency [batch_size, num_sensors, num_sensors]
            labels: Contamination source labels [batch_size]
        Returns:
            loss: Contrastive loss scalar
        """
        batch_size, num_sensors, hidden_dim = features.shape
        aggregated_features = torch.bmm(adjacency, features)
        pooled_features = aggregated_features.mean(dim=1)
        pooled_features = F.normalize(pooled_features, p=2, dim=1)
        similarity_matrix = torch.mm(pooled_features, pooled_features.t()) / self.temperature
        labels = labels.unsqueeze(1)  # [batch, 1]
        positive_mask = (labels == labels.t()).float()  # [batch, batch]
        
        mask_eye = torch.eye(batch_size, device=features.device)
        positive_mask = positive_mask * (1 - mask_eye)

        num_positives_per_sample = positive_mask.sum(dim=1)  # [batch]
        has_positive = num_positives_per_sample > 0  # [batch]

        if not has_positive.any():
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = (exp_sim * (1 - mask_eye)).sum(dim=1, keepdim=True)  # [batch, 1]
        
        log_denominator = torch.log(sum_exp_sim + 1e-8)
        log_prob_matrix = similarity_matrix - log_denominator  # [batch, batch]
        log_prob_positives = log_prob_matrix * positive_mask  # [batch, batch]
        loss_per_sample = -log_prob_positives.sum(dim=1) / (num_positives_per_sample + 1e-8)  # [batch]
        loss = loss_per_sample[has_positive].mean()

        loss = torch.clamp(loss, 0.0, 100.0)
        
        return loss


class SensorContaminationDataset(Dataset):
    def __init__(self,
                 network: str,
                 partitions: List[List[str]],
                 sensor_nodes: Optional[List[str]] = None,
                 data_dir: Optional[str] = None,
                 max_scenarios: Optional[int] = None,
                 preload_data: bool = True,
                 noise_level: float = 0.01,
                 logger_adapter=None):
        """
        Args:
            partitions: List of node partitions for classification
            sensor_nodes: List of sensor node names (if None, use all sensors from scenarios)
            data_dir: Directory containing scenario_*.mat files for the selected network
            max_scenarios: Maximum scenarios to load
            preload_data: Whether to preload all data into memory
            noise_level: Noise level for data augmentation
            logger_adapter: Logger adapter for logging
        """
        self.network = network
        self.data_dir = data_dir or os.path.join('data', network)
        self.partitions = partitions
        self.sensor_nodes = sensor_nodes
        self.max_scenarios = max_scenarios
        self.preload_data = preload_data
        self.noise_level = noise_level
        self.logger_adapter = logger_adapter
        
        # For caching preprocessed data
        self.cached_data = []
        self.cached_labels = []
        
        # Find all scenario files (.mat)
        self.scenario_files = []
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}. "
                "Download the Zenodo dataset and pass --data_dir if needed."
            )
        for file in os.listdir(self.data_dir):
            if file.startswith('scenario_') and file.endswith('.mat'):
                self.scenario_files.append(file)
        
        # Sort files by scenario number
        self.scenario_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        # Limit scenarios if specified
        if max_scenarios and max_scenarios > 0:
            self.scenario_files = self.scenario_files[:max_scenarios]
        
        # Create node to partition mapping
        self.node_to_partition = {}
        for partition_id, nodes in enumerate(partitions):
            for node in nodes:
                self.node_to_partition[node] = partition_id
        
        if self.logger_adapter:
            self.logger_adapter.info(f"SensorContaminationDataset initialized with {len(self.scenario_files)} scenarios")
        else:
            print(f"SensorContaminationDataset initialized with {len(self.scenario_files)} scenarios")
        
        # Pre-load data if enabled
        if self.preload_data:
            cache_file = os.path.join(self.data_dir, 'cached_sensor_data.pt')
            if os.path.exists(cache_file):
                if self.logger_adapter:
                    self.logger_adapter.info("Loading cached pre-loaded data from disk...")
                else:
                    print("Loading cached pre-loaded data from disk...")
                self.cached_data, self.cached_labels = torch.load(cache_file, weights_only=False)
                if self.logger_adapter:
                    self.logger_adapter.info(f"Loaded {len(self.cached_data)} samples from cache")
                else:
                    print(f"Loaded {len(self.cached_data)} samples from cache")
            else:
                self._preload_sensor_data()
    
    def _decode_string_from_mat(self, data):
        """Decode string data from MATLAB .mat files"""
        if hasattr(data, 'flatten'):
            return ''.join(chr(int(c)) for c in data.flatten())
        else:
            return str(data)
    
    def _load_mat_scenario(self, file_path: str):
        """Load scenario from .mat file"""
        with h5py.File(file_path, 'r') as f:
            scenario_data = {}
            
            # Data is under 'scenario_data' group
            data_group = f['scenario_data']
            
            # Extract contamination node
            contamination_node_data = data_group['contamination_node'][:]
            scenario_data['contamination_node'] = self._decode_string_from_mat(contamination_node_data)
            
            # Extract contamination intensity (if available)
            if 'contamination_intensity' in data_group:
                scenario_data['contamination_intensity'] = float(data_group['contamination_intensity'][0, 0])
            
            # Extract sensor indices (1-based)
            scenario_data['sensor_indices'] = data_group['sensor_indices'][:].flatten().astype(int)
            
            # Extract sensor locations
            sensor_locations_refs = data_group['sensor_locations']
            scenario_data['sensor_locations'] = []
            
            # Handle different possible structures
            if len(sensor_locations_refs.shape) == 2:
                for i in range(sensor_locations_refs.shape[0]):
                    for j in range(sensor_locations_refs.shape[1]):
                        ref = sensor_locations_refs[i, j]
                        if isinstance(ref, h5py.Reference):
                            sensor_name_data = f[ref][:]
                            sensor_name = self._decode_string_from_mat(sensor_name_data)
                            scenario_data['sensor_locations'].append(sensor_name)
                        elif ref != 0:
                            scenario_data['sensor_locations'].append(str(ref))
            else:
                for ref in sensor_locations_refs:
                    if isinstance(ref, h5py.Reference):
                        sensor_name_data = f[ref][:]
                        sensor_name = self._decode_string_from_mat(sensor_name_data)
                        scenario_data['sensor_locations'].append(sensor_name)
                    elif ref != 0:
                        scenario_data['sensor_locations'].append(str(ref))
            
            # Extract concentrations data
            concentrations_group = data_group['concentrations']
            scenario_data['concentrations'] = {}
            
            # Get all sensor fields from concentrations group
            for field_name in concentrations_group.keys():
                if field_name.startswith('sensor_'):
                    # Extract sensor index from field name (0-based)
                    sensor_key = int(field_name.split('_')[1])
                    concentration_data = concentrations_group[field_name][0, :].flatten()
                    scenario_data['concentrations'][sensor_key] = concentration_data
            
            return scenario_data
    
    def _preload_sensor_data(self):
        """Pre-load and cache all sensor data for faster training"""
        if self.logger_adapter:
            self.logger_adapter.info("Pre-loading sensor data into memory...")
        else:
            print("Pre-loading sensor data into memory...")
        
        for idx in range(len(self.scenario_files)):
            if (idx + 1) % 100 == 0:
                msg = f"  Pre-loaded {idx + 1}/{len(self.scenario_files)} scenarios..."
                if self.logger_adapter:
                    self.logger_adapter.info(msg)
                else:
                    print(msg)
            
            file_path = os.path.join(self.data_dir, self.scenario_files[idx])
            scenario_data = self._load_mat_scenario(file_path)
            
            contamination_node = scenario_data['contamination_node']
            source_partition = self.node_to_partition.get(contamination_node, -1)
            
            if source_partition == -1:
                continue  # Skip if source not in any partition
            
            # Process sensor data
            sensor_tensor = self._create_sensor_only_data(scenario_data)
            
            if sensor_tensor is not None:
                self.cached_data.append(sensor_tensor)
                self.cached_labels.append(source_partition)
        
        # Save cache to disk
        cache_file = os.path.join(self.data_dir, 'cached_sensor_data.pt')
        torch.save((self.cached_data, self.cached_labels), cache_file)
        
        if self.logger_adapter:
            self.logger_adapter.info(f"Pre-loaded {len(self.cached_data)} samples into memory")
        else:
            print(f"Pre-loaded {len(self.cached_data)} samples into memory")
    
    def _create_sensor_only_data(self, scenario_data):
        """Create sensor-only feature tensor from scenario data"""
        concentrations = scenario_data['concentrations']
        sensor_locations = scenario_data['sensor_locations']
        sensor_indices = scenario_data['sensor_indices']
        
        if len(sensor_locations) != len(sensor_indices):
            if self.logger_adapter:
                self.logger_adapter.warning(f"Mismatch in sensor locations and indices lengths: {len(sensor_locations)} vs {len(sensor_indices)}")
            return None
        
        # Get time steps from concentration data
        if not concentrations:
            if self.logger_adapter:
                self.logger_adapter.warning("No concentration data found")
            return None
        
        time_steps = len(list(concentrations.values())[0])
        
        # Create features for sensor nodes
        sensor_features = []
        
        # If specific sensor nodes are specified, filter to those
        if self.sensor_nodes is not None:
            for sensor_name in self.sensor_nodes:
                if sensor_name in sensor_locations:
                    sensor_idx_in_list = sensor_locations.index(sensor_name)
                    matlab_sensor_idx = sensor_indices[sensor_idx_in_list]
                    sensor_key = matlab_sensor_idx - 1  # Convert to 0-based
                    
                    if sensor_key in concentrations:
                        sensor_data = concentrations[sensor_key]
                        if isinstance(sensor_data, np.ndarray):
                            conc_series = sensor_data.reshape(-1)
                        else:
                            conc_series = np.array(sensor_data).reshape(-1)
                        
                        # Normalize concentration values between 0 and 1
                        conc_min = np.min(conc_series)
                        conc_max = np.max(conc_series)
                        if conc_max > conc_min:
                            conc_series = (conc_series - conc_min) / (conc_max - conc_min)
                        
                        sensor_features.append(conc_series)
                    else:
                        # Sensor location but no data - use zeros
                        sensor_features.append(np.zeros(time_steps))
                else:
                    # Sensor not in this scenario - use zeros
                    sensor_features.append(np.zeros(time_steps))
        else:
            # Use all sensors from scenario
            for i, sensor_name in enumerate(sensor_locations):
                matlab_sensor_idx = sensor_indices[i]
                sensor_key = matlab_sensor_idx - 1  # Convert to 0-based
                
                if sensor_key in concentrations:
                    sensor_data = concentrations[sensor_key]
                    if isinstance(sensor_data, np.ndarray):
                        conc_series = sensor_data.reshape(-1)
                    else:
                        conc_series = np.array(sensor_data).reshape(-1)
                    
                    # Normalize concentration values between 0 and 1
                    conc_min = np.min(conc_series)
                    conc_max = np.max(conc_series)
                    if conc_max > conc_min:
                        conc_series = (conc_series - conc_min) / (conc_max - conc_min)
                    
                    sensor_features.append(conc_series)
                else:
                    # Sensor location but no data - use zeros
                    sensor_features.append(np.zeros(time_steps))
        
        # Convert to tensor
        sensor_features = np.array(sensor_features)  # [num_sensors, time_steps]
        sensor_tensor = torch.tensor(sensor_features, dtype=torch.float32)
        
        return sensor_tensor
    
    def __len__(self):
        """Return dataset size"""
        if self.preload_data and self.cached_data:
            return len(self.cached_data)
        return len(self.scenario_files)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Returns:
            tuple: (sensor_tensor, label)
                - sensor_tensor: [num_sensors, time_steps]
                - label: int (partition ID)
        """
        # Use cached data if available
        if self.preload_data and self.cached_data:
            sensor_tensor = self.cached_data[idx].clone()
            label = self.cached_labels[idx]
            
            # Add noise for data augmentation
            if self.noise_level > 0:
                noise = torch.randn_like(sensor_tensor) * self.noise_level
                sensor_tensor = torch.clamp(sensor_tensor + noise, 0, 1)
            
            return sensor_tensor, label
        
        else:
            # Load data on-the-fly
            file_path = os.path.join(self.data_dir, self.scenario_files[idx])
            scenario_data = self._load_mat_scenario(file_path)
            
            contamination_node = scenario_data['contamination_node']
            
            # Get partition label
            source_partition = self.node_to_partition.get(contamination_node, -1)
            if source_partition == -1:
                raise ValueError(f"Source node {contamination_node} not found in any partition")
            
            sensor_tensor = self._create_sensor_only_data(scenario_data)
            
            if sensor_tensor is None:
                raise ValueError(f"Failed to create sensor data for scenario {idx}")
            
            return sensor_tensor, source_partition


class SCGNNTrainer:
    """
    Trainer for SCGNN.
    """
    def __init__(self,
                 model: SCGNN,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 device: torch.device,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 save_dir: str = './results',
                 contrastive_lambda: float = 0.0,
                 contrastive_temp: float = 0.1):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        self.contrastive_lambda = contrastive_lambda
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Optimizer and loss
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Contrastive loss for adjacency learning
        if contrastive_lambda > 0:
            self.contrastive_criterion = ContrastiveAdjacencyLoss(
                temperature=contrastive_temp
            )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_acc': 0.0
        }
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_contrastive_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, adj = self.model(x)
            
            # Classification loss
            cls_loss = self.criterion(logits, y)
            
            # Contrastive loss for adjacency (if enabled)
            if self.contrastive_lambda > 0 and self.model.learn_adjacency:
                # Get sensor features before GCN layers
                # Re-extract features (same as in model forward)
                batch_size = x.shape[0]
                x_reshaped = x.reshape(batch_size * self.model.num_sensors, 1, -1)
                features = self.model.temporal_encoder(x_reshaped)
                features = features.squeeze(-1)
                sensor_features = features.reshape(batch_size, self.model.num_sensors, -1)
                
                contrastive_loss = self.contrastive_criterion(sensor_features, adj, y)
                loss = cls_loss + self.contrastive_lambda * contrastive_loss
                
                total_contrastive_loss += contrastive_loss.item()
            else:
                loss = cls_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
        avg_loss = total_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)
        avg_contrastive_loss = total_contrastive_loss / len(self.train_loader) if self.contrastive_lambda > 0 else 0.0
        acc = accuracy_score(all_labels, all_preds)
        
        return avg_loss, acc, avg_cls_loss, avg_contrastive_loss
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                logits, adj = self.model(x)
                loss = self.criterion(logits, y)
                
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        acc = accuracy_score(all_labels, all_preds)
        
        return avg_loss, acc
    
    def test(self):
        """Test model"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_adjacencies = []
        
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                logits, adj = self.model(x)
                preds = logits.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_adjacencies.append(adj.cpu().numpy())
        
        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        results = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_preds,
            'labels': all_labels,
            'adjacencies': np.concatenate(all_adjacencies, axis=0)
        }
        
        return results
    
    def train(self, num_epochs: int, early_stopping_patience: int = 15):
        """
        Train the model
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
        """
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"\nTraining SCGNN for {num_epochs} epochs...")
        print("=" * 70)
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc, cls_loss, contrastive_loss = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_acc)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            if self.contrastive_lambda > 0:
                print(f"Epoch {epoch+1:03d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} (cls: {cls_loss:.4f}, contr: {contrastive_loss:.4f}) | "
                      f"Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1:03d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join(self.save_dir, 'best_model.pth'))
                print(f"  → New best model saved (Val Acc: {val_acc:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        print("\n" + "=" * 70)
        print(f"Training completed! Best Val Acc: {best_val_acc:.4f}")
        
        # Load best model and test
        checkpoint = torch.load(os.path.join(self.save_dir, 'best_model.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_results = self.test()
        self.history['test_acc'] = test_results['accuracy']
        
        print(f"\nTest Results:")
        print(f"  Accuracy:  {test_results['accuracy']:.4f}")
        print(f"  Precision: {test_results['precision']:.4f}")
        print(f"  Recall:    {test_results['recall']:.4f}")
        print(f"  F1-Score:  {test_results['f1']:.4f}")
        
        # Save results
        self._save_results(test_results)
        
        return test_results
    
    def _save_results(self, test_results: Dict):
        """Save training history and test results"""
        # Save history with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        history_file = os.path.join(self.save_dir, f'history_{timestamp}.json')
        with open(history_file, 'w') as f:
            json.dump({k: v for k, v in self.history.items() 
                      if k != 'predictions' and k != 'labels'}, f, indent=2)
        
        # Load existing test results if file exists
        test_results_file = os.path.join(self.save_dir, 'test_results.json')
        if os.path.exists(test_results_file):
            with open(test_results_file, 'r') as f:
                all_results = json.load(f)
            # Ensure it's a list
            if not isinstance(all_results, list):
                all_results = [all_results]
        else:
            all_results = []
        
        # Prepare new result entry with metadata
        results_to_save = {
            'timestamp': timestamp,
            'learn_adjacency': self.model.learn_adjacency,
            'accuracy': test_results['accuracy'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'f1': test_results['f1']
        }
        
        # Append new results
        all_results.append(results_to_save)
        
        # Save updated results
        with open(test_results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Save confusion matrix
        self._plot_confusion_matrix(test_results['labels'], test_results['predictions'])
        
        # Save learned adjacency visualization
        if len(test_results['adjacencies']) > 0:
            self._plot_learned_adjacency(test_results['adjacencies'][0])
        
        # Save training curves
        self._plot_training_curves()
    
    def _plot_confusion_matrix(self, labels, predictions):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
    
    def _plot_learned_adjacency(self, adjacency):
        """Plot learned sensor adjacency matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(adjacency, cmap='viridis', cbar=True)
        plt.title('Learned Sensor Adjacency Matrix')
        plt.xlabel('Sensor Index')
        plt.ylabel('Sensor Index')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'learned_adjacency.png'), dpi=300)
        plt.close()
    
    def _plot_training_curves(self):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.axhline(y=self.history['test_acc'], color='r', 
                   linestyle='--', label=f'Test Acc ({self.history["test_acc"]:.4f})')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300)
        plt.close()


def load_partitions(partition_file: str) -> List[List[str]]:
    """Load partitions from JSON file"""
    with open(partition_file, 'r') as f:
        data = json.load(f)
    return data['partitions']


def main():
    parser = argparse.ArgumentParser(
        description='SCGNN contamination source isolation'
    )
    
    # Data arguments
    parser.add_argument('--network', type=str, default='ZJ')
    parser.add_argument('--max_scenarios', type=int, default=None,
                       help='Maximum number of scenarios to load')
    parser.add_argument('--preload_data', action='store_true',
                       help='Preload all data into memory for faster training')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory containing scenario_*.mat files. Defaults to data/<network>.')
    parser.add_argument('--sensor_file', type=str, default=None,
                       help='Optional file containing sensor node names in training order.')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension size (default: 128)')
    parser.add_argument('--num_gcn_layers', type=int, default=2,
                       help='Number of GCN layers (default: 2)')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads for adjacency learning (default: 4)')
    parser.add_argument('--top_k', type=int, default=7,
                       help='Number of top edges to keep per sensor in learned adjacency (default: 7)')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--no_learn_adj', action='store_true',
                       help='Disable adjacency learning')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--early_stopping', type=int, default=30)
    parser.add_argument('--contrastive_lambda', type=float, default=0.7)
    parser.add_argument('--contrastive_temp', type=float, default=0.5)
    
    # Other arguments
    parser.add_argument('--save_dir', type=str, default='./results/scgnn')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load sensor nodes (optional)
    sensor_nodes = None
    data_dir = args.data_dir or os.path.join('data', args.network)
    sensor_file = args.sensor_file or os.path.join(data_dir, 'sensors.txt')
    if os.path.exists(sensor_file):
        print(f"\nLoading sensor nodes from {sensor_file}...")
        with open(sensor_file, 'r') as f:
            sensor_nodes = f.read().split()
        print(f"Loaded {len(sensor_nodes)} sensors")
    else:
        print("\nNo sensor file found. Will use all sensors from scenarios.")
    
    # Load partitions
    partition_file = f'./networks/{args.network}_partitions_louvain.json'
    partitions = load_partitions(partition_file)
    num_classes = len(partitions)
    print(f"Loaded {num_classes} partitions")

    # Create dataset
    print(f"\nCreating dataset...")
    dataset = SensorContaminationDataset(
        network=args.network,
        partitions=partitions,
        sensor_nodes=sensor_nodes,
        data_dir=data_dir,
        max_scenarios=args.max_scenarios,
        preload_data=args.preload_data,
        logger_adapter=None
    )
    
    # Determine actual number of sensors from first sample
    if len(dataset) > 0:
        sample_data, _ = dataset[0]
        actual_num_sensors = sample_data.shape[0]
        actual_time_steps = sample_data.shape[1]
        print(f"\nActual sensors in dataset: {actual_num_sensors}")
        print(f"Actual time steps in dataset: {actual_time_steps}")
    else:
        raise ValueError("Dataset is empty!")
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)
    
    # Create model
    print(f"\nCreating SCGNN model...")
    print(f'Using learned adjacency: {not args.no_learn_adj}')
    model = SCGNN(
        num_sensors=actual_num_sensors,
        time_steps=actual_time_steps,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        num_gcn_layers=args.num_gcn_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        learn_adjacency=not args.no_learn_adj,
        top_k=args.top_k
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config['num_sensors'] = actual_num_sensors
    config['time_steps'] = actual_time_steps
    config['num_classes'] = num_classes
    config['total_scenarios'] = total_size
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    trainer = SCGNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        contrastive_lambda=args.contrastive_lambda,
        contrastive_temp=args.contrastive_temp
    )
    
    # Train
    test_results = trainer.train(
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping
    )
    
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print(f"Results saved to: {args.save_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
