import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data
except ImportError:
    logging.warning("PyTorch or PyTorch Geometric not installed. Using simulation mode.")

class GraphNeuralNetwork:
    """Graph Neural Network for cryptocurrency price prediction.
    
    This model constructs a graph representation of OHLCV data where:
    - Nodes represent individual candles
    - Edges connect temporally adjacent candles and similar price patterns
    - Node features include OHLCV values and technical indicators
    """
    
    def __init__(self, hidden_channels=64, num_layers=3):
        """Initialize the GNN model.
        
        Args:
            hidden_channels: Number of hidden channels in GNN layers
            num_layers: Number of GNN layers
        """
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.logger = logging.getLogger(__name__)
        
        try:
            # Check if PyTorch and PyTorch Geometric are available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self.device}")
            
            # Create the GNN model
            self.model = self._create_model()
            
        except NameError:
            self.logger.warning("Running in simulation mode without PyTorch/PyTorch Geometric")
            self.device = None
            self.model = None
    
    def _create_model(self):
        """Create the GNN model architecture."""
        if self.device is not None:
            class GNNModel(nn.Module):
                def __init__(self, input_dim, hidden_channels, num_layers):
                    super(GNNModel, self).__init__()
                    
                    # Input embedding
                    self.embedding = nn.Linear(input_dim, hidden_channels)
                    
                    # GNN layers
                    self.convs = nn.ModuleList()
                    self.batch_norms = nn.ModuleList()
                    
                    # First layer
                    self.convs.append(GCNConv(hidden_channels, hidden_channels))
                    self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
                    
                    # Additional layers
                    for _ in range(num_layers - 1):
                        self.convs.append(GCNConv(hidden_channels, hidden_channels))
                        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
                    
                    # Output layers
                    self.fc_out = nn.Linear(hidden_channels, 5)  # Predict next OHLCV
                    self.signal_head = nn.Linear(hidden_channels, 3)  # BUY, SELL, HOLD probabilities
                
                def forward(self, x, edge_index, batch):
                    # Initial embedding
                    x = self.embedding(x)
                    x = F.gelu(x)
                    
                    # Apply GNN layers with residual connections
                    for i, conv in enumerate(self.convs):
                        identity = x
                        x = conv(x, edge_index)
                        x = self.batch_norms[i](x)
                        x = F.gelu(x)
                        x = F.dropout(x, p=0.2, training=self.training)
                        x = x + identity  # Residual connection
                    
                    # Global pooling
                    x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
                    
                    # Generate predictions
                    next_candle = self.fc_out(x)  # [batch_size, 5]
                    signal_probs = F.softmax(self.signal_head(x), dim=-1)  # [batch_size, 3]
                    
                    # Store node embeddings for visualization
                    node_embeddings = x.detach().cpu().numpy()
                    
                    return next_candle, signal_probs, node_embeddings
            
            model = GNNModel(
                input_dim=5,  # OHLCV
                hidden_channels=self.hidden_channels,
                num_layers=self.num_layers
            ).to(self.device)
            
            return model
        return None
    
    def _construct_graph(self, ohlcv_data: np.ndarray) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Construct a graph from OHLCV data.
        
        Args:
            ohlcv_data: OHLCV data array
            
        Returns:
            Tuple of (node_features, edge_index, batch) or None if in simulation mode
        """
        if self.device is not None:
            # Use the last 100 candles to construct the graph
            seq_length = min(100, ohlcv_data.shape[0])
            recent_data = ohlcv_data[-seq_length:]
            
            # Normalize the data
            mean = np.mean(recent_data, axis=0, keepdims=True)
            std = np.std(recent_data, axis=0, keepdims=True) + 1e-8
            normalized_data = (recent_data - mean) / std
            
            # Node features are the OHLCV values
            node_features = torch.tensor(normalized_data, dtype=torch.float32).to(self.device)
            
            # Create temporal edges (connecting adjacent candles)
            num_nodes = node_features.shape[0]
            source_nodes = torch.arange(0, num_nodes - 1, dtype=torch.long)
            target_nodes = torch.arange(1, num_nodes, dtype=torch.long)
            temporal_edges = torch.stack([source_nodes, target_nodes], dim=0).to(self.device)
            
            # Create similarity edges (connecting candles with similar patterns)
            # For simplicity, we'll connect nodes that have similar closing prices
            similarity_edges = []
            for i in range(num_nodes):
                for j in range(i + 2, min(i + 10, num_nodes)):  # Connect to nearby but not adjacent nodes
                    # If closing prices are similar, add an edge
                    if abs(normalized_data[i, 3] - normalized_data[j, 3]) < 0.5:  # Threshold for similarity
                        similarity_edges.append([i, j])
                        similarity_edges.append([j, i])  # Add in both directions for undirected graph
            
            if similarity_edges:
                similarity_edges = torch.tensor(similarity_edges, dtype=torch.long).t().to(self.device)
                # Combine temporal and similarity edges
                edge_index = torch.cat([temporal_edges, similarity_edges], dim=1)
            else:
                edge_index = temporal_edges
            
            # Batch index (all nodes belong to the same graph in this case)
            batch = torch.zeros(num_nodes, dtype=torch.long).to(self.device)
            
            return node_features, edge_index, batch
        return None
    
    def predict(self, ohlcv_data: np.ndarray) -> Dict[str, Any]:
        """Generate predictions using the GNN model.
        
        Args:
            ohlcv_data: OHLCV data array
            
        Returns:
            Dictionary containing predictions and model metrics
        """
        self.logger.info("Generating GNN predictions")
        
        if self.device is not None and self.model is not None:
            # Real GNN computation
            self.model.eval()
            with torch.no_grad():
                # Construct graph from OHLCV data
                graph_data = self._construct_graph(ohlcv_data)
                
                if graph_data is not None:
                    node_features, edge_index, batch = graph_data
                    
                    # Forward pass
                    next_candle, signal_probs, node_embeddings = self.model(node_features, edge_index, batch)
                    
                    # Convert to numpy for further processing
                    next_candle_np = next_candle.cpu().numpy()[0]  # [5]
                    signal_probs_np = signal_probs.cpu().numpy()[0]  # [3]
                    
                    # Calculate predicted move percentage
                    last_close = ohlcv_data[-1, 3]  # Assuming close is at index 3
                    predicted_close = next_candle_np[3] * np.std(ohlcv_data[:, 3]) + np.mean(ohlcv_data[:, 3])
                    predicted_move_pct = (predicted_close - last_close) / last_close
                    
                    # Determine signal type and confidence
                    buy_prob, sell_prob, hold_prob = signal_probs_np
                    signal_type = "BUY" if buy_prob > sell_prob and buy_prob > hold_prob else \
                                "SELL" if sell_prob > buy_prob and sell_prob > hold_prob else "HOLD"
                    confidence = float(max(buy_prob, sell_prob, hold_prob))
                    
                    # Calculate graph metrics
                    graph_density = edge_index.shape[1] / (node_features.shape[0] * (node_features.shape[0] - 1))
                    
                    return {
                        "buy_probability": float(buy_prob),
                        "sell_probability": float(sell_prob),
                        "hold_probability": float(hold_prob),
                        "signal_type": signal_type,
                        "confidence": confidence,
                        "predicted_move": float(predicted_move_pct),
                        "next_candle": next_candle_np.tolist(),
                        "node_embeddings": node_embeddings.tolist(),
                        "graph_density": float(graph_density),
                        "activation": float(np.mean(node_embeddings))  # Average activation for signal generation
                    }
        
        # Simulation mode when PyTorch not available
        self.logger.info("Using simulated GNN predictions")
        
        # Simulate GNN behavior
        buy_prob = 0.3 + np.random.random() * 0.4  # Random between 0.3 and 0.7
        sell_prob = 0.3 + np.random.random() * 0.4  # Random between 0.3 and 0.7
        
        # Normalize to ensure probabilities sum to 1
        total = buy_prob + sell_prob
        if total > 1.0:
            buy_prob /= total
            sell_prob /= total
        
        hold_prob = 1.0 - buy_prob - sell_prob
        
        # Generate simulated node embeddings
        node_embeddings = np.random.random((1, self.hidden_channels)).tolist()
        
        # Generate simulated predicted move
        predicted_move_pct = (np.random.random() - 0.5) * 0.03  # -1.5% to +1.5%
        
        return {
            "buy_probability": float(buy_prob),
            "sell_probability": float(sell_prob),
            "hold_probability": float(hold_prob),
            "signal_type": "BUY" if buy_prob > sell_prob and buy_prob > hold_prob else \
                          "SELL" if sell_prob > buy_prob and sell_prob > hold_prob else "HOLD",
            "confidence": float(max(buy_prob, sell_prob, hold_prob)),
            "predicted_move": float(predicted_move_pct),
            "next_candle": [0.0, 0.0, 0.0, 0.0, 0.0],  # Placeholder
            "node_embeddings": node_embeddings,
            "graph_density": float(0.1 + np.random.random() * 0.2),  # Random between 0.1 and 0.3
            "activation": float(0.6 + np.random.random() * 0.3)  # Random between 0.6 and 0.9
        }