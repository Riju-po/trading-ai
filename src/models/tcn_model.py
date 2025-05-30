import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    logging.warning("PyTorch not installed. Using simulation mode.")

class TemporalBlock(nn.Module):
    """Temporal block for TCN consisting of dilated causal convolutions."""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.GELU()
        self.init_weights()
    
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network with dilated convolutions."""
    
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2, dilations=None):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        if dilations is None:
            dilations = [2 ** i for i in range(num_levels)]
        
        for i in range(num_levels):
            dilation_size = dilations[i]
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Calculate padding to maintain sequence length (causal padding)
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                                       dilation=dilation_size, padding=padding, dropout=dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape: [batch, sequence, features] -> [batch, features, sequence]
        x = x.transpose(1, 2)
        output = self.network(x)
        # Return to original shape
        return output.transpose(1, 2)


class CryptoTCN:
    """Temporal Convolutional Network for cryptocurrency price pattern recognition."""
    
    def __init__(self, channels=[64, 128, 256, 512, 1024], kernel_size=3, 
                 dilations=[1, 2, 4, 8, 16, 32], dropout=0.2):
        """Initialize the TCN model.
        
        Args:
            channels: List of channel sizes for each layer
            kernel_size: Size of the convolutional kernel
            dilations: List of dilation factors for each layer
            dropout: Dropout rate
        """
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.dropout = dropout
        self.logger = logging.getLogger(__name__)
        
        try:
            # Check if PyTorch is available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self.device}")
            
            # Create the TCN model
            self.model = self._create_model()
            
        except NameError:
            self.logger.warning("Running in simulation mode without PyTorch")
            self.device = None
            self.model = None
    
    def _create_model(self):
        """Create the TCN model architecture."""
        if self.device is not None:
            class TCNModel(nn.Module):
                def __init__(self, input_size, channels, kernel_size, dilations, dropout):
                    super(TCNModel, self).__init__()
                    
                    # TCN for feature extraction
                    self.tcn = TemporalConvNet(input_size, channels, kernel_size, dropout, dilations)
                    
                    # Output layers
                    self.fc_out = nn.Linear(channels[-1], 5)  # Predict next OHLCV
                    self.signal_head = nn.Linear(channels[-1], 3)  # BUY, SELL, HOLD probabilities
                    
                def forward(self, x):
                    # x shape: [batch_size, seq_length, features]
                    batch_size = x.shape[0]
                    
                    # Apply TCN
                    tcn_output = self.tcn(x)  # [batch_size, seq_length, channels[-1]]
                    
                    # Use the last sequence element for prediction
                    last_hidden = tcn_output[:, -1, :]  # [batch_size, channels[-1]]
                    
                    # Generate predictions
                    next_candle = self.fc_out(last_hidden)  # [batch_size, 5]
                    signal_probs = F.softmax(self.signal_head(last_hidden), dim=-1)  # [batch_size, 3]
                    
                    # Store activations for visualization
                    activations = tcn_output.detach().mean(dim=1).cpu().numpy()
                    
                    return next_candle, signal_probs, activations
            
            model = TCNModel(
                input_size=5,  # OHLCV
                channels=self.channels,
                kernel_size=self.kernel_size,
                dilations=self.dilations,
                dropout=self.dropout
            ).to(self.device)
            
            return model
        return None
    
    def _preprocess_data(self, ohlcv_data: np.ndarray) -> Optional[torch.Tensor]:
        """Preprocess OHLCV data for the TCN model.
        
        Args:
            ohlcv_data: OHLCV data array
            
        Returns:
            Preprocessed data tensor or None if in simulation mode
        """
        if self.device is not None:
            # Ensure we have at least 100 data points for sequence analysis
            seq_length = 100
            
            # Extract the most recent sequence
            if ohlcv_data.shape[0] >= seq_length:
                recent_data = ohlcv_data[-seq_length:]
            else:
                # Pad with zeros if not enough data
                padding = np.zeros((seq_length - ohlcv_data.shape[0], ohlcv_data.shape[1]))
                recent_data = np.vstack([padding, ohlcv_data])
            
            # Normalize the data
            mean = np.mean(recent_data, axis=0, keepdims=True)
            std = np.std(recent_data, axis=0, keepdims=True) + 1e-8
            normalized_data = (recent_data - mean) / std
            
            # Convert to tensor
            tensor_data = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(0).to(self.device)
            return tensor_data
        return None
    
    def predict(self, ohlcv_data: np.ndarray) -> Dict[str, Any]:
        """Generate predictions using the TCN model.
        
        Args:
            ohlcv_data: OHLCV data array
            
        Returns:
            Dictionary containing predictions and model metrics
        """
        self.logger.info("Generating TCN predictions")
        
        if self.device is not None and self.model is not None:
            # Real TCN computation
            self.model.eval()
            with torch.no_grad():
                # Preprocess data
                tensor_data = self._preprocess_data(ohlcv_data)
                
                # Forward pass
                next_candle, signal_probs, activations = self.model(tensor_data)
                
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
                
                return {
                    "buy_probability": float(buy_prob),
                    "sell_probability": float(sell_prob),
                    "hold_probability": float(hold_prob),
                    "signal_type": signal_type,
                    "confidence": confidence,
                    "predicted_move": float(predicted_move_pct),
                    "next_candle": next_candle_np.tolist(),
                    "activations": activations.tolist(),
                    "activation": float(np.mean(activations))  # Average activation for signal generation
                }
        else:
            # Simulation mode when PyTorch not available
            self.logger.info("Using simulated TCN predictions")
            
            # Simulate TCN behavior
            buy_prob = 0.4 + np.random.random() * 0.5  # Random between 0.4 and 0.9
            sell_prob = 0.4 + np.random.random() * 0.5  # Random between 0.4 and 0.9
            
            # Normalize to ensure probabilities sum to 1
            total = buy_prob + sell_prob
            if total > 1.0:
                buy_prob /= total
                sell_prob /= total
            
            hold_prob = 1.0 - buy_prob - sell_prob
            
            # Generate simulated activations
            activations = np.random.random((1, len(self.channels))).tolist()
            
            # Generate simulated predicted move
            predicted_move_pct = (np.random.random() - 0.5) * 0.05  # -2.5% to +2.5%
            
            return {
                "buy_probability": float(buy_prob),
                "sell_probability": float(sell_prob),
                "hold_probability": float(hold_prob),
                "signal_type": "BUY" if buy_prob > sell_prob and buy_prob > hold_prob else \
                              "SELL" if sell_prob > buy_prob and sell_prob > hold_prob else "HOLD",
                "confidence": float(max(buy_prob, sell_prob, hold_prob)),
                "predicted_move": float(predicted_move_pct),
                "next_candle": [0.0, 0.0, 0.0, 0.0, 0.0],  # Placeholder
                "activations": activations,
                "activation": float(0.7 + np.random.random() * 0.3)  # Random between 0.7 and 1.0
            }