import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    logging.warning("PyTorch not installed. Using simulation mode.")

class PriceTransformer:
    """Transformer model for price sequence analysis."""
    
    def __init__(self, d_model: int = 512, n_heads: int = 16, n_layers: int = 8, 
                 seq_length: int = 100, dropout: float = 0.1):
        """Initialize the transformer model.
        
        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            seq_length: Input sequence length
            dropout: Dropout rate
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.seq_length = seq_length
        self.dropout = dropout
        self.logger = logging.getLogger(__name__)
        
        try:
            # Check if PyTorch is available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self.device}")
            
            # Create the transformer model
            self.model = self._create_model()
            
        except NameError:
            self.logger.warning("Running in simulation mode without PyTorch")
            self.device = None
            self.model = None
    
    def _create_model(self):
        """Create the transformer model architecture."""
        if self.device is not None:
            class TransformerModel(nn.Module):
                def __init__(self, d_model, n_heads, n_layers, seq_length, dropout):
                    super().__init__()
                    
                    # Input embedding
                    self.embedding = nn.Linear(5, d_model)  # 5 for OHLCV
                    
                    # Positional encoding
                    self.pos_encoder = nn.Parameter(torch.zeros(seq_length, d_model))
                    
                    # Transformer encoder layers
                    encoder_layers = nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=n_heads,
                        dim_feedforward=d_model*4,
                        dropout=dropout,
                        activation="gelu"
                    )
                    self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
                    
                    # Output layers
                    self.fc_out = nn.Linear(d_model, 5)  # Predict next OHLCV
                    self.signal_head = nn.Linear(d_model, 3)  # BUY, SELL, HOLD probabilities
                    
                def forward(self, x):
                    # x shape: [batch_size, seq_length, 5]
                    batch_size = x.shape[0]
                    
                    # Embed input
                    x = self.embedding(x)  # [batch_size, seq_length, d_model]
                    
                    # Add positional encoding
                    x = x + self.pos_encoder.unsqueeze(0)
                    
                    # Reshape for transformer: [seq_length, batch_size, d_model]
                    x = x.permute(1, 0, 2)
                    
                    # Apply transformer encoder
                    attention_weights = []
                    
                    # Store attention weights (simplified for demonstration)
                    def get_attention_hook(layer_idx):
                        def hook(module, input, output):
                            attention_weights.append(output.detach())
                        return hook
                    
                    # Register hooks for attention weights
                    for i, layer in enumerate(self.transformer_encoder.layers):
                        layer.self_attn.register_forward_hook(get_attention_hook(i))
                    
                    # Pass through transformer
                    transformer_output = self.transformer_encoder(x)
                    
                    # Use the last sequence element for prediction
                    last_hidden = transformer_output[-1]  # [batch_size, d_model]
                    
                    # Generate predictions
                    next_candle = self.fc_out(last_hidden)  # [batch_size, 5]
                    signal_probs = F.softmax(self.signal_head(last_hidden), dim=-1)  # [batch_size, 3]
                    
                    return next_candle, signal_probs, attention_weights
            
            model = TransformerModel(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                seq_length=self.seq_length,
                dropout=self.dropout
            ).to(self.device)
            
            return model
        return None
    
    def _preprocess_data(self, ohlcv_data: np.ndarray) -> Optional[torch.Tensor]:
        """Preprocess OHLCV data for the transformer model.
        
        Args:
            ohlcv_data: OHLCV data array
            
        Returns:
            Preprocessed data tensor or None if in simulation mode
        """
        if self.device is not None:
            # Extract the most recent sequence
            if ohlcv_data.shape[0] >= self.seq_length:
                recent_data = ohlcv_data[-self.seq_length:]
            else:
                # Pad with zeros if not enough data
                padding = np.zeros((self.seq_length - ohlcv_data.shape[0], ohlcv_data.shape[1]))
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
        """Generate predictions using the transformer model.
        
        Args:
            ohlcv_data: OHLCV data array
            
        Returns:
            Dictionary containing predictions and model metrics
        """
        self.logger.info("Generating transformer predictions")
        
        if self.device is not None and self.model is not None:
            # Real transformer computation
            self.model.eval()
            with torch.no_grad():
                # Preprocess data
                tensor_data = self._preprocess_data(ohlcv_data)
                
                # Forward pass
                next_candle, signal_probs, attention_weights = self.model(tensor_data)
                
                # Convert to numpy for further processing
                next_candle_np = next_candle.cpu().numpy()[0]  # [5]
                signal_probs_np = signal_probs.cpu().numpy()[0]  # [3]
                
                # Process attention weights
                attn_weights_processed = []
                for layer_weights in attention_weights:
                    # Simplify attention weights for visualization
                    attn_weights_processed.append(layer_weights.cpu().numpy().mean(axis=0).tolist())
                
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
                    "attention_weights": attn_weights_processed,
                    "activation": float(np.mean(np.abs(attn_weights_processed[-1])))
                }
        else:
            # Simulation mode when PyTorch not available
            self.logger.info("Using simulated transformer predictions")
            
            # Simulate transformer behavior
            buy_prob = 0.4 + np.random.random() * 0.5  # Random between 0.4 and 0.9
            sell_prob = 0.4 + np.random.random() * 0.5  # Random between 0.4 and 0.9
            
            # Normalize to ensure probabilities sum to 1
            total = buy_prob + sell_prob
            if total > 1.0:
                buy_prob /= total
                sell_prob /= total
            
            hold_prob = 1.0 - buy_prob - sell_prob
            
            # Generate simulated attention weights
            attention_weights = []
            for _ in range(self.n_layers):
                layer_weights = np.random.random((self.seq_length, self.seq_length))
                attention_weights.append(layer_weights.tolist())
            
            # Simulate predicted move
            predicted_move_pct = (np.random.random() * 0.1 - 0.05)  # Between -5% and +5%
            
            return {
                "buy_probability": float(buy_prob),
                "sell_probability": float(sell_prob),
                "hold_probability": float(hold_prob),
                "signal_type": "BUY" if buy_prob > sell_prob and buy_prob > hold_prob else \
                              "SELL" if sell_prob > buy_prob and sell_prob > hold_prob else "HOLD",
                "confidence": float(max(buy_prob, sell_prob, hold_prob)),
                "predicted_move": float(predicted_move_pct),
                "next_candle": np.random.random(5).tolist(),
                "attention_weights": attention_weights,
                "activation": float(0.7 + np.random.random() * 0.3)  # Random between 0.7 and 1.0
            }