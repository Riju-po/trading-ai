import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration manager for the trading AI system."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Set default configuration if not loaded
        if not self.config:
            self.config = self._get_default_config()
            self._save_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        logger.warning(f"Configuration file {self.config_path} not found. Using defaults.")
        return {}
    
    def _save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "data": {
                "timeframes": ["15m", "1h", "4h"],
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "lookback_candles": 200,
                "data_dir": "data"
            },
            "exchange": {
                "name": "delta",
                "use_testnet": False,
                "rate_limit": True
            },
            "models": {
                "quantum": {
                    "n_qubits": 8,
                    "layers": 6,
                    "entanglement": "full",
                    "weight": 0.25
                },
                "transformer": {
                    "d_model": 512,
                    "n_heads": 16,
                    "n_layers": 8,
                    "seq_length": 100,
                    "weight": 0.30
                },
                "tcn": {
                    "channels": [64, 128, 256, 512, 1024],
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32],
                    "dropout": 0.2,
                    "weight": 0.25
                },
                "gnn": {
                    "weight": 0.10
                },
                "chaos": {
                    "weight": 0.05
                },
                "swarm": {
                    "weight": 0.05
                }
            },
            "signal": {
                "min_confidence": 0.85,
                "min_risk_reward": 2.0
            },
            "risk_management": {
                "max_stop_loss_pct": 0.05,
                "min_stop_loss_pct": 0.01,
                "min_reward_risk_ratio": 2.0
            },
            "gpu": {
                "use_gpu": True,
                "memory_fraction": 0.8,
                "enable_cudnn_benchmark": True,
                "use_half_precision": False
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: Configuration key (can be nested with dots, e.g., 'models.quantum.n_qubits')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                logger.warning(f"Configuration key '{key}' not found. Using default: {default}")
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set a configuration value.
        
        Args:
            key: Configuration key (can be nested with dots, e.g., 'models.quantum.n_qubits')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the nested dictionary
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        logger.info(f"Set configuration '{key}' to {value}")
        
        # Save the updated configuration
        self._save_config()
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get model weights for signal fusion.
        
        Returns:
            Dictionary of model weights
        """
        weights = {}
        models_config = self.get('models', {})
        
        for model_name, model_config in models_config.items():
            if isinstance(model_config, dict) and 'weight' in model_config:
                weights[model_name] = model_config['weight']
        
        return weights

# Example usage
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize configuration manager
    config = ConfigManager()
    
    # Get configuration values
    quantum_qubits = config.get('models.quantum.n_qubits', 4)
    logger.info(f"Quantum qubits: {quantum_qubits}")
    
    # Get model weights
    weights = config.get_model_weights()
    logger.info(f"Model weights: {weights}")
    
    # Set a configuration value
    config.set('models.quantum.n_qubits', 10)
    
    # Verify the change
    quantum_qubits = config.get('models.quantum.n_qubits')
    logger.info(f"Updated quantum qubits: {quantum_qubits}")

if __name__ == "__main__":
    main()