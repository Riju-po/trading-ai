import os
import torch
import pickle
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class ModelPersistence:
    """Utility for saving and loading AI models."""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize the model persistence utility.
        
        Args:
            models_dir: Directory to store model files
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    def save_torch_model(self, model: torch.nn.Module, model_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Save a PyTorch model to disk.
        
        Args:
            model: PyTorch model to save
            model_name: Name of the model (used for filename)
            metadata: Optional metadata to save with the model
        """
        try:
            # Create the full path
            model_path = os.path.join(self.models_dir, f"{model_name}.pt")
            
            # Prepare data to save
            save_dict = {
                'model_state_dict': model.state_dict(),
                'metadata': metadata or {}
            }
            
            # Save the model
            torch.save(save_dict, model_path)
            logger.info(f"Saved PyTorch model to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving PyTorch model {model_name}: {e}")
            return False
    
    def load_torch_model(self, model: torch.nn.Module, model_name: str) -> Optional[Dict[str, Any]]:
        """Load a PyTorch model from disk.
        
        Args:
            model: PyTorch model instance to load weights into
            model_name: Name of the model to load
            
        Returns:
            Metadata dictionary if successful, None otherwise
        """
        try:
            # Create the full path
            model_path = os.path.join(self.models_dir, f"{model_name}.pt")
            
            # Check if file exists
            if not os.path.exists(model_path):
                logger.warning(f"Model file {model_path} not found")
                return None
            
            # Load the model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(model_path, map_location=device)
            
            # Load state dict into model
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            logger.info(f"Loaded PyTorch model from {model_path}")
            return checkpoint.get('metadata', {})
        except Exception as e:
            logger.error(f"Error loading PyTorch model {model_name}: {e}")
            return None
    
    def save_pickle_model(self, model: Any, model_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Save any model using pickle serialization.
        
        Args:
            model: Model object to save
            model_name: Name of the model (used for filename)
            metadata: Optional metadata to save with the model
        """
        try:
            # Create the full path
            model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
            
            # Prepare data to save
            save_dict = {
                'model': model,
                'metadata': metadata or {}
            }
            
            # Save the model
            with open(model_path, 'wb') as f:
                pickle.dump(save_dict, f)
                
            logger.info(f"Saved pickle model to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving pickle model {model_name}: {e}")
            return False
    
    def load_pickle_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load a model using pickle deserialization.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Dictionary containing 'model' and 'metadata' if successful, None otherwise
        """
        try:
            # Create the full path
            model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
            
            # Check if file exists
            if not os.path.exists(model_path):
                logger.warning(f"Model file {model_path} not found")
                return None
            
            # Load the model
            with open(model_path, 'rb') as f:
                loaded_dict = pickle.load(f)
                
            logger.info(f"Loaded pickle model from {model_path}")
            return loaded_dict
        except Exception as e:
            logger.error(f"Error loading pickle model {model_name}: {e}")
            return None
    
    def save_weights(self, weights: Dict[str, Any], model_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Save model weights (for non-PyTorch models like quantum models).
        
        Args:
            weights: Dictionary of model weights
            model_name: Name of the model (used for filename)
            metadata: Optional metadata to save with the weights
        """
        try:
            # Create the full path
            weights_path = os.path.join(self.models_dir, f"{model_name}_weights.pkl")
            
            # Prepare data to save
            save_dict = {
                'weights': weights,
                'metadata': metadata or {}
            }
            
            # Save the weights
            with open(weights_path, 'wb') as f:
                pickle.dump(save_dict, f)
                
            logger.info(f"Saved model weights to {weights_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model weights {model_name}: {e}")
            return False
    
    def load_weights(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load model weights (for non-PyTorch models like quantum models).
        
        Args:
            model_name: Name of the model to load weights for
            
        Returns:
            Dictionary containing 'weights' and 'metadata' if successful, None otherwise
        """
        try:
            # Create the full path
            weights_path = os.path.join(self.models_dir, f"{model_name}_weights.pkl")
            
            # Check if file exists
            if not os.path.exists(weights_path):
                logger.warning(f"Weights file {weights_path} not found")
                return None
            
            # Load the weights
            with open(weights_path, 'rb') as f:
                loaded_dict = pickle.load(f)
                
            logger.info(f"Loaded model weights from {weights_path}")
            return loaded_dict
        except Exception as e:
            logger.error(f"Error loading model weights {model_name}: {e}")
            return None
    
    def list_saved_models(self) -> Dict[str, list]:
        """List all saved models in the models directory.
        
        Returns:
            Dictionary with lists of PyTorch models, pickle models, and weight files
        """
        try:
            if not os.path.exists(self.models_dir):
                return {'torch_models': [], 'pickle_models': [], 'weight_files': []}
            
            files = os.listdir(self.models_dir)
            
            torch_models = [f.replace('.pt', '') for f in files if f.endswith('.pt')]
            pickle_models = [f.replace('.pkl', '') for f in files if f.endswith('.pkl') and not f.endswith('_weights.pkl')]
            weight_files = [f.replace('_weights.pkl', '') for f in files if f.endswith('_weights.pkl')]
            
            return {
                'torch_models': torch_models,
                'pickle_models': pickle_models,
                'weight_files': weight_files
            }
        except Exception as e:
            logger.error(f"Error listing saved models: {e}")
            return {'torch_models': [], 'pickle_models': [], 'weight_files': []}