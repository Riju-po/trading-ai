import numpy as np
from typing import Dict, List, Any, Optional
import logging

class SignalFusion:
    """Fusion algorithm for combining predictions from multiple models."""
    
    def __init__(self, weights: Dict[str, float]):
        """Initialize the signal fusion system.
        
        Args:
            weights: Dictionary mapping model names to their weights in the ensemble
        """
        self.weights = weights
        self.logger = logging.getLogger(__name__)
        
        # Normalize weights to sum to 1
        weight_sum = sum(weights.values())
        if weight_sum != 1.0:
            for key in weights:
                self.weights[key] /= weight_sum
    
    def fuse(self, predictions: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fuse predictions from multiple models.
        
        Args:
            predictions: Dictionary mapping model names to their prediction outputs
            
        Returns:
            List of fused predictions with confidence scores
        """
        self.logger.info("Fusing predictions from multiple models")
        
        # Extract market regime from chaos theory analyzer
        market_regime = "trending"  # Default
        if 'chaos' in predictions and 'regime' in predictions['chaos']:
            market_regime = predictions['chaos']['regime']
        
        # Initialize weighted probabilities
        buy_prob = 0.0
        sell_prob = 0.0
        hold_prob = 0.0
        
        # Apply weights based on model strengths and market regime
        regime_adjusted_weights = self._adjust_weights_for_regime(market_regime)
        
        for model_name, model_pred in predictions.items():
            if model_name in regime_adjusted_weights:
                weight = regime_adjusted_weights[model_name]
                
                # Extract probabilities
                if 'buy_probability' in model_pred and 'sell_probability' in model_pred:
                    buy_prob += model_pred['buy_probability'] * weight
                    sell_prob += model_pred['sell_probability'] * weight
                    
                    # Handle hold probability if available
                    if 'hold_probability' in model_pred:
                        hold_prob += model_pred['hold_probability'] * weight
                    else:
                        # Assume remaining probability is hold
                        hold_prob += (1.0 - model_pred['buy_probability'] - model_pred['sell_probability']) * weight
        
        # Normalize probabilities
        total_prob = buy_prob + sell_prob + hold_prob
        if total_prob > 0:
            buy_prob /= total_prob
            sell_prob /= total_prob
            hold_prob /= total_prob
        
        # Check for consensus (>60% agreement)
        signal_type = "HOLD"
        if buy_prob > 0.6:
            signal_type = "BUY"
        elif sell_prob > 0.6:
            signal_type = "SELL"
        
        # Calculate confidence score
        confidence = max(buy_prob, sell_prob)
        
        # Calculate predicted move
        predicted_move = 0.0
        if 'transformer' in predictions and 'predicted_move' in predictions['transformer']:
            predicted_move = predictions['transformer']['predicted_move']
        
        # Calculate risk/reward ratio
        risk_reward_ratio = 2.0  # Default minimum
        if predicted_move != 0:
            # Assume stop loss is 1% by default
            stop_loss_pct = 0.01
            risk_reward_ratio = abs(predicted_move) / stop_loss_pct
        
        # Create fused prediction
        fused_pred = {
            "signal_type": signal_type,
            "confidence": float(confidence),
            "predicted_move": float(predicted_move),
            "risk_reward_ratio": float(risk_reward_ratio),
            "buy_probability": float(buy_prob),
            "sell_probability": float(sell_prob),
            "hold_probability": float(hold_prob),
            "market_regime": market_regime
        }
        
        return [fused_pred] if signal_type != "HOLD" and confidence >= 0.85 and risk_reward_ratio >= 2.0 else []
    
    def _adjust_weights_for_regime(self, market_regime: str) -> Dict[str, float]:
        """Adjust model weights based on market regime.
        
        Args:
            market_regime: Current market regime ('trending', 'ranging', 'volatile')
            
        Returns:
            Adjusted weights dictionary
        """
        adjusted_weights = self.weights.copy()
        
        if market_regime == "trending":
            # In trending markets, favor transformer and TCN
            if 'transformer' in adjusted_weights:
                adjusted_weights['transformer'] *= 1.2
            if 'tcn' in adjusted_weights:
                adjusted_weights['tcn'] *= 1.1
            if 'chaos' in adjusted_weights:
                adjusted_weights['chaos'] *= 0.8
                
        elif market_regime == "ranging":
            # In ranging markets, favor quantum and swarm
            if 'quantum' in adjusted_weights:
                adjusted_weights['quantum'] *= 1.2
            if 'swarm' in adjusted_weights:
                adjusted_weights['swarm'] *= 1.2
            if 'transformer' in adjusted_weights:
                adjusted_weights['transformer'] *= 0.9
                
        elif market_regime == "volatile":
            # In volatile markets, favor chaos theory and quantum
            if 'chaos' in adjusted_weights:
                adjusted_weights['chaos'] *= 1.5
            if 'quantum' in adjusted_weights:
                adjusted_weights['quantum'] *= 1.2
            if 'tcn' in adjusted_weights:
                adjusted_weights['tcn'] *= 0.8
        
        # Normalize weights to sum to 1
        weight_sum = sum(adjusted_weights.values())
        for key in adjusted_weights:
            adjusted_weights[key] /= weight_sum
        
        return adjusted_weights