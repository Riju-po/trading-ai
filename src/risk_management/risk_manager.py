import numpy as np
from typing import Dict, List, Any, Tuple
import logging

class RiskManager:
    """AI-based risk management system."""
    
    def __init__(self, min_stop_pct: float = 0.01, max_stop_pct: float = 0.03, 
                 min_reward_risk_ratio: float = 2.0):
        """Initialize the risk management system.
        
        Args:
            min_stop_pct: Minimum stop loss distance (percentage)
            max_stop_pct: Maximum stop loss distance (percentage)
            min_reward_risk_ratio: Minimum reward-to-risk ratio
        """
        self.min_stop_pct = min_stop_pct
        self.max_stop_pct = max_stop_pct
        self.min_reward_risk_ratio = min_reward_risk_ratio
        self.logger = logging.getLogger(__name__)
    
    def calculate_levels(self, prediction: Dict[str, Any], ohlcv_data: np.ndarray, 
                        probability_distribution: List[float]) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels.
        
        Args:
            prediction: Prediction dictionary from signal fusion
            ohlcv_data: OHLCV data array
            probability_distribution: Quantum probability distribution
            
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        self.logger.info("Calculating risk management levels")
        
        # Extract current price (last close)
        current_price = ohlcv_data[-1, 3]  # Assuming close is at index 3
        
        # Extract signal type and predicted move
        signal_type = prediction['signal_type']
        predicted_move = prediction['predicted_move']
        market_regime = prediction.get('market_regime', 'trending')
        
        # Calculate dynamic stop loss based on market regime and quantum probabilities
        stop_loss_pct = self._calculate_dynamic_stop_loss(
            market_regime, probability_distribution, ohlcv_data
        )
        
        # Ensure stop loss is within bounds
        stop_loss_pct = max(self.min_stop_pct, min(self.max_stop_pct, stop_loss_pct))
        
        # Calculate take profit based on predicted move and minimum reward/risk ratio
        take_profit_pct = max(abs(predicted_move), stop_loss_pct * self.min_reward_risk_ratio)
        
        # Calculate actual price levels
        if signal_type == "BUY":
            stop_loss_price = current_price * (1 - stop_loss_pct)
            take_profit_price = current_price * (1 + take_profit_pct)
        elif signal_type == "SELL":
            stop_loss_price = current_price * (1 + stop_loss_pct)
            take_profit_price = current_price * (1 - take_profit_pct)
        else:  # EXIT or unknown
            stop_loss_price = 0.0
            take_profit_price = 0.0
        
        self.logger.info(f"Calculated stop loss: {stop_loss_price}, take profit: {take_profit_price}")
        return stop_loss_price, take_profit_price
    
    def _calculate_dynamic_stop_loss(self, market_regime: str, 
                                   probability_distribution: List[float],
                                   ohlcv_data: np.ndarray) -> float:
        """Calculate dynamic stop loss percentage based on market conditions.
        
        Args:
            market_regime: Current market regime ('trending', 'ranging', 'volatile')
            probability_distribution: Quantum probability distribution
            ohlcv_data: OHLCV data array
            
        Returns:
            Stop loss percentage
        """
        # Calculate volatility (standard deviation of returns)
        returns = np.diff(ohlcv_data[:, 3]) / ohlcv_data[:-1, 3]  # Close price returns
        volatility = np.std(returns) * np.sqrt(96)  # Annualized (assuming 15-min data)
        
        # Base stop loss on volatility
        base_stop_pct = volatility * 0.5  # Half of daily volatility
        
        # Adjust for market regime
        regime_multiplier = 1.0
        if market_regime == "trending":
            regime_multiplier = 0.8  # Tighter stops in trending markets
        elif market_regime == "ranging":
            regime_multiplier = 1.2  # Wider stops in ranging markets
        elif market_regime == "volatile":
            regime_multiplier = 1.5  # Even wider stops in volatile markets
        
        # Adjust based on quantum probability distribution
        # Higher uncertainty = wider stops
        quantum_uncertainty = np.std(probability_distribution) * 5.0
        
        # Combine factors
        stop_loss_pct = base_stop_pct * regime_multiplier * (1.0 + quantum_uncertainty)
        
        return stop_loss_pct