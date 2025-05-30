from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any

@dataclass
class AISignal:
    """Class representing an AI-generated trading signal."""
    timestamp: datetime
    signal_type: str  # 'BUY', 'SELL', 'EXIT'
    confidence: float  # 0.0 to 1.0
    predicted_move: float  # Expected price change %
    risk_reward_ratio: float  # Minimum 1:2
    stop_loss: float  # AI-calculated stop level
    take_profit: float  # AI-calculated target
    quantum_probability: float
    neural_activation: float
    market_regime: str  # 'trending', 'ranging', 'volatile'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary format for export."""
        return {
            "timestamp": int(self.timestamp.timestamp()),
            "action": self.signal_type,
            "confidence": self.confidence,
            "predictedMove": self.predicted_move,
            "riskReward": self.risk_reward_ratio,
            "stopLoss": self.stop_loss,
            "takeProfit": self.take_profit,
            "quantumProb": self.quantum_probability,
            "neuralActivation": self.neural_activation,
            "regime": self.market_regime
        }
    
    def validate(self) -> bool:
        """Validate that the signal meets minimum criteria."""
        return (
            self.confidence >= 0.85 and
            self.risk_reward_ratio >= 2.0 and
            self.signal_type in ['BUY', 'SELL', 'EXIT']
        )