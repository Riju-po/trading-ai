import unittest
import numpy as np
from src.risk_management.risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.risk_manager = RiskManager()
        
        # Create sample OHLCV data
        self.ohlcv_data = np.random.random((100, 5)) * 100
        self.ohlcv_data[:, 0] = np.sort(self.ohlcv_data[:, 0])  # Ensure open prices are sorted
        
        # Create sample prediction for BUY signal
        self.buy_prediction = {
            'signal_type': 'BUY',
            'predicted_move': 0.025,  # 2.5% expected move
            'confidence': 0.85
        }
        
        # Create sample prediction for SELL signal
        self.sell_prediction = {
            'signal_type': 'SELL',
            'predicted_move': 0.02,  # 2% expected move
            'confidence': 0.9
        }
        
        # Create sample quantum probability distribution
        self.quantum_distribution = np.random.random(10)
        self.quantum_distribution = self.quantum_distribution / np.sum(self.quantum_distribution)  # Normalize
    
    def test_calculate_levels_buy(self):
        # Test stop loss and take profit calculation for BUY signal
        stop_loss, take_profit = self.risk_manager.calculate_levels(
            self.buy_prediction, 
            self.ohlcv_data, 
            self.quantum_distribution
        )
        
        # Get current price (last close)
        current_price = self.ohlcv_data[-1, 3]  # Last close price
        
        # Check that stop loss is below current price for BUY
        self.assertLess(stop_loss, current_price)
        
        # Check that take profit is above current price for BUY
        self.assertGreater(take_profit, current_price)
        
        # Check that risk/reward ratio is reasonable
        risk = current_price - stop_loss
        reward = take_profit - current_price
        risk_reward_ratio = reward / risk
        
        self.assertGreaterEqual(risk_reward_ratio, 1.5)  # Minimum acceptable R:R
    
    def test_calculate_levels_sell(self):
        # Test stop loss and take profit calculation for SELL signal
        stop_loss, take_profit = self.risk_manager.calculate_levels(
            self.sell_prediction, 
            self.ohlcv_data, 
            self.quantum_distribution
        )
        
        # Get current price (last close)
        current_price = self.ohlcv_data[-1, 3]  # Last close price
        
        # Check that stop loss is above current price for SELL
        self.assertGreater(stop_loss, current_price)
        
        # Check that take profit is below current price for SELL
        self.assertLess(take_profit, current_price)
        
        # Check that risk/reward ratio is reasonable
        risk = stop_loss - current_price
        reward = current_price - take_profit
        risk_reward_ratio = reward / risk
        
        self.assertGreaterEqual(risk_reward_ratio, 1.5)  # Minimum acceptable R:R
    
    def test_dynamic_stop_loss(self):
        # Test with different market regimes
        regimes = ['trending', 'ranging', 'volatile']
        
        for regime in regimes:
            # Update prediction with current regime
            self.buy_prediction['market_regime'] = regime
            
            stop_loss, take_profit = self.risk_manager.calculate_levels(
                self.buy_prediction, 
                self.ohlcv_data, 
                self.quantum_distribution
            )
            
            # Get current price (last close)
            current_price = self.ohlcv_data[-1, 3]  # Last close price
            
            # Check that stop loss is valid
            self.assertLess(stop_loss, current_price)
            
            # Check that take profit is valid
            self.assertGreater(take_profit, current_price)

if __name__ == '__main__':
    unittest.main()