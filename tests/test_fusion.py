import unittest
import numpy as np
from src.fusion.ensemble import SignalFusion

class TestFusion(unittest.TestCase):
    def setUp(self):
        self.weights = {
            'quantum': 0.25,
            'transformer': 0.30,
            'tcn': 0.25,
            'gnn': 0.10,
            'chaos': 0.05,
            'swarm': 0.05
        }
        
        self.fusion = SignalFusion(weights=self.weights)
        
        # Create sample predictions
        self.predictions = {
            'quantum': {
                'buy_probability': 0.7,
                'sell_probability': 0.2,
                'hold_probability': 0.1,
                'signal_type': 'BUY',
                'confidence': 0.7,
                'predicted_move': 0.02,
                'probability_distribution': np.random.random(10)
            },
            'transformer': {
                'buy_probability': 0.8,
                'sell_probability': 0.1,
                'hold_probability': 0.1,
                'signal_type': 'BUY',
                'confidence': 0.8,
                'predicted_move': 0.025,
                'attention_weights': np.random.random((5, 5))
            },
            'tcn': {
                'buy_probability': 0.75,
                'sell_probability': 0.15,
                'hold_probability': 0.1,
                'signal_type': 'BUY',
                'confidence': 0.75,
                'predicted_move': 0.022,
                'activations': np.random.random(10)
            },
            'gnn': {
                'buy_probability': 0.65,
                'sell_probability': 0.25,
                'hold_probability': 0.1,
                'signal_type': 'BUY',
                'confidence': 0.65,
                'predicted_move': 0.018
            },
            'chaos': {
                'regime': 'trending',
                'chaos_level': 0.3,
                'signal_type': 'BUY',
                'buy_probability': 0.6,
                'sell_probability': 0.3,
                'hold_probability': 0.1,
                'confidence': 0.6
            },
            'swarm': {
                'buy_probability': 0.7,
                'sell_probability': 0.2,
                'hold_probability': 0.1,
                'signal_type': 'BUY',
                'confidence': 0.7,
                'predicted_move': 0.021
            }
        }
    
    def test_fusion(self):
        result = self.fusion.fuse(self.predictions)
        
        # Check that fusion returns a list
        self.assertIsInstance(result, list)
        
        # If result is not empty, check its properties
        if result:
            self.assertIn('signal_type', result[0])
            self.assertIn('confidence', result[0])
            self.assertIn('predicted_move', result[0])
            self.assertIn('risk_reward_ratio', result[0])
            
            # Check that confidence is valid
            self.assertGreaterEqual(result[0]['confidence'], 0.0)
            self.assertLessEqual(result[0]['confidence'], 1.0)
            
            # Check that signal type is valid
            self.assertIn(result[0]['signal_type'], ['BUY', 'SELL', 'HOLD'])
            
            # Check that risk/reward ratio is positive
            self.assertGreater(result[0]['risk_reward_ratio'], 0.0)
    
    def test_weight_normalization(self):
        # Test with unnormalized weights
        unnormalized_weights = {
            'quantum': 1.0,
            'transformer': 2.0,
            'tcn': 1.0,
            'gnn': 0.5,
            'chaos': 0.25,
            'swarm': 0.25
        }
        
        fusion = SignalFusion(weights=unnormalized_weights)
        result = fusion.fuse(self.predictions)
        
        # Check that fusion still works with unnormalized weights
        self.assertIsInstance(result, list)
        
    def test_regime_weight_adjustment(self):
        # Test with different market regimes
        regimes = ['trending', 'ranging', 'volatile']
        
        for regime in regimes:
            # Update chaos model prediction with current regime
            self.predictions['chaos']['regime'] = regime
            
            result = self.fusion.fuse(self.predictions)
            
            # Check that fusion works for all regimes
            self.assertIsInstance(result, list)
    
    def test_missing_models(self):
        # Test with some models missing
        incomplete_predictions = {
            'quantum': self.predictions['quantum'],
            'transformer': self.predictions['transformer']
        }
        
        result = self.fusion.fuse(incomplete_predictions)
        
        # Check that fusion still works with incomplete predictions
        self.assertIsInstance(result, list)

if __name__ == '__main__':
    unittest.main()