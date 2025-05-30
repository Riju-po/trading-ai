import unittest
from datetime import datetime
from src.signal_generation.signal import AISignal

class TestSignal(unittest.TestCase):
    def setUp(self):
        # Create a sample signal
        self.signal = AISignal(
            timestamp=datetime.now(),
            signal_type="BUY",
            confidence=0.9,
            predicted_move=0.025,
            risk_reward_ratio=2.5,
            stop_loss=39500.0,
            take_profit=41500.0,
            quantum_probability=0.85,
            neural_activation=0.78,
            market_regime="trending"
        )
    
    def test_signal_creation(self):
        # Check that signal attributes are set correctly
        self.assertEqual(self.signal.signal_type, "BUY")
        self.assertEqual(self.signal.confidence, 0.9)
        self.assertEqual(self.signal.predicted_move, 0.025)
        self.assertEqual(self.signal.risk_reward_ratio, 2.5)
        self.assertEqual(self.signal.stop_loss, 39500.0)
        self.assertEqual(self.signal.take_profit, 41500.0)
        self.assertEqual(self.signal.quantum_probability, 0.85)
        self.assertEqual(self.signal.neural_activation, 0.78)
        self.assertEqual(self.signal.market_regime, "trending")
    
    def test_to_dict(self):
        # Convert signal to dictionary
        signal_dict = self.signal.to_dict()
        
        # Check that dictionary contains all expected keys
        expected_keys = [
            "timestamp", "signal_type", "confidence", "predicted_move",
            "risk_reward_ratio", "stop_loss", "take_profit",
            "quantum_probability", "neural_activation", "market_regime"
        ]
        
        for key in expected_keys:
            self.assertIn(key, signal_dict)
        
        # Check that values match
        self.assertEqual(signal_dict["signal_type"], "BUY")
        self.assertEqual(signal_dict["confidence"], 0.9)
        self.assertEqual(signal_dict["predicted_move"], 0.025)
        self.assertEqual(signal_dict["risk_reward_ratio"], 2.5)
        self.assertEqual(signal_dict["stop_loss"], 39500.0)
        self.assertEqual(signal_dict["take_profit"], 41500.0)
        self.assertEqual(signal_dict["quantum_probability"], 0.85)
        self.assertEqual(signal_dict["neural_activation"], 0.78)
        self.assertEqual(signal_dict["market_regime"], "trending")
    
    def test_is_valid(self):
        # Test valid signal
        self.assertTrue(self.signal.is_valid())
        
        # Test invalid signal (low confidence)
        invalid_signal = AISignal(
            timestamp=datetime.now(),
            signal_type="BUY",
            confidence=0.7,  # Below threshold
            predicted_move=0.025,
            risk_reward_ratio=2.5,
            stop_loss=39500.0,
            take_profit=41500.0,
            quantum_probability=0.85,
            neural_activation=0.78,
            market_regime="trending"
        )
        self.assertFalse(invalid_signal.is_valid())
        
        # Test invalid signal (low risk/reward)
        invalid_signal = AISignal(
            timestamp=datetime.now(),
            signal_type="BUY",
            confidence=0.9,
            predicted_move=0.025,
            risk_reward_ratio=1.5,  # Below threshold
            stop_loss=39500.0,
            take_profit=41500.0,
            quantum_probability=0.85,
            neural_activation=0.78,
            market_regime="trending"
        )
        self.assertFalse(invalid_signal.is_valid())
        
        # Test invalid signal (invalid signal type)
        invalid_signal = AISignal(
            timestamp=datetime.now(),
            signal_type="INVALID",  # Invalid type
            confidence=0.9,
            predicted_move=0.025,
            risk_reward_ratio=2.5,
            stop_loss=39500.0,
            take_profit=41500.0,
            quantum_probability=0.85,
            neural_activation=0.78,
            market_regime="trending"
        )
        self.assertFalse(invalid_signal.is_valid())

if __name__ == '__main__':
    unittest.main()