import unittest
import numpy as np
from src.models.quantum_model import QuantumPricePredictor
from src.models.transformer_model import PriceTransformer
from src.models.tcn_model import CryptoTCN
from src.models.gnn_model import GraphNeuralNetwork
from src.models.exotic_models import ChaosTheoryAnalyzer, SwarmIntelligence, FractalAnalyzer, TopologicalAnalyzer

class TestModels(unittest.TestCase):
    def setUp(self):
        # Create sample OHLCV data for testing
        self.test_data = np.random.random((100, 5)) * 100
        self.test_data[:, 0] = np.sort(self.test_data[:, 0])  # Ensure open prices are sorted
        
    def test_quantum_model(self):
        model = QuantumPricePredictor(n_qubits=4, layers=2)
        result = model.predict(self.test_data)
        
        # Check that the model returns the expected keys
        self.assertIn('buy_probability', result)
        self.assertIn('sell_probability', result)
        self.assertIn('hold_probability', result)
        self.assertIn('signal_type', result)
        
        # Check probability values
        self.assertGreaterEqual(result['buy_probability'], 0.0)
        self.assertLessEqual(result['buy_probability'], 1.0)
        self.assertGreaterEqual(result['sell_probability'], 0.0)
        self.assertLessEqual(result['sell_probability'], 1.0)
        self.assertGreaterEqual(result['hold_probability'], 0.0)
        self.assertLessEqual(result['hold_probability'], 1.0)
        
        # Check that probabilities sum to approximately 1
        total_prob = result['buy_probability'] + result['sell_probability'] + result['hold_probability']
        self.assertAlmostEqual(total_prob, 1.0, delta=0.01)
        
        # Check that signal type is valid
        self.assertIn(result['signal_type'], ['BUY', 'SELL', 'HOLD'])
        
    def test_transformer_model(self):
        model = PriceTransformer(d_model=64, n_heads=2, n_layers=2, seq_length=50)
        result = model.predict(self.test_data)
        
        self.assertIn('signal_type', result)
        self.assertIn('confidence', result)
        self.assertIn('predicted_move', result)
        self.assertIn('attention_weights', result)
        
        # Check confidence value
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
        
        # Check signal type
        self.assertIn(result['signal_type'], ['BUY', 'SELL', 'HOLD'])
        
    def test_tcn_model(self):
        model = CryptoTCN(channels=[32, 64], kernel_size=2, dilations=[1, 2, 4])
        result = model.predict(self.test_data)
        
        self.assertIn('signal_type', result)
        self.assertIn('confidence', result)
        self.assertIn('activations', result)
        
        # Check confidence value
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
        
    def test_gnn_model(self):
        model = GraphNeuralNetwork()
        result = model.predict(self.test_data)
        
        self.assertIn('signal_type', result)
        self.assertIn('confidence', result)
        
        # Check confidence value
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
        
    def test_exotic_models(self):
        chaos_model = ChaosTheoryAnalyzer()
        result = chaos_model.analyze(self.test_data)
        
        self.assertIn('regime', result)
        self.assertIn('chaos_level', result)
        self.assertIn('signal_type', result)
        
        # Check that regime is valid
        self.assertIn(result['regime'], ['trending', 'ranging', 'volatile'])
        
        # Check chaos level
        self.assertGreaterEqual(result['chaos_level'], 0.0)
        self.assertLessEqual(result['chaos_level'], 1.0)
        
        # Test Swarm Intelligence
        swarm_model = SwarmIntelligence()
        result = swarm_model.optimize(self.test_data)
        
        self.assertIn('signal_type', result)
        self.assertIn('confidence', result)
        
        # Test Fractal Analyzer
        fractal_model = FractalAnalyzer()
        result = fractal_model.analyze(self.test_data)
        
        self.assertIn('signal_type', result)
        self.assertIn('fractal_dimension', result)
        
        # Test Topological Analyzer
        topo_model = TopologicalAnalyzer()
        result = topo_model.extract_features(self.test_data)
        
        self.assertIn('signal_type', result)
        self.assertIn('topological_features', result)

if __name__ == '__main__':
    unittest.main()