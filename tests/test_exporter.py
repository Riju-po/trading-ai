import unittest
import os
import json
import numpy as np
from datetime import datetime
from src.signal_generation.signal import AISignal
from src.tradingview.exporter import TradingViewExporter

class TestExporter(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test output
        os.makedirs("data", exist_ok=True)
        os.makedirs("tradingview", exist_ok=True)
        
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
        
        # Create exporter
        self.exporter = TradingViewExporter()
    
    def tearDown(self):
        # Clean up test files
        if os.path.exists("data/signals.json"):
            os.remove("data/signals.json")
        if os.path.exists("tradingview/signals.js"):
            os.remove("tradingview/signals.js")
        if os.path.exists("tradingview/ai_signals_indicator.pine"):
            os.remove("tradingview/ai_signals_indicator.pine")
    
    def test_export_signals(self):
        # Create sample data for export
        signals = [self.signal]
        quantum_coherence = np.random.random(10)
        transformer_attention = np.random.random((5, 5))
        tcn_activations = np.random.random(10)
        chaos_level = 0.45
        
        # Export signals
        self.exporter.export_signals(
            signals,
            quantum_coherence=quantum_coherence,
            transformer_attention=transformer_attention,
            tcn_activations=tcn_activations,
            chaos_level=chaos_level
        )
        
        # Check that files were created
        self.assertTrue(os.path.exists("data/signals.json"))
        self.assertTrue(os.path.exists("tradingview/signals.js"))
        self.assertTrue(os.path.exists("tradingview/ai_signals_indicator.pine"))
        
        # Check JSON file content
        with open("data/signals.json", "r") as f:
            json_data = json.load(f)
            
            # Check that data contains the signal
            self.assertEqual(len(json_data), 1)
            self.assertEqual(json_data[0]["signal_type"], "BUY")
            self.assertEqual(json_data[0]["confidence"], 0.9)
            self.assertEqual(json_data[0]["risk_reward_ratio"], 2.5)
        
        # Check JS file content
        with open("tradingview/signals.js", "r") as f:
            js_content = f.read()
            
            # Check that JS file contains the signal data
            self.assertIn("BUY", js_content)
            self.assertIn("0.9", js_content)
            self.assertIn("2.5", js_content)
        
        # Check Pine Script file content
        with open("tradingview/ai_signals_indicator.pine", "r") as f:
            pine_content = f.read()
            
            # Check that Pine Script file contains expected content
            self.assertIn("AI Quantum Trading Signals", pine_content)
            self.assertIn("study", pine_content)
            self.assertIn("plot", pine_content)

if __name__ == '__main__':
    unittest.main()