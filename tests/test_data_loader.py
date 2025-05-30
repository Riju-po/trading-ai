import unittest
import numpy as np
import os
import tempfile
from src.utils.data_loader import OHLCVDataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize data loader with test parameters
        self.data_loader = OHLCVDataLoader(
            timeframe="15m",
            symbol="BTC/USDT",
            lookback_candles=100,
            data_dir=self.temp_dir
        )
    
    def tearDown(self):
        # Clean up temporary files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
    
    def test_load_latest_data(self):
        # Test loading data
        data = self.data_loader.load_latest_data()
        
        # Check that data has the correct shape
        self.assertEqual(data.shape[0], 100)  # 100 candles
        self.assertEqual(data.shape[1], 5)    # OHLCV format
        
        # Check that data is valid
        self.assertTrue(np.all(data[:, 1] >= data[:, 2]))  # High >= Low
        self.assertTrue(np.all(data > 0))  # All values are positive
    
    def test_save_and_load_cache(self):
        # Generate some test data
        test_data = np.random.random((100, 5)) * 1000
        
        # Save to cache
        self.data_loader._save_data(test_data)
        
        # Load from cache
        loaded_data = self.data_loader._load_cached_data()
        
        # Check that loaded data matches saved data
        self.assertTrue(np.array_equal(test_data, loaded_data))
    
    def test_generate_simulated_data(self):
        # Generate simulated data
        data = self.data_loader._generate_simulated_data()
        
        # Check that data has the correct shape
        self.assertEqual(data.shape[0], 100)  # 100 candles
        self.assertEqual(data.shape[1], 5)    # OHLCV format
        
        # Check that data is valid
        self.assertTrue(np.all(data[:, 1] >= data[:, 2]))  # High >= Low
        self.assertTrue(np.all(data > 0))  # All values are positive

if __name__ == '__main__':
    unittest.main()