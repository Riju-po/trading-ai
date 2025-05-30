import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, timedelta
import logging
from typing import Optional, List, Dict, Any
import dotenv
# Remove this duplicate import
# import os

try:
    import ccxt
except ImportError:
    logging.warning("CCXT not installed. Using simulation mode.")

# Load environment variables
dotenv.load_dotenv()

class OHLCVDataLoader:
    """Data loader for OHLCV cryptocurrency data."""
    
    def __init__(self, timeframe: str = "15m", symbol: str = "BTC/USDT", 
                 lookback_candles: int = 200, data_dir: str = "data"):
        """Initialize the OHLCV data loader.
        
        Args:
            timeframe: Timeframe for candles (e.g., '15m', '1h')
            symbol: Trading pair symbol
            lookback_candles: Number of historical candles to load
            data_dir: Directory to store/load data
        """
        self.timeframe = timeframe
        self.symbol = symbol
        self.lookback_candles = lookback_candles
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize exchange connection if available
        try:
            # Use Delta Exchange instead of Binance
            self.exchange = ccxt.delta({
                'enableRateLimit': True,
                'apiKey': os.getenv('DELTA_API_KEY'),
                'secret': os.getenv('DELTA_API_SECRET')
            })
            self.simulation_mode = False
            self.logger.info("Connected to Delta Exchange")
        except NameError:
            self.logger.warning("Running in simulation mode without exchange connection")
            self.exchange = None
            self.simulation_mode = True
        except Exception as e:
            self.logger.error(f"Error connecting to Delta Exchange: {e}")
            self.exchange = None
            self.simulation_mode = True
    
    def load_latest_data(self) -> np.ndarray:
        """Load the latest OHLCV data.
        
        Returns:
            Numpy array of OHLCV data with shape (lookback_candles, 5)
        """
        self.logger.info(f"Loading latest {self.lookback_candles} candles for {self.symbol} ({self.timeframe})")
        
        if not self.simulation_mode and self.exchange is not None:
            try:
                # Fetch data from exchange
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    limit=self.lookback_candles
                )
                
                # Convert to numpy array
                data = np.array(ohlcv)[:, 1:6]  # Skip timestamp, keep OHLCV
                
                # Save data for future use
                self._save_data(data)
                
                return data
                
            except Exception as e:
                self.logger.error(f"Error fetching data from exchange: {e}")
                self.logger.info("Falling back to cached or simulated data")
        
        # Try to load cached data
        cached_data = self._load_cached_data()
        if cached_data is not None:
            return cached_data
        
        # If all else fails, generate simulated data
        return self._generate_simulated_data()
    
    def _save_data(self, data: np.ndarray):
        """Save OHLCV data to cache.
        
        Args:
            data: OHLCV data array
        """
        cache_file = os.path.join(self.data_dir, f"{self.symbol.replace('/', '_')}_{self.timeframe}.npy")
        np.save(cache_file, data)
        self.logger.info(f"Saved data to {cache_file}")
    
    def _load_cached_data(self) -> Optional[np.ndarray]:
        """Load OHLCV data from cache.
        
        Returns:
            OHLCV data array or None if cache doesn't exist
        """
        cache_file = os.path.join(self.data_dir, f"{self.symbol.replace('/', '_')}_{self.timeframe}.npy")
        
        if os.path.exists(cache_file):
            try:
                data = np.load(cache_file)
                self.logger.info(f"Loaded cached data from {cache_file}")
                return data
            except Exception as e:
                self.logger.error(f"Error loading cached data: {e}")
        
        return None
    
    def _generate_simulated_data(self) -> np.ndarray:
        """Generate simulated OHLCV data for testing.
        
        Returns:
            Simulated OHLCV data array
        """
        self.logger.info("Generating simulated OHLCV data")
        
        # Start with a base price
        base_price = 40000.0  # For BTC/USDT
        if "ETH" in self.symbol:
            base_price = 2000.0
        
        # Generate random walk
        returns = np.random.normal(0, 0.01, self.lookback_candles)
        price_multipliers = np.cumprod(1 + returns)
        closes = base_price * price_multipliers
        
        # Generate OHLCV data
        highs = closes * (1 + np.random.uniform(0.001, 0.015, self.lookback_candles))
        lows = closes * (1 - np.random.uniform(0.001, 0.015, self.lookback_candles))
        opens = closes * (1 + np.random.normal(0, 0.005, self.lookback_candles))
        volumes = np.random.uniform(500, 5000, self.lookback_candles) * closes
        
        # Combine into OHLCV array
        ohlcv = np.column_stack((opens, highs, lows, closes, volumes))
        
        # Save simulated data
        self._save_data(ohlcv)
        
        return ohlcv
    
    def _load_exchange_data(self):
        try:
            # Existing code to fetch from exchange
            pass # Add pass statement to resolve IndentationError
        except Exception as e:
            logger.warning(f"Could not fetch data from exchange: {e}")
            logger.info("Using cached or simulated data instead")
            # Try to load cached data first
            cached_data = self._load_cached_data()
            if cached_data is not None:
                return cached_data
            # If no cached data, generate simulated data
            return self._generate_simulated_data()
