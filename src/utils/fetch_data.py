import logging
import os
import sys
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.data_loader import OHLCVDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Fetching latest OHLCV data for AI Signal Generator")
    
    # Initialize data loader for multiple timeframes and symbols
    timeframes = ['15m', '1h', '4h']
    symbols = ['BTC/USDT', 'ETH/USDT']
    
    for timeframe in timeframes:
        for symbol in symbols:
            logger.info(f"Fetching {symbol} data for {timeframe} timeframe")
            data_loader = OHLCVDataLoader(
                timeframe=timeframe,
                symbol=symbol,
                lookback_candles=200
            )
            
            # Load and save data
            data = data_loader.load_latest_data()
            logger.info(f"Fetched {len(data)} candles for {symbol} ({timeframe})")

if __name__ == "__main__":
    main()