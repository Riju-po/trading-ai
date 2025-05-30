import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.quantum_model import QuantumPricePredictor
from src.models.transformer_model import PriceTransformer
from src.models.tcn_model import CryptoTCN
from src.models.gnn_model import GraphNeuralNetwork
from src.models.exotic_models import ChaosTheoryAnalyzer, SwarmIntelligence, FractalAnalyzer, TopologicalAnalyzer
from src.fusion.ensemble import SignalFusion
from src.utils.data_loader import OHLCVDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def benchmark_model(model_name, model_instance, data, iterations=5):
    """Benchmark a model's performance.
    
    Args:
        model_name: Name of the model
        model_instance: Instance of the model
        data: OHLCV data for prediction
        iterations: Number of iterations to run
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking {model_name}...")
    
    # Warm-up run
    if hasattr(model_instance, 'predict'):
        model_instance.predict(data)
    elif hasattr(model_instance, 'analyze'):
        model_instance.analyze(data)
    elif hasattr(model_instance, 'optimize'):
        model_instance.optimize(data)
    elif hasattr(model_instance, 'extract_features'):
        model_instance.extract_features(data)
    
    # Benchmark runs
    execution_times = []
    for i in range(iterations):
        start_time = time.time()
        
        if hasattr(model_instance, 'predict'):
            result = model_instance.predict(data)
        elif hasattr(model_instance, 'analyze'):
            result = model_instance.analyze(data)
        elif hasattr(model_instance, 'optimize'):
            result = model_instance.optimize(data)
        elif hasattr(model_instance, 'extract_features'):
            result = model_instance.extract_features(data)
            
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)
        
        logger.info(f"  Run {i+1}/{iterations}: {execution_time:.4f} seconds")
    
    # Calculate statistics
    avg_time = np.mean(execution_times)
    min_time = np.min(execution_times)
    max_time = np.max(execution_times)
    std_time = np.std(execution_times)
    
    logger.info(f"  Average: {avg_time:.4f} seconds")
    logger.info(f"  Min: {min_time:.4f} seconds")
    logger.info(f"  Max: {max_time:.4f} seconds")
    logger.info(f"  Std Dev: {std_time:.4f} seconds")
    
    return {
        "model": model_name,
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "std_time": std_time,
        "all_times": execution_times
    }

def benchmark_fusion(fusion_instance, predictions, iterations=5):
    """Benchmark the signal fusion performance.
    
    Args:
        fusion_instance: Instance of SignalFusion
        predictions: Dictionary of model predictions
        iterations: Number of iterations to run
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info("Benchmarking Signal Fusion...")
    
    # Warm-up run
    fusion_instance.fuse(predictions)
    
    # Benchmark runs
    execution_times = []
    for i in range(iterations):
        start_time = time.time()
        fusion_instance.fuse(predictions)
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)
        
        logger.info(f"  Run {i+1}/{iterations}: {execution_time:.4f} seconds")
    
    # Calculate statistics
    avg_time = np.mean(execution_times)
    min_time = np.min(execution_times)
    max_time = np.max(execution_times)
    std_time = np.std(execution_times)
    
    logger.info(f"  Average: {avg_time:.4f} seconds")
    logger.info(f"  Min: {min_time:.4f} seconds")
    logger.info(f"  Max: {max_time:.4f} seconds")
    logger.info(f"  Std Dev: {std_time:.4f} seconds")
    
    return {
        "model": "Signal Fusion",
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "std_time": std_time,
        "all_times": execution_times
    }

def plot_benchmark_results(results, output_dir="data"):
    """Plot benchmark results and save to file.
    
    Args:
        results: List of benchmark result dictionaries
        output_dir: Directory to save the plot
    """
    models = [r["model"] for r in results]
    avg_times = [r["avg_time"] for r in results]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, avg_times, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('Model Performance Benchmark')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}s', ha='center', va='bottom')
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path)
    logger.info(f"Benchmark plot saved to {plot_path}")
    
    # Save raw data
    json_path = os.path.join(output_dir, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Benchmark data saved to {json_path}")

def main():
    logger.info("Starting performance benchmark")
    
    # 1. Load OHLCV data
    data_loader = OHLCVDataLoader(timeframe="15m", lookback_candles=200)
    ohlcv_data = data_loader.load_latest_data()
    
    # 2. Initialize models
    quantum_model = QuantumPricePredictor(n_qubits=8, layers=6, entanglement="full")
    transformer_model = PriceTransformer(d_model=512, n_heads=16, n_layers=8, seq_length=100)
    tcn_model = CryptoTCN(channels=[64, 128, 256, 512, 1024], kernel_size=3, 
                         dilations=[1, 2, 4, 8, 16, 32], dropout=0.2)
    gnn_model = GraphNeuralNetwork()
    
    # 3. Initialize exotic algorithms
    chaos_analyzer = ChaosTheoryAnalyzer()
    swarm_intelligence = SwarmIntelligence()
    fractal_analyzer = FractalAnalyzer()
    topological_analyzer = TopologicalAnalyzer()
    
    # 4. Benchmark individual models
    results = []
    results.append(benchmark_model("Quantum Model", quantum_model, ohlcv_data))
    results.append(benchmark_model("Transformer Model", transformer_model, ohlcv_data))
    results.append(benchmark_model("TCN Model", tcn_model, ohlcv_data))
    results.append(benchmark_model("GNN Model", gnn_model, ohlcv_data))
    results.append(benchmark_model("Chaos Theory Analyzer", chaos_analyzer, ohlcv_data))
    results.append(benchmark_model("Swarm Intelligence", swarm_intelligence, ohlcv_data))
    results.append(benchmark_model("Fractal Analyzer", fractal_analyzer, ohlcv_data))
    results.append(benchmark_model("Topological Analyzer", topological_analyzer, ohlcv_data))
    
    # 5. Run models to get predictions for fusion benchmark
    quantum_pred = quantum_model.predict(ohlcv_data)
    transformer_pred = transformer_model.predict(ohlcv_data)
    tcn_pred = tcn_model.predict(ohlcv_data)
    gnn_pred = gnn_model.predict(ohlcv_data)
    chaos_pred = chaos_analyzer.analyze(ohlcv_data)
    swarm_pred = swarm_intelligence.optimize(ohlcv_data)
    fractal_pred = fractal_analyzer.analyze(ohlcv_data)
    topological_pred = topological_analyzer.extract_features(ohlcv_data)
    
    # 6. Benchmark fusion
    fusion = SignalFusion(weights={
        'quantum': 0.25,
        'transformer': 0.30,
        'tcn': 0.25,
        'gnn': 0.10,
        'chaos': 0.05,
        'swarm': 0.05
    })
    
    predictions = {
        'quantum': quantum_pred,
        'transformer': transformer_pred,
        'tcn': tcn_pred,
        'gnn': gnn_pred,
        'chaos': chaos_pred,
        'swarm': swarm_pred,
        'fractal': fractal_pred,
        'topological': topological_pred
    }
    
    results.append(benchmark_fusion(fusion, predictions))
    
    # 7. Plot and save results
    plot_benchmark_results(results)
    
    logger.info("Benchmark completed")

if __name__ == "__main__":
    main()