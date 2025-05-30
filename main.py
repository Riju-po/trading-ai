import logging
from datetime import datetime
from src.models.quantum_model import QuantumPricePredictor
from src.models.transformer_model import PriceTransformer
from src.models.tcn_model import CryptoTCN
from src.models.gnn_model import GraphNeuralNetwork
from src.models.exotic_models import ChaosTheoryAnalyzer, SwarmIntelligence, FractalAnalyzer, TopologicalAnalyzer
from src.signal_generation.signal import AISignal
from src.fusion.ensemble import SignalFusion
from src.risk_management.risk_manager import RiskManager
from src.tradingview.exporter import TradingViewExporter
from src.utils.data_loader import OHLCVDataLoader
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# At the beginning of main()
def main():
    logger.info("Starting AI Signal Generator")
    
    # Check if we're running in simulation mode
    simulation_mode = os.environ.get('USE_SIMULATION_MODE', 'false').lower() == 'true'
    if simulation_mode:
        logger.info("Running in simulation mode")
    
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
    
    # 4. Run models in parallel and get predictions
    quantum_pred = quantum_model.predict(ohlcv_data)
    transformer_pred = transformer_model.predict(ohlcv_data)
    tcn_pred = tcn_model.predict(ohlcv_data)
    gnn_pred = gnn_model.predict(ohlcv_data)
    chaos_pred = chaos_analyzer.analyze(ohlcv_data)
    swarm_pred = swarm_intelligence.optimize(ohlcv_data)
    fractal_pred = fractal_analyzer.analyze(ohlcv_data)
    topological_pred = topological_analyzer.extract_features(ohlcv_data)
    
    # 5. Fuse predictions
    fusion = SignalFusion(weights={
        'quantum': 0.25,
        'transformer': 0.30,
        'tcn': 0.25,
        'gnn': 0.10,
        'chaos': 0.05,
        'swarm': 0.05
    })
    
    fused_predictions = fusion.fuse({
        'quantum': quantum_pred,
        'transformer': transformer_pred,
        'tcn': tcn_pred,
        'gnn': gnn_pred,
        'chaos': chaos_pred,
        'swarm': swarm_pred,
        'fractal': fractal_pred,
        'topological': topological_pred
    })
    
    # 6. Generate signals with risk management
    risk_manager = RiskManager()
    signals = []
    
    for pred in fused_predictions:
        if pred['confidence'] >= 0.85 and pred['risk_reward_ratio'] >= 2.0:
            stop_loss, take_profit = risk_manager.calculate_levels(
                pred, ohlcv_data, quantum_pred['probability_distribution']
            )
            
            signal = AISignal(
                timestamp=datetime.now(),
                signal_type=pred['signal_type'],
                confidence=pred['confidence'],
                predicted_move=pred['predicted_move'],
                risk_reward_ratio=pred['risk_reward_ratio'],
                stop_loss=stop_loss,
                take_profit=take_profit,
                quantum_probability=quantum_pred['probability'],
                neural_activation=transformer_pred['activation'],
                market_regime=chaos_pred['regime']
            )
            
            signals.append(signal)
    
    # 7. Export to TradingView format
    if signals:
        exporter = TradingViewExporter()
        exporter.export_signals(signals, 
                               quantum_coherence=quantum_pred['coherence'],
                               transformer_attention=transformer_pred['attention_weights'],
                               tcn_activations=tcn_pred['activations'],
                               chaos_level=chaos_pred['chaos_level'])
        
        logger.info(f"Generated {len(signals)} signals")
    else:
        logger.info("No signals generated that meet criteria")

if __name__ == "__main__":
    main()