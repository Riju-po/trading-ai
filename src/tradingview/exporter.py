import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from src.signal_generation.signal import AISignal

class TradingViewExporter:
    """Exporter for TradingView integration."""
    
    def __init__(self, signals_json_path: str = "data/signals.json", 
                 tradingview_js_path: str = "tradingview/signals.js"):
        """Initialize the TradingView exporter.
        
        Args:
            signals_json_path: Path to save the JSON signals file
            tradingview_js_path: Path to save the TradingView JavaScript file
        """
        self.signals_json_path = signals_json_path
        self.tradingview_js_path = tradingview_js_path
        self.logger = logging.getLogger(__name__)
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(signals_json_path), exist_ok=True)
        os.makedirs(os.path.dirname(tradingview_js_path), exist_ok=True)
    
    def export_signals(self, signals: List[AISignal], quantum_coherence: float = 0.0,
                      transformer_attention: Optional[List[Any]] = None,
                      tcn_activations: Optional[List[Any]] = None,
                      chaos_level: float = 0.0):
        """Export signals to TradingView format.
        
        Args:
            signals: List of AISignal objects
            quantum_coherence: Quantum coherence metric
            transformer_attention: Transformer attention weights
            tcn_activations: TCN activation patterns
            chaos_level: Chaos level metric
        """
        self.logger.info(f"Exporting {len(signals)} signals to TradingView format")
        
        # Create the signals data structure
        signals_data = {
            "lastUpdate": datetime.now().isoformat(),
            "signals": [signal.to_dict() for signal in signals],
            "modelMetrics": {
                "quantumCoherence": quantum_coherence,
                "transformerAttention": transformer_attention if transformer_attention else [],
                "tcnActivations": tcn_activations if tcn_activations else [],
                "chaosLevel": chaos_level
            }
        }
        
        # Save as JSON
        with open(self.signals_json_path, "w") as f:
            json.dump(signals_data, f, indent=2)
        
        # Save as JavaScript for TradingView
        with open(self.tradingview_js_path, "w") as f:
            f.write(f"window.AISignals = {json.dumps(signals_data, indent=2)};")
        
        self.logger.info(f"Signals exported to {self.signals_json_path} and {self.tradingview_js_path}")
        
        # Generate Pine Script indicator
        self._generate_pine_script(signals_data)
    
    def _generate_pine_script(self, signals_data: Dict[str, Any]):
        """Generate Pine Script indicator for TradingView.
        
        Args:
            signals_data: Signals data dictionary
        """
        pine_script = """//@version=5
indicator("AI Quantum Trading Signals", overlay=true)

// Input parameters
showSignals = input.bool(true, "Show Buy/Sell Signals")
showProbability = input.bool(true, "Show Quantum Probability Cloud")
showActivation = input.bool(true, "Show Neural Activation Heatmap")
showRiskReward = input.bool(true, "Show Risk/Reward Visualization")
showRegime = input.bool(true, "Show Market Regime Indicator")

// Colors
color buyColor = color.rgb(0, 150, 136, 80)
color sellColor = color.rgb(255, 82, 82, 80)
color exitColor = color.rgb(156, 39, 176, 80)
color regimeTrending = color.rgb(0, 150, 136)
color regimeRanging = color.rgb(255, 152, 0)
color regimeVolatile = color.rgb(244, 67, 54)

// Variables for signal detection
var float lastSignalTime = 0
var string currentSignal = "NONE"
var float signalConfidence = 0.0
var float signalStopLoss = 0.0
var float signalTakeProfit = 0.0
var string marketRegime = "trending"

// Function to detect signals (in a real implementation, this would fetch from the AI system)
// This is a placeholder - in reality you would use external data or API
getSignal() =>
    // Placeholder logic - in production this would connect to your AI system
    signalType = "NONE"
    confidence = 0.0
    stopLevel = 0.0
    takeLevel = 0.0
    regime = "trending"
    
    // Return values
    [signalType, confidence, stopLevel, takeLevel, regime]

// Get signal for current bar
[signalType, confidence, stopLevel, takeLevel, regime] = getSignal()

// Update signal variables if new signal detected
if (signalType != "NONE" and confidence >= 0.85)
    currentSignal := signalType
    signalConfidence := confidence
    signalStopLoss := stopLevel
    signalTakeProfit := takeLevel
    marketRegime := regime
    lastSignalTime := time

// Plot signals
if (showSignals)
    plotshape(
        currentSignal == "BUY" and barstate.isconfirmed, 
        title="Buy Signal", 
        location=location.belowbar, 
        color=buyColor, 
        style=shape.triangleup, 
        size=size.normal
    )
    
    plotshape(
        currentSignal == "SELL" and barstate.isconfirmed, 
        title="Sell Signal", 
        location=location.abovebar, 
        color=sellColor, 
        style=shape.triangledown, 
        size=size.normal
    )
    
    plotshape(
        currentSignal == "EXIT" and barstate.isconfirmed, 
        title="Exit Signal", 
        location=location.abovebar, 
        color=exitColor, 
        style=shape.circle, 
        size=size.normal
    )

// Plot stop loss and take profit levels
if (showRiskReward and currentSignal != "NONE")
    plot(
        currentSignal == "BUY" ? signalStopLoss : na, 
        title="Stop Loss", 
        color=color.red, 
        style=plot.style_circles, 
        linewidth=2
    )
    
    plot(
        currentSignal == "BUY" ? signalTakeProfit : na, 
        title="Take Profit", 
        color=color.green, 
        style=plot.style_circles, 
        linewidth=2
    )
    
    plot(
        currentSignal == "SELL" ? signalStopLoss : na, 
        title="Stop Loss", 
        color=color.red, 
        style=plot.style_circles, 
        linewidth=2
    )
    
    plot(
        currentSignal == "SELL" ? signalTakeProfit : na, 
        title="Take Profit", 
        color=color.green, 
        style=plot.style_circles, 
        linewidth=2
    )

// Market regime indicator
if (showRegime)
    regimeColor = marketRegime == "trending" ? regimeTrending : 
                 marketRegime == "ranging" ? regimeRanging : regimeVolatile
    
    hline(0, "Market Regime", color=color.new(regimeColor, 90))
    bgcolor(color.new(regimeColor, 95))

// Confidence visualization
if (showProbability and currentSignal != "NONE")
    confidenceColor = currentSignal == "BUY" ? buyColor : 
                     currentSignal == "SELL" ? sellColor : exitColor
    
    plot(
        close, 
        title="Confidence Cloud", 
        color=color.new(confidenceColor, 100 - math.round(signalConfidence * 100)), 
        style=plot.style_area, 
        linewidth=0
    )

// Display signal information
var table signalTable = table.new(position.top_right, 5, 5, color.black, color.white, 1, color.gray, 1)

if (barstate.islast)
    table.cell(signalTable, 0, 0, "AI Trading Signals", bgcolor=color.black, text_color=color.white)
    table.cell(signalTable, 0, 1, "Signal: " + currentSignal, bgcolor=color.black, text_color=color.white)
    table.cell(signalTable, 0, 2, "Confidence: " + str.tostring(math.round(signalConfidence * 100)) + "%", bgcolor=color.black, text_color=color.white)
    table.cell(signalTable, 0, 3, "Market Regime: " + marketRegime, bgcolor=color.black, text_color=color.white)
    
    if (currentSignal == "BUY" or currentSignal == "SELL")
        rr = math.abs((signalTakeProfit - close) / (close - signalStopLoss))
        table.cell(signalTable, 0, 4, "Risk/Reward: 1:" + str.tostring(math.round(rr * 10) / 10), bgcolor=color.black, text_color=color.white)
"""
        
        # Save Pine Script
        with open("tradingview/ai_signals_indicator.pine", "w") as f:
            f.write(pine_script)
        
        self.logger.info("Pine Script indicator generated")