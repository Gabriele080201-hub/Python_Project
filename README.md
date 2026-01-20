# Predictive Maintenance for Aircraft Engines

A real-time monitoring system that predicts Remaining Useful Life (RUL) for aircraft engines using a Spatial-Temporal Graph Neural Network Transformer.

## Overview

This project uses deep learning to predict when aircraft engines will fail, allowing for proactive maintenance. The system processes sensor data from multiple engines and provides real-time RUL predictions through an interactive web dashboard.

## Features

- **Real-time Monitoring**: Track multiple engines simultaneously
- **RUL Prediction**: Predict remaining useful life in operating cycles
- **Interactive Dashboard**: Streamlit-based web interface
- **Sensor Visualization**: View trends for all engine sensors
- **Fleet Management**: Monitor entire fleet status at a glance

## Project Structure

```
.
├── model/                          # Neural network and scaling
│   ├── GNN_Transformer/            # Model architecture
│   │   ├── st_gnn_transformer.py   # Main model
│   │   ├── layers.py               # Graph convolution layer
│   │   ├── positional_encoding.py  # Positional encoding
│   │   └── losses.py               # Loss functions
│   └── scaling.py                  # TimeSeriesScaler class
├── inference/                      # Inference components
│   ├── data_source.py              # Data streaming simulation
│   ├── engine_state.py             # Individual engine state
│   ├── engine_manager.py           # Prediction manager
│   ├── fleet_controller.py         # Fleet orchestration
│   └── load_bundle.py              # Model loading
├── ui/                             # Dashboard UI components
│   ├── config.py                   # Centralized configuration
│   ├── charts.py                   # Plotly chart functions
│   └── components.py               # Streamlit UI components
├── preprocessing/                  # Legacy compatibility (for scaler loading)
│   └── scaling.py                  # Redirects to model/scaling.py
├── artifacts/                      # Trained model files
│   ├── best_model.pt               # Model checkpoint
│   └── scaler.joblib               # Data scaler
├── data/                           # Test data
│   └── test_FD001.txt              # NASA C-MAPSS test set
├── streamlit_app.py                # Web dashboard entry point
└── requirements.txt                # Python dependencies
```

## Model Architecture

The model combines two powerful approaches:

1. **Graph Neural Network (GNN)**: Learns relationships between different sensors
2. **Transformer**: Learns temporal patterns in sensor data over time

This combination allows the model to understand both:
- Which sensors are related (spatial relationships)
- How sensor values change over time (temporal patterns)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit dashboard:
```bash
streamlit run streamlit_app.py
```

The dashboard will open in your browser with:
- **Manual Step**: Advance simulation one time step
- **Start/Stop Simulation**: Auto-run simulation
- **Reset System**: Clear all data and restart

## Data

Uses the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset:
- 21 sensor measurements
- 3 operational settings
- Multiple engine units
- Run-to-failure time series data

## How It Works

1. **Data Streaming**: Sensor data is fed to the system one cycle at a time
2. **Sliding Window**: The model requires 30 consecutive cycles to make a prediction
3. **Preprocessing**: Data is normalized using the trained scaler
4. **Prediction**: The model outputs RUL in remaining operating cycles
5. **Visualization**: Results are displayed in real-time on the dashboard

## Technical Details

- **Window Size**: 30 time steps
- **Features**: 14 sensor measurements (selected subset)
- **Model**: GNN + Transformer architecture
- **Framework**: PyTorch
- **UI**: Streamlit

## License

This project is provided as-is for educational and research purposes.
