"""
Engine Fleet Monitor - Predictive Maintenance Dashboard

A simple application to monitor aircraft engines and predict
their Remaining Useful Life (RUL) using sensor data.

To run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd

from predictor import Predictor
from simulation import Fleet, DataSource, COLUMN_NAMES


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_resource
def load_fleet():
    """
    Load and initialize the fleet simulation.

    Returns:
        Fleet: Ready fleet object with predictor and data source
    """
    # Load the predictor (ML model)
    predictor = Predictor()

    # Load sensor data from file
    df = pd.read_csv("data/test_FD001.txt", sep=" ", header=None)
    df = df.iloc[:, :26]
    df.columns = COLUMN_NAMES

    # Create data source and fleet
    data_source = DataSource(df, predictor.feature_cols)
    return Fleet(predictor, data_source)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

st.title("Engine Fleet Monitor")
st.write("Monitor aircraft engines and predict their Remaining Useful Life.")

# Load fleet (cached)
fleet = load_fleet()

# --- Simulation Control ---
st.subheader("Simulation")

if st.button("Run 10 Steps"):
    for _ in range(10):
        fleet.step()

# --- Fleet Status ---
st.subheader("Fleet Status")

status = fleet.get_status()

if not status:
    st.info("No data yet. Click 'Run 10 Steps' to start.")
else:
    df_status = pd.DataFrame(status)
    df_status = df_status.sort_values("rul_prediction", ascending=True, na_position="last")
    st.dataframe(df_status, use_container_width=True)

# --- Engine Detail ---
st.subheader("Engine Detail")

engine_ids = fleet.get_engine_ids()

if not engine_ids:
    st.info("No engines available yet.")
else:
    # Select engine
    selected_id = st.selectbox("Select an engine", engine_ids)

    # Get engine object
    engine = fleet.get_engine(selected_id)

    # Show engine info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Cycle", engine.cycle)
    with col2:
        rul_display = f"{engine.rul:.1f}" if engine.rul else "N/A"
        st.metric("Predicted RUL", rul_display)

    # Show RUL history chart
    history = engine.get_history()

    if history:
        st.write("**RUL Prediction Over Time**")
        df_history = pd.DataFrame(history, columns=["cycle", "rul"])
        st.line_chart(df_history.set_index("cycle"))

    # Show single sensor over time
    st.write("**Sensor Data Over Time**")
    df_sensors = fleet.get_engine_history(selected_id)

    if not df_sensors.empty:
        # Let user pick one sensor to display
        sensor = st.selectbox("Select a sensor", fleet.feature_cols)

        # Show sensor chart
        sensor_data = df_sensors[["cycle", sensor]].set_index("cycle")
        st.line_chart(sensor_data)
