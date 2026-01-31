"""
Components Module - UI components for the Streamlit dashboard.

Contains functions to render the user interface
and manage session state.
"""

import os
import time

import pandas as pd
import streamlit as st

from predictor import Predictor
from simulation import Fleet, DataSource, COLUMN_NAMES
from ui import charts
from ui.config import DATA_PATH, UPDATE_DELAY


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def initialize_state():
    """
    Initialize Streamlit session state.

    Loads the model, data, and creates the Fleet for simulation.
    Runs only once at application startup.
    """
    if "fleet" not in st.session_state:
        with st.spinner("Starting system..."):
            st.session_state.fleet = _create_fleet()
            st.session_state.autorun = False


def _create_fleet():
    """
    Create and configure the Fleet for simulation.

    Returns:
        Fleet: Fleet object ready for simulation
    """
    # Create Predictor (loads the model)
    predictor = Predictor()

    # Check that data file exists
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found: {DATA_PATH}")
        st.stop()

    # Load NASA C-MAPSS dataset
    df = pd.read_csv(DATA_PATH, sep=" ", header=None)
    df = df.iloc[:, :26]  # Remove empty columns
    df.columns = COLUMN_NAMES

    # Create DataSource
    data_source = DataSource(df, predictor.feature_cols)

    # Create and return Fleet
    return Fleet(predictor, data_source)


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_control_bar():
    """Render the control bar with buttons."""
    col1, col2, col3, _ = st.columns([1, 1, 1, 4])

    with col1:
        if st.button("Manual Step", use_container_width=True):
            st.session_state.fleet.step()

    with col2:
        label = "Stop" if st.session_state.autorun else "Start Simulation"
        if st.button(label, use_container_width=True):
            st.session_state.autorun = not st.session_state.autorun

    with col3:
        if st.button("Reset", type="primary", use_container_width=True):
            st.session_state.fleet.reset()
            st.session_state.autorun = False
            st.rerun()


def render_fleet_table():
    """Render the fleet status table."""
    st.subheader("Fleet Status")

    fleet_data = st.session_state.fleet.get_status()

    if not fleet_data:
        st.info("No data available. Start the simulation.")
        return

    # Create and format DataFrame
    df = pd.DataFrame(fleet_data)

    if "rul_prediction" in df.columns:
        df["rul_prediction"] = pd.to_numeric(df["rul_prediction"], errors="coerce")
        df = df.sort_values(by="rul_prediction", ascending=True, na_position="last")

    # Display with color gradient
    st.dataframe(
        df.style
        .background_gradient(subset=["rul_prediction"], cmap="RdYlGn", vmin=0, vmax=150)
        .format({"rul_prediction": "{:.2f}"}, na_rep="-"),
        use_container_width=True,
        height=350,
    )


def render_engine_detail():
    """Render the engine detail section."""
    st.subheader("Engine Detail")

    # Get available engine IDs
    available_ids = st.session_state.fleet.get_engine_ids()

    if not available_ids:
        st.write("Waiting for data...")
        return

    # Engine selection
    selected_id = st.selectbox("Select engine:", available_ids)

    # Get engine history
    engine_df = st.session_state.fleet.get_engine_history(selected_id)

    if engine_df.empty:
        st.warning("Insufficient data for selected engine.")
        return

    # RUL trend chart
    st.plotly_chart(charts.create_rul_chart(engine_df), use_container_width=True)

    # Sensor grid
    st.markdown("##### Sensor Telemetry")
    sensors = st.session_state.fleet.feature_cols
    st.plotly_chart(
        charts.create_sensor_grid(engine_df, sensors_to_plot=sensors),
        use_container_width=True
    )


# =============================================================================
# AUTORUN
# =============================================================================

def handle_autorun():
    """Handle the automatic simulation loop."""
    if st.session_state.autorun:
        time.sleep(UPDATE_DELAY)
        st.session_state.fleet.step()
        st.rerun()
