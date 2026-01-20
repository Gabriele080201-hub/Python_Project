"""
Predictive Maintenance Dashboard

Main entry point for the Streamlit application.
Provides real-time monitoring and RUL predictions for aircraft engines.
"""

import streamlit as st

from ui import components
from ui.config import PAGE_CONFIG, PAGE_TITLE

# Configure page
st.set_page_config(**PAGE_CONFIG)

# Initialize state (loads model, scaler, data)
components.initialize_state()

# Page header
st.title(PAGE_TITLE)
st.markdown("Real-time monitoring system for aircraft engine fleet.")

# Control bar (buttons)
components.render_control_bar()

st.divider()

# Fleet status table
components.render_fleet_table()

st.divider()

# Engine detail (charts)
components.render_engine_detail()

# Handle automatic simulation
components.handle_autorun()
