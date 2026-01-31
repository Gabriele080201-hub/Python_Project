"""
Predictive Maintenance Dashboard

Entry point for the Streamlit application.
Monitors an aircraft engine fleet in real-time
and predicts the Remaining Useful Life (RUL).

To run:
    streamlit run app.py
"""

import streamlit as st

from ui import components
from ui.config import PAGE_CONFIG, PAGE_TITLE


# Configure page
st.set_page_config(**PAGE_CONFIG)

# Initialize state (loads model and data)
components.initialize_state()

# Header
st.title(PAGE_TITLE)
st.markdown("Real-time monitoring system for engine fleet.")

# Control bar
components.render_control_bar()

st.divider()

# Fleet status table
components.render_fleet_table()

st.divider()

# Selected engine detail
components.render_engine_detail()

# Handle automatic simulation
components.handle_autorun()
