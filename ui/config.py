"""
Configuration Module

Centralized configuration for the Streamlit dashboard.
"""

import os

# Page settings
PAGE_TITLE = "NASA C-MAPSS Dashboard"

# Data paths
DATA_PATH = os.path.join("data", "test_FD001.txt")

# Simulation settings
UPDATE_DELAY = 1  # Seconds between automatic updates

# Streamlit page configuration
PAGE_CONFIG = {
    "page_title": PAGE_TITLE,
    "layout": "wide",
    "initial_sidebar_state": "collapsed"
}
