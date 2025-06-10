# shared_setup.py

"""
Shared setup module for all Streamlit pages.
This ensures consistent theming and configuration across all pages.
"""

import sys
import os
import warnings
import logging


def setup_environment_only():
    """
    Setup environment without any Streamlit commands.
    This can be called before st.set_page_config().
    """
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", message=".*torch.classes.*")
    logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

    # Add backend to path
    backend_path = os.path.join(os.path.dirname(__file__), 'backend')
    if backend_path not in sys.path:
        sys.path.append(backend_path)


def setup_page_after_config():
    """
    Setup function that should be called AFTER st.set_page_config().
    This applies theme and initializes state.
    """
    # Apply theme and configuration
    from config import apply_complete_theme
    from utils import AppState

    # Initialize state if not already done
    AppState.initialize()

    # Apply theme
    apply_complete_theme()


def setup_main_page():
    """
    Setup specifically for the main page (streamlit_app.py)
    """
    import streamlit as st
    from config import app_config

    # Environment setup first
    setup_environment_only()

    # Page config must be first Streamlit command
    st.set_page_config(
        page_title=app_config.PAGE_TITLE,
        page_icon=app_config.PAGE_ICON,
        layout=app_config.LAYOUT
    )

    # Then apply theme and state
    setup_page_after_config()