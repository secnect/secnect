# streamlit_app.py

"""
Main Streamlit application - Home page.
This serves as the landing page in a native multi-page app.
"""

from shared_setup import setup_main_page

setup_main_page()  # Must be called before other imports

import streamlit as st
from components import header_component
from config import app_config


def main():
    """Main home page."""
    # Render header
    header_component.render()

    # Home page content
    st.title("Security Log Analysis Platform")

    st.markdown("""
    Welcome to the Security Log Analysis Platform. This tool helps you analyze security logs 
    and generate Splunk configurations for better SIEM integration.

    ## Getting Started

    1. **Log Analysis** - Upload and analyze your security logs
    2. **Corrections Management** - Manage and review analysis corrections
    3. **Named Entity Recognition** - Advanced entity extraction (coming soon)
    4. **Splunk Configuration** - Generate Splunk configuration files

    Use the navigation in the sidebar to explore the different pages.
    """)

    # Quick stats or overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Log Formats Supported", "10+")

    with col2:
        st.metric("Analysis Models", "5")

    with col3:
        st.metric("Splunk Configs", "Auto-Generated")


if __name__ == "__main__":
    main()