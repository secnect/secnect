# pages/01_log_analysis.py
from shared_setup import setup_environment_only
setup_environment_only()

import streamlit as st
st.set_page_config(page_title="Log Analysis", layout="wide")

from shared_setup import setup_page_after_config
setup_page_after_config()

from page_classes.log_analysis_page import log_analysis_page
log_analysis_page.render()