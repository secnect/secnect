# pages/02_CORRECTIONS_MANAGEMENT.py

from shared_setup import setup_environment_only
setup_environment_only()

import streamlit as st
st.set_page_config(page_title="Corrections Management", layout="wide")

from shared_setup import setup_page_after_config
setup_page_after_config()

from page_classes.corrections_page import corrections_page
corrections_page.render()