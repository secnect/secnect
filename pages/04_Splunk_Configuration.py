# pages/04_splunk_configuration.py
from shared_setup import setup_environment_only
setup_environment_only()

import streamlit as st
st.set_page_config(page_title="Splunk Configuration", layout="wide")

from shared_setup import setup_page_after_config
setup_page_after_config()

from page_classes.splunk_config_page import splunk_config_page
splunk_config_page.render()