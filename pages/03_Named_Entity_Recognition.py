# pages/03_NAMED_ENTITY_RECOGNITION.py

from shared_setup import setup_environment_only
setup_environment_only()

import streamlit as st
st.set_page_config(page_title="Named Entity Recognition", layout="wide")

from shared_setup import setup_page_after_config
setup_page_after_config()

from page_classes.named_entity_recognition_page import named_entity_recognition_page
named_entity_recognition_page.render()