from pathlib import Path
import streamlit as st
import pandas as pd
import hashlib
import json
import os

FEEDBACK_FILE = Path("feedback") / "secbert_feedback.json"
CORRECTIONS_FILE = Path("feedback") / "corrections.json"

def save_feedback(log_text, prediction, user_correction):
    """Save user feedback to a JSON file"""
    feedback_data = {
        "log_text": log_text,
        "original_prediction": prediction,
        "user_correction": user_correction,
        "timestamp": str(pd.Timestamp.now()),
        "record_id": hashlib.md5((log_text + str(pd.Timestamp.now())).encode()).hexdigest()[:12],
        "used_for_training": False
    }
    
    # Create feedback directory if it doesn't exist
    os.makedirs(os.path.dirname(CORRECTIONS_FILE), exist_ok=True)
    
    # Load existing feedback or create new file
    try:
        with open(CORRECTIONS_FILE, 'r') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []
    
    # Append new feedback
    existing_data.append(feedback_data)
    
    # Save back to file
    with open(CORRECTIONS_FILE, 'w') as f:
        json.dump(existing_data, f, indent=2)

def add_feedback_ui(log_text, prediction, idx):
    """Add feedback UI for a single log line"""
    st.markdown("---")
    st.markdown("**Was this classification correct?**")
    
    # Create a stable unique key using MD5 hash of the log text and index
    unique_hash = hashlib.md5((log_text + str(idx)).encode()).hexdigest()
    unique_key = f"feedback_{unique_hash}"
    
    # Initialize session state for this feedback item if not exists
    if unique_key not in st.session_state:
        st.session_state[unique_key] = {
            'submitted': False,
            'show_correction': False
        }
    
    # Create columns for the buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Correct", 
                    key=f"correct_{unique_key}",
                    disabled=st.session_state[unique_key]['submitted']):
            save_feedback(log_text, prediction, prediction)
            st.session_state[unique_key]['submitted'] = True
            st.session_state[unique_key]['show_correction'] = False
            st.success("Thank you for your feedback!")
            st.rerun()
    
    with col2:
        if st.button("❌ Incorrect", 
                    key=f"incorrect_{unique_key}",
                    disabled=st.session_state[unique_key]['submitted']):
            st.session_state[unique_key]['show_correction'] = True
            st.rerun()
        
        if st.session_state[unique_key]['show_correction']:
            correction = st.radio(
                "What should the correct classification be?",
                ["Successful login", "Failed login", "Other login"],
                key=f"correction_{unique_key}"
            )
            
            if st.button("Submit Correction", 
                        key=f"submit_{unique_key}",
                        disabled=st.session_state[unique_key]['submitted']):
                corrected_pred = {
                    "label": correction.lower().replace(" ", "_"),
                    "confidence": 1.0  # User is certain
                }
                save_feedback(log_text, prediction, corrected_pred)
                st.session_state[unique_key]['submitted'] = True
                st.session_state[unique_key]['show_correction'] = False
                st.success("Thank you for your correction!")
                st.rerun()