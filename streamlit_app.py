import streamlit as st
import pandas as pd
import numpy as np
import regex as re
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json
import os
import hashlib
from datetime import datetime
from typing import List, Dict, Any

# Import backend utilities
from model.model_utils.model_utils import (
    load_model,
    normalize_text,
    compute_similarities,
    build_results_df,
    load_positive_examples,
    get_similarity_breakdown,
    extract_log_fields,
    highlight_text
)
from model.model_utils.secbert_model import get_secbert_analyzer

# Feedback system setup
FEEDBACK_FILE = Path("feedback") / "secbert_feedback.json"
CORRECTIONS_FILE = Path("feedback") / "corrections.json"

class CorrectionsManager:
    """Manages corrections data and training tracking"""
    
    def __init__(self, corrections_file: Path = CORRECTIONS_FILE):
        self.corrections_file = corrections_file
        self.corrections_data = self.load_corrections()
    
    def load_corrections(self) -> List[Dict]:
        """Load corrections from JSON file"""
        try:
            if self.corrections_file.exists():
                with open(self.corrections_file, 'r') as f:
                    data = json.load(f)
                # Ensure each record has required fields
                for record in data:
                    if 'used_for_training' not in record:
                        record['used_for_training'] = False
                    if 'record_id' not in record:
                        record['record_id'] = hashlib.md5(
                            (record.get('log_text', '') + str(record.get('timestamp', ''))).encode()
                        ).hexdigest()[:12]
                return data
            return []
        except (FileNotFoundError, json.JSONDecodeError) as e:
            st.error(f"Error loading corrections file: {e}")
            return []
    
    def save_corrections(self) -> bool:
        """Save corrections data back to file"""
        try:
            os.makedirs(os.path.dirname(self.corrections_file), exist_ok=True)
            with open(self.corrections_file, 'w') as f:
                json.dump(self.corrections_data, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving corrections: {e}")
            return False
    
    def get_corrections_df(self) -> pd.DataFrame:
        """Convert corrections to DataFrame for display"""
        if not self.corrections_data:
            return pd.DataFrame(columns=[
                'Record ID', 'Log Text', 'Original Prediction', 'User Correction', 
                'Timestamp', 'Used for Training'
            ])
        
        df_data = []
        for record in self.corrections_data:
            df_data.append({
                'Record ID': record.get('record_id', 'N/A'),
                'Log Text': record.get('log_text', '')[:100] + '...' if len(record.get('log_text', '')) > 100 else record.get('log_text', ''),
                'Full Log Text': record.get('log_text', ''),
                'Original Prediction': self._format_prediction(record.get('original_prediction', {})),
                'User Correction': self._format_prediction(record.get('user_correction', {})),
                'Timestamp': record.get('timestamp', 'N/A'),
                'Used for Training': record.get('used_for_training', False)
            })
        
        return pd.DataFrame(df_data)
    
    def _format_prediction(self, prediction: Dict) -> str:
        """Format prediction for display"""
        if not prediction:
            return "N/A"
        label = prediction.get('label', 'Unknown')
        confidence = prediction.get('confidence', 0)
        return f"{label} ({confidence:.3f})"
    
    def mark_as_used(self, record_ids: List[str]) -> bool:
        """Mark specific records as used for training"""
        try:
            for record in self.corrections_data:
                if record.get('record_id') in record_ids:
                    record['used_for_training'] = True
                    record['training_timestamp'] = str(datetime.now())
            return self.save_corrections()
        except Exception as e:
            st.error(f"Error marking records as used: {e}")
            return False
    
    def get_unused_corrections(self) -> List[Dict]:
        """Get corrections that haven't been used for training"""
        return [record for record in self.corrections_data if not record.get('used_for_training', False)]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about corrections"""
        total = len(self.corrections_data)
        used = sum(1 for record in self.corrections_data if record.get('used_for_training', False))
        unused = total - used
        
        # Count by correction type
        correction_types = {}
        for record in self.corrections_data:
            correction = record.get('user_correction', {})
            label = correction.get('label', 'Unknown')
            correction_types[label] = correction_types.get(label, 0) + 1
        
        return {
            'total': total,
            'used': used,
            'unused': unused,
            'correction_types': correction_types
        }

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

def show_corrections_management():
    """Display corrections management interface"""
    st.header("🔧 Corrections Management")
    st.markdown("Manage user corrections and track their usage in training datasets.")
    
    # Initialize corrections manager
    corrections_manager = CorrectionsManager()
    
    # Show statistics
    stats = corrections_manager.get_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Corrections", stats['total'])
    with col2:
        st.metric("Used for Training", stats['used'], delta=f"{stats['used']/max(stats['total'], 1)*100:.1f}%")
    with col3:
        st.metric("Unused", stats['unused'])
    with col4:
        if stats['total'] > 0:
            st.metric("Usage Rate", f"{stats['used']/stats['total']*100:.1f}%")
        else:
            st.metric("Usage Rate", "0%")
    
    # Show correction types distribution
    if stats['correction_types']:
        st.subheader("📊 Correction Types Distribution")
        correction_df = pd.DataFrame(list(stats['correction_types'].items()), 
                                   columns=['Correction Type', 'Count'])
        st.bar_chart(correction_df.set_index('Correction Type'))
    
    # Display corrections table
    st.subheader("📋 All Corrections")
    
    corrections_df = corrections_manager.get_corrections_df()
    
    if corrections_df.empty:
        st.info("No corrections found. Start analyzing logs to generate corrections.")
        return
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        filter_used = st.selectbox(
            "Filter by training status:",
            ["All", "Used for Training", "Not Used for Training"]
        )
    
    with col2:
        search_text = st.text_input("Search in log text:", placeholder="Enter search term...")
    
    # Apply filters
    filtered_df = corrections_df.copy()
    
    if filter_used == "Used for Training":
        filtered_df = filtered_df[filtered_df['Used for Training'] == True]
    elif filter_used == "Not Used for Training":
        filtered_df = filtered_df[filtered_df['Used for Training'] == False]
    
    if search_text:
        filtered_df = filtered_df[
            filtered_df['Full Log Text'].str.contains(search_text, case=False, na=False)
        ]
    
    st.write(f"Showing {len(filtered_df)} of {len(corrections_df)} corrections")
    
    # Display table with selection
    if len(filtered_df) > 0:
        # Add selection column
        selection_col = st.columns([0.1, 0.9])
        
        with selection_col[0]:
            select_all = st.checkbox("Select All")
        
        with selection_col[1]:
            if st.button("Mark Selected as Used for Training", disabled=len(filtered_df) == 0):
                selected_records = []
                if select_all:
                    selected_records = filtered_df['Record ID'].tolist()
                else:
                    # Get selected records from checkboxes
                    for idx, row in filtered_df.iterrows():
                        if st.session_state.get(f"select_{row['Record ID']}", False):
                            selected_records.append(row['Record ID'])
                
                if selected_records:
                    if corrections_manager.mark_as_used(selected_records):
                        st.success(f"Marked {len(selected_records)} records as used for training!")
                        st.rerun()
                    else:
                        st.error("Failed to update records.")
                else:
                    st.warning("No records selected.")
        
        # Display detailed table
        for idx, row in filtered_df.iterrows():
            with st.expander(f"Record {row['Record ID']} - {row['Original Prediction']} → {row['User Correction']}"):
                
                # Selection checkbox (if not used for training)
                if not row['Used for Training']:
                    st.checkbox(
                        f"Select for training", 
                        key=f"select_{row['Record ID']}",
                        value=select_all
                    )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Log Text:**")
                    st.text_area("", value=row['Full Log Text'], height=100, disabled=True, key=f"log_{row['Record ID']}")
                    st.markdown(f"**Timestamp:** {row['Timestamp']}")
                
                with col2:
                    st.markdown("**Original Prediction:**")
                    st.write(row['Original Prediction'])
                    st.markdown("**User Correction:**")
                    st.write(row['User Correction'])
                    st.markdown(f"**Training Status:** {'✅ Used' if row['Used for Training'] else '❌ Not Used'}")
                
                if row['Used for Training']:
                    st.success("This correction has been used for training.")
    
    # Export functionality
    st.subheader("📤 Export Corrections")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export All Corrections (CSV)"):
            csv = corrections_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"corrections_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        unused_corrections = corrections_manager.get_unused_corrections()
        if unused_corrections:
            if st.button("Export Unused for Training (JSON)"):
                json_data = json.dumps(unused_corrections, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"unused_corrections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# Set page config
st.set_page_config(
    page_title="Secnect - Failed Login Detector",
    page_icon="🔐",
    layout="wide"
)

# CSS header
st.markdown("""
<style>
    .company-header { display: flex; align-items: center; margin-bottom: 20px; }
    .company-name  { font-size: 24px; font-weight: bold; color: #1E3A8A; margin-right: 10px; }
    .beta-tag      { background-color: #EF4444; color: white; font-size: 12px; font-weight: bold;
                      padding: 3px 8px; border-radius: 4px; text-transform: uppercase; }
</style>
<div class="company-header">
    <div class="company-name">Secnect</div>
    <div class="beta-tag">Beta</div>
</div>
""", unsafe_allow_html=True)

# Navigation
page = st.sidebar.selectbox("Choose a page", ["Log Analysis", "Corrections Management"])

# Model selection options
MODEL_OPTIONS = ["SecBERT model (New)"] #, "LLM model (X)"

# Helper to load default similarity model
def get_similarity_model():
    return load_model()

def main():
    if page == "Corrections Management":
        show_corrections_management()
        return
    
    # Initialize session state variables
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    st.title("🔐 Failed Login Detector")
    st.markdown('''
    This app uses semantic similarity to identify potential failed login events in log files.
    Upload your log file and we'll rank each line by its similarity to known failed login patterns.
    
    For testing purposes, sample log files can be found in the [LogHub repository](https://github.com/logpai/loghub) on GitHub, which contains a collection of system logs from various technologies.
    ''')
    
    # Load positive examples for similarity
    positive_examples_df = load_positive_examples()
    if positive_examples_df is None:
        return

    # Sidebar configuration
    st.sidebar.header("Configuration")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.05, help="Log lines with similarity scores above this threshold will be highlighted")
    
    top_n = st.sidebar.number_input("Number of top results to display", 5, 100, 20, 5)
    model_selection = st.sidebar.selectbox("Model selection", MODEL_OPTIONS)

    st.sidebar.subheader("Change log*", divider=True)
    st.sidebar.markdown("*Currently available model is **:blue[SecBERT]**")
    st.sidebar.markdown("*We've added similarity details tailored to each model, and an NER model is coming soon.")
    st.sidebar.markdown("Our model is still under development, and we apologize for any inconvenience.")

    # File uploader
    st.header("📁 Upload Log File")
    uploaded_file = st.file_uploader("Choose a log file", type=['log','txt','csv'], help="Upload a log file in .log, .txt, or .csv format")
    st.write(f"You selected: **{model_selection}**")
    
    if not uploaded_file:
        st.session_state.show_results = False
        return

    # Read lines
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        log_lines = df.iloc[:, -1].tolist() if df.shape[1]>1 else df.iloc[:,0].tolist()
    else:
        text = uploaded_file.read().decode('utf-8', errors='ignore')
        log_lines = [l for l in text.splitlines() if l.strip()]
    st.success(f"Loaded {len(log_lines)} log lines")

    with st.expander("View first 5 log lines"):
        for i, l in enumerate(log_lines[:5]): st.text(f"{i+1}: {l}")

    # Analyze button
    if st.button("🔍 Analyze Log File", type="primary"):
        with st.spinner("Analyzing..."):
            progress = st.progress(0)

            # Branch on SecBERT vs similarity
            if model_selection == "SecBERT model (New)":
                analyzer = get_secbert_analyzer()
                df_results = analyzer.analyze_logs(log_lines, confidence_threshold)
                # Add model column if not present
                if 'model' not in df_results.columns:
                    df_results['model'] = "SecBERT"
            else:
                # Similarity processing
                progress.progress(10)
                sim_model = get_similarity_model()
                normalized = [normalize_text(l) for l in log_lines]
                pos_texts = positive_examples_df['normalized_log'].tolist()
                progress.progress(45)
                sims, idxs = compute_similarities(sim_model, pos_texts, normalized)
                progress.progress(80)
                df_results = build_results_df(log_lines, normalized, sims, idxs, positive_examples_df)
                df_results['model'] = "BERT Similarity"
            
            # Store results in session state
            st.session_state.analysis_results = df_results
            st.session_state.show_results = True
            progress.progress(100)

    # Display results from session state if available
    if st.session_state.show_results and st.session_state.analysis_results is not None:
        df_results = st.session_state.analysis_results
        
        # Display metrics
        st.header("📊 Analysis Results")
        cols = st.columns(4)
        cols[0].metric("Total lines", len(df_results))
        cols[1].metric("Above threshold", len(df_results[df_results['max_similarity_score']>=confidence_threshold]))
        cols[2].metric("Mean score", f"{df_results['max_similarity_score'].mean():.3f}")
        cols[3].metric("Max score", f"{df_results['max_similarity_score'].max():.3f}")

        st.subheader(f"Top {top_n} results")
        for idx, row in df_results.head(top_n).iterrows():
            score = row['max_similarity_score']
            highlight_color = 'red' if score >= confidence_threshold else 'orange'
            st.markdown(f"**Score:** :{highlight_color}[{score:.4f}]")

            # Common display fields
            st.markdown(f"**Original Log Line:**")
            st.text(row['original_log_line'])
            st.markdown(f"**Normalized Log:**")
            st.text(row.get('normalized_log', normalize_text(row['original_log_line'])))

            with st.expander("Analysis Details"):
                # Show prediction and confidence for all models
                prediction = row.get('prediction', 'N/A')
                st.markdown(f"**Prediction:** {prediction}")
                st.markdown(f"**Confidence:** {score:.4f}")
                
                # Model-specific details
                if row.get('model') == "SecBERT":
                    # SecBERT specific details
                    st.markdown("**Model:** SecBERT")
                    if 'explanation' in row:
                        st.markdown("**Explanation:**")
                        st.write(row['explanation'])
                        
                        # If it's a pattern match, display it specially
                        if "Matched" in row['explanation']:
                            st.markdown("**Pattern Matched:**")
                            st.code(row['explanation'].split(": ")[1])
                else:
                    # Original BERT similarity details
                    st.markdown("**Model:** BERT Similarity")
                    if 'most_similar_positive_example' in row:
                        breakdown = get_similarity_breakdown(
                            sim_model, 
                            row['original_log_line'], 
                            row['most_similar_positive_example']
                        )
                        st.markdown("**Common Tokens:**")
                        st.write(breakdown['common_tokens'])
                        st.markdown("**Unique to Log:**")
                        st.write(breakdown['unique_to_log'])
                        st.markdown("**Unique to Example:**")
                        st.write(breakdown['unique_to_example'])
            
            # Add feedback UI for this log line
            add_feedback_ui(
                row['original_log_line'],
                {"label": prediction, "confidence": score},
                idx
            )
            
            st.divider()

        # Distribution plot
        st.subheader("📈 Score Distribution")
        fig, ax = plt.subplots()
        ax.hist(df_results['max_similarity_score'], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(confidence_threshold, linestyle='--', label=f'Threshold: {confidence_threshold}')
        ax.axvline(df_results['max_similarity_score'].mean(), linestyle='--', label=f"Mean: {df_results['max_similarity_score'].mean():.3f}")
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        st.pyplot(fig)

        # Download
        st.subheader("💾 Download Results")
        csv = df_results.to_csv(index=False)
        st.download_button("Download Full Results", data=csv, file_name="results.csv")

    st.markdown("---")
    st.markdown('''
    **How it works:**
    1. Upload your log file
    2. The app normalizes text by removing timestamps, IPs, and numbers
    3. Uses Sentence-BERT to compute semantic embeddings
    4. Calculates cosine similarity with known failed login patterns
    5. Ranks log lines by similarity score
    ''')

if __name__ == "__main__":
    main()