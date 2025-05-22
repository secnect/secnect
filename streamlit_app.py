import streamlit as st
import pandas as pd
import numpy as np
import regex as re
import matplotlib.pyplot as plt

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

# Model selection options
MODEL_OPTIONS = ["BERT Similarity (Default)", "SecBERT model (New)"] #, "LLM model (X)"

# Helper to load default similarity model
def get_similarity_model():
    return load_model()

# Main function

def main():
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
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.0, 1.0, 0.6, 0.05,
        help="Log lines with similarity scores above this threshold will be highlighted"
    )
    top_n = st.sidebar.number_input("Number of top results to display", 5, 100, 20, 5)
    model_selection = st.sidebar.selectbox("Model selection", MODEL_OPTIONS)
    
    
    st.subheader("Change log*", divider=True)
    st.markdown("*Added model selection, available models are **:orange[BERT Similarity]** (Default) and **:blue[SecBERT] (New)**")
    st.markdown("*We’ve added similarity details tailored to each model, and an NER model is coming soon.")
    st.markdown(" Our models are still under development, and we apologize for any inconvenience.")

    # File uploader
    st.header("📁 Upload Log File")
    uploaded_file = st.file_uploader("Choose a log file", type=['log','txt','csv'], help="Upload a log file in .log, .txt, or .csv format")
    st.write(f"You selected: **{model_selection}**")
    
    
    if not uploaded_file:
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
            if model_selection == "SecBERT model":
                # SecBERT processing
                analyzer = get_secbert_analyzer()
                df_results = analyzer.analyze_logs(log_lines, confidence_threshold)
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
            progress.progress(100)

        # Display metrics
        st.header("📊 Analysis Results")
        cols = st.columns(4)
        cols[0].metric("Total lines", len(df_results))
        cols[1].metric("Above threshold", len(df_results[df_results['max_similarity_score']>=confidence_threshold]))
        cols[2].metric("Mean score", f"{df_results['max_similarity_score'].mean():.3f}")
        cols[3].metric("Max score", f"{df_results['max_similarity_score'].max():.3f}")

        # Show top N
        st.subheader(f"Top {top_n} results")
        for _, row in df_results.head(top_n).iterrows():
            score = row['max_similarity_score']
            highlight_color = 'red' if score>=confidence_threshold else 'orange'
            st.markdown(f"**Score:** :{highlight_color}[{score:.4f}]")

            # Common display fields
            st.markdown(f"**Original Log Line:**")
            st.text(row['original_log_line'])
            st.markdown(f"**Normalized Log:**")
            st.text(row.get('normalized_log', normalize_text(row['original_log_line'])))

            # Branch for SecBERT extra attributes
            if model_selection == "SecBERT model":
                st.markdown(f"**Prediction:** {row['prediction']}")
                st.markdown(f"**Explanation:** {row['explanation']}")
                st.markdown(f"**Model:** {row['model']}")
            else:
                # Similarity breakdown
                breakdown = get_similarity_breakdown(sim_model, row['original_log_line'], row['most_similar_positive_example'])
                with st.expander("Similarity Details"):
                    st.markdown("**Common Tokens:**")
                    st.write(breakdown['common_tokens'])
                    st.markdown("**Unique to Log:**")
                    st.write(breakdown['unique_to_log'])
                    st.markdown("**Unique to Example:**")
                    st.write(breakdown['unique_to_example'])
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

    # Footer

    
    
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
