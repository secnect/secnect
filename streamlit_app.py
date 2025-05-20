#Todo:
#Minor things: 
#   Adding back loading bar during analysis
#Major things:
#   Find what is similar to the One log and what is discarded
#   Based on what its decided
#   Field identification 


import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import io

# Import backend utilities (excluding load_positive_examples)
from model.model_utils.model_utils import (
    load_model,
    normalize_text,
    compute_similarities,
    build_results_df,
    load_positive_examples,
    get_similarity_breakdown,
    enhanced_similarity,
    extract_log_fields,
    highlight_text
)

# Set page config
st.set_page_config(
    page_title="Secnect - Failed Login Detector",
    page_icon="🔐",
    layout="wide"
)

# Simple CSS for the company name and beta tag
st.markdown("""
<style>
    .company-header {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .company-name {
        font-size: 24px;
        font-weight: bold;
        color: #1E3A8A;
        margin-right: 10px;
    }
    .beta-tag {
        background-color: #EF4444;
        color: white;
        font-size: 12px;
        font-weight: bold;
        padding: 3px 8px;
        border-radius: 4px;
        text-transform: uppercase;
    }
</style>

<div class="company-header">
    <div class="company-name">Secnect</div>
    <div class="beta-tag">Beta</div>
</div>
""", unsafe_allow_html=True)

# Main app
def main():
    st.title("🔐 Failed Login Detector")
    st.markdown("""
    This app uses semantic similarity to identify potential failed login events in log files.
    Upload your log file and we'll rank each line by its similarity to known failed login patterns.
    
    For testing purposes, sample log files can be found in the [LogHub repository](https://github.com/logpai/loghub) on GitHub, which contains a collection of system logs from various technologies.
    """)
    
    # Load positive examples
    positive_examples_df = load_positive_examples()
    if positive_examples_df is None:
        return
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.6,
        step=0.05,
        help="Log lines with similarity scores above this threshold will be highlighted"
    )
    
    top_n = st.sidebar.number_input(
        "Number of top results to display",
        min_value=5,
        max_value=100,
        value=20,
        step=5
    )
    
    # File upload
    st.header("📁 Upload Log File")
    uploaded_file = st.file_uploader(
        "Choose a log file", 
        type=['log', 'txt', 'csv'],
        help="Upload a log file in .log, .txt, or .csv format"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                log_lines = df.iloc[:, -1].tolist() if len(df.columns) > 1 else df.iloc[:, 0].tolist()
            else:
                content = uploaded_file.read().decode('utf-8')
                log_lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            st.success(f"Successfully loaded {len(log_lines)} log lines")
            
            # Show sample of uploaded file
            with st.expander("View first 5 log lines"):
                for i, line in enumerate(log_lines[:5]):
                    st.text(f"{i+1}: {line}")
            
            # Process button
            if st.button("🔍 Analyze Log File", type="primary"):
                with st.spinner("Analyzing log file..."):
                    # Load model
                    progress_bar = st.progress(0)
                    model = load_model()
                    
                    # Normalize texts
                    positive_texts = positive_examples_df['normalized_log'].tolist()
                    normalized_log_lines = [normalize_text(line) for line in log_lines]
                    progress_bar = st.progress(45)
                    # Compute similarities
                    max_similarities, most_similar_idx = compute_similarities(
                        model, positive_texts, normalized_log_lines
                    )
                    progress_bar = st.progress(100)
                    # Build and sort results
                    results_df = build_results_df(
                        log_lines,
                        normalized_log_lines,
                        max_similarities,
                        most_similar_idx,
                        positive_examples_df
                    )
                    
                    # Display results
                    st.header("📊 Analysis Results")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Log Lines", len(results_df))
                    with col2:
                        st.metric("High Confidence", len(results_df[results_df['max_similarity_score'] >= confidence_threshold]))
                    with col3:
                        st.metric("Mean Similarity", f"{results_df['max_similarity_score'].mean():.3f}")
                    with col4:
                        st.metric("Max Similarity", f"{results_df['max_similarity_score'].max():.3f}")
                    
                    # Top results
                    st.subheader(f"Top {top_n} Most Likely Failed Login Events")
                    for _, row in results_df.head(top_n).iterrows():
                        
                        score = row['max_similarity_score']
                        color = "red" if score >= confidence_threshold else "orange"
                        # Get similarity breakdown
                        breakdown = get_similarity_breakdown(
                            model, 
                            row['original_log_line'], 
                            row['most_similar_positive_example']
                        )
                        with st.container():
                            st.markdown(f"**Score:** :{color}[{score:.4f}]")
                            
                            # Display original log with highlights
                            st.markdown("**Original Log:**")
                            highlighted = highlight_text(
                                row['original_log_line'], 
                                breakdown['common_tokens']
                            )
                            st.markdown(highlighted, unsafe_allow_html=True)
                            # Display details in expander
                            with st.expander("Detailed Analysis"):
                                st.markdown("**Matching Components:**")
                                st.write(list(breakdown['common_tokens']))
                                
                                st.markdown("**Unique to This Log:**")
                                st.write(list(breakdown['unique_to_log']))
                                
                                st.markdown("**Unique to Example:**")
                                st.write(list(breakdown['unique_to_example']))
                                
                                st.markdown("**Field Extraction:**")
                                st.write(extract_log_fields(row['original_log_line']))
                            st.divider()
                    
                    # Visualization
                    st.subheader("📈 Similarity Score Distribution")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(results_df['max_similarity_score'], bins=50, edgecolor='black', alpha=0.7)
                    ax.axvline(x=confidence_threshold, color='red', linestyle='--', 
                              label=f'Threshold: {confidence_threshold}')
                    ax.axvline(x=results_df['max_similarity_score'].mean(), color='green', linestyle='--', 
                              label=f"Mean: {results_df['max_similarity_score'].mean():.3f}")
                    ax.set_xlabel('Similarity Score')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Similarity Scores')
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Threshold analysis
                    st.subheader("🎯 Threshold Analysis")
                    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    threshold_data = []
                    for threshold in thresholds:
                        count = len(results_df[results_df['max_similarity_score'] >= threshold])
                        percentage = (count / len(results_df)) * 100
                        threshold_data.append({
                            'Threshold': threshold,
                            'Count': count,
                            'Percentage': f"{percentage:.1f}%"
                        })
                    threshold_df = pd.DataFrame(threshold_data)
                    st.dataframe(threshold_df, use_container_width=True)
                    
                    # Download results
                    st.subheader("💾 Download Results")
                    dcol1, dcol2 = st.columns(2)
                    with dcol1:
                        csv_all = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Full Results (CSV)",
                            data=csv_all,
                            file_name="failed_login_analysis_results.csv",
                            mime="text/csv"
                        )
                    with dcol2:
                        high_conf_df = results_df[results_df['max_similarity_score'] >= confidence_threshold]
                        csv_high = high_conf_df.to_csv(index=False)
                        st.download_button(
                            label=f"Download High Confidence Results (Score >= {confidence_threshold})",
                            data=csv_high,
                            file_name=f"high_confidence_failed_logins_{confidence_threshold}.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. Upload your log file
    2. The app normalizes text by removing timestamps, IPs, and numbers
    3. Uses Sentence-BERT to compute semantic embeddings
    4. Calculates cosine similarity with known failed login patterns
    5. Ranks log lines by similarity score
    """)

if __name__ == "__main__":
    main()
