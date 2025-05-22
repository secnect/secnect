import streamlit as st
import pandas as pd
import numpy as np
import regex as re
import matplotlib.pyplot as plt
import io
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import gc
from tqdm import tqdm

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
    page_title="Secnect - Login Event Analyzer",
    page_icon="🔐",
    layout="wide"
)

class SecBERTAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"Using device: {self.device}")
        
        # Load SecBERT model
        model_name = "jackaduma/SecBERT"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        self.class_names = ["Non-login event", "Failed login", "Successful login"]
        self.explainer = LimeTextExplainer(class_names=self.class_names)
        
        # Login patterns
        self.success_patterns = [
            r"accepted\s+password", r"login\s+successful", 
            r"authentication\s+granted", r"sign[-_]in\s+successful",
            r"credentials\s+accepted", r"result\":\"SUCCESS\"",
            r"consolelogin\":\"success\"", r"status\":\"success\"",
            r"eventid=4624", r"access\s+granted",
            r"authentication\s+succeeded", r"user\s+logged\s+in"
        ]
        
        self.failure_patterns = [
            r"login\s+failed", r"authentication\s+failure",
            r"access\s+denied", r"sign[-_]in\s+denied",
            r"invalid\s+credentials", r"failed\s+password",
            r"result\":\"FAILURE\"", r"result\":\"DENIED\"",
            r"consolelogin\":\"failure\"", r"status\":\"failure\"",
            r"eventid=4625", r"authentication\s+unsuccessful",
            r"user\s+not\s+found", r"account\s+locked"
        ]
        
        self.success_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.success_patterns]
        self.failure_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.failure_patterns]
    
    def _free_memory(self):
        torch.cuda.empty_cache()
        gc.collect()
    
    def is_login_event(self, log_line):
        """Check if the log line contains any login-related keywords"""
        lower_log = log_line.lower()
        login_keywords = [
            'login', 'logon', 'authenticat', 'signin', 'sign-in',
            'session', 'access', 'credential', 'password', 'auth'
        ]
        return any(keyword in lower_log for keyword in login_keywords)
    
    def predict_proba(self, texts):
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**encodings)
                probabilities = torch.softmax(outputs.logits, dim=1)
            
            return probabilities.cpu().numpy()
        except RuntimeError as e:
            st.error(f"Prediction error: {str(e)}")
            return np.array([[1.0, 0.0, 0.0]])  # Default to non-login event
    
    def analyze_logs(self, log_lines, confidence_threshold=0.6):
        results = []
        
        for log in tqdm(log_lines, desc="Analyzing logs"):
            try:
                # First check for clear patterns
                for pattern in self.success_regex:
                    if pattern.search(log):
                        results.append({
                            "original_log_line": log,
                            "normalized_log": normalize_text(log),
                            "prediction": "Successful login",
                            "max_similarity_score": 1.0,
                            "explanation": f"Matched success pattern: '{pattern.pattern}'",
                            "model": "SecBERT"
                        })
                        break
                else:  # No break occurred, no success pattern matched
                    for pattern in self.failure_regex:
                        if pattern.search(log):
                            results.append({
                                "original_log_line": log,
                                "normalized_log": normalize_text(log),
                                "prediction": "Failed login",
                                "max_similarity_score": 1.0,
                                "explanation": f"Matched failure pattern: '{pattern.pattern}'",
                                "model": "SecBERT"
                            })
                            break
                    else:  # No pattern matched
                        if not self.is_login_event(log):
                            results.append({
                                "original_log_line": log,
                                "normalized_log": normalize_text(log),
                                "prediction": "Non-login event",
                                "max_similarity_score": 0.0,
                                "explanation": "No login-related keywords found",
                                "model": "SecBERT"
                            })
                        else:
                            # Use SecBERT for classification
                            probabilities = self.predict_proba(log)[0]
                            predicted_class = np.argmax(probabilities)
                            confidence = probabilities[predicted_class]
                            
                            if confidence < confidence_threshold:
                                prediction = "Ambiguous event"
                            else:
                                prediction = self.class_names[predicted_class]
                            
                            results.append({
                                "original_log_line": log,
                                "normalized_log": normalize_text(log),
                                "prediction": prediction,
                                "max_similarity_score": float(confidence),
                                "explanation": f"SecBERT classification with {confidence:.2f} confidence",
                                "model": "SecBERT"
                            })
            except Exception as e:
                results.append({
                    "original_log_line": log,
                    "normalized_log": normalize_text(log),
                    "prediction": "Error",
                    "max_similarity_score": 0.0,
                    "explanation": f"Processing error: {str(e)}",
                    "model": "SecBERT"
                })
        
        return pd.DataFrame(results)

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
    .highlight-success {
        background-color: #d4edda;
        padding: 2px 4px;
        border-radius: 3px;
    }
    .highlight-failure {
        background-color: #f8d7da;
        padding: 2px 4px;
        border-radius: 3px;
    }
    .highlight-other {
        background-color: #e2e3e5;
        padding: 2px 4px;
        border-radius: 3px;
    }
</style>

<div class="company-header">
    <div class="company-name">Secnect</div>
    <div class="beta-tag">Beta</div>
</div>
""", unsafe_allow_html=True)

def display_results(results_df, confidence_threshold, top_n):
    """Display results in a consistent format across different models"""
    st.header("📊 Analysis Results")
    
    # Calculate metrics
    high_conf = results_df[results_df['max_similarity_score'] >= confidence_threshold]
    high_conf_failed = high_conf[high_conf['prediction'] == 'Failed login']
    high_conf_success = high_conf[high_conf['prediction'] == 'Successful login']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Log Lines", len(results_df))
    with col2:
        st.metric("High Confidence Failed", len(high_conf_failed))
    with col3:
        st.metric("High Confidence Success", len(high_conf_success))
    with col4:
        st.metric("Mean Confidence", f"{results_df['max_similarity_score'].mean():.3f}")
    
    # Top results display
    st.subheader(f"Top {top_n} Most Relevant Login Events")
    
    # Sort by confidence score (descending) and filter high confidence
    sorted_df = results_df.sort_values('max_similarity_score', ascending=False)
    
    for _, row in sorted_df.head(top_n).iterrows():
        score = row['max_similarity_score']
        prediction = row['prediction']
        
        # Determine color based on prediction and confidence
        if prediction == "Failed login":
            color = "red" if score >= confidence_threshold else "orange"
            icon = "❌"
        elif prediction == "Successful login":
            color = "green" if score >= confidence_threshold else "lightgreen"
            icon = "✅"
        else:
            color = "gray"
            icon = "➖"
        
        with st.container():
            st.markdown(f"{icon} **{prediction}** - **Score:** :{color}[{score:.4f}]")
            
            # Display original log with highlights
            st.markdown("**Original Log:**")
            st.code(row['original_log_line'], language='text')
            
            # Display details in expander
            with st.expander("Analysis Details"):
                st.markdown("**Explanation:**")
                st.info(row['explanation'])
                
                if 'most_similar_positive_example' in row:
                    st.markdown("**Similarity Breakdown:**")
                    breakdown = get_similarity_breakdown(
                        None,  # Model not needed for this display
                        row['original_log_line'], 
                        row.get('most_similar_positive_example', '')
                    )
                    st.write(breakdown)
                
                st.markdown("**Extracted Fields:**")
                st.write(extract_log_fields(row['original_log_line']))
            
            st.divider()
    
    # Visualization
    st.subheader("📈 Confidence Score Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate data by prediction type
    failed = results_df[results_df['prediction'] == 'Failed login']['max_similarity_score']
    success = results_df[results_df['prediction'] == 'Successful login']['max_similarity_score']
    other = results_df[~results_df['prediction'].isin(['Failed login', 'Successful login'])]['max_similarity_score']
    
    ax.hist([failed, success, other], 
            bins=30, 
            edgecolor='black', 
            alpha=0.7,
            label=['Failed login', 'Successful login', 'Other events'])
    
    ax.axvline(x=confidence_threshold, color='red', linestyle='--', 
              label=f'Threshold: {confidence_threshold}')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Confidence Scores by Event Type')
    ax.legend()
    st.pyplot(fig)

def main():
    st.title("🔐 Login Event Analyzer")
    st.markdown("""
    This app analyzes log files to identify login events and classify them as successful or failed.
    Upload your log file and select the analysis model to get started.
    
    For testing purposes, sample log files can be found in the [LogHub repository](https://github.com/logpai/loghub).
    """)
    
    # Initialize SecBERT analyzer
    if 'secbert_analyzer' not in st.session_state:
        with st.spinner("Loading SecBERT model..."):
            st.session_state.secbert_analyzer = SecBERTAnalyzer()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.6,
        step=0.05,
        help="Log lines with confidence scores above this threshold will be highlighted"
    )
    
    top_n = st.sidebar.number_input(
        "Number of top results to display",
        min_value=5,
        max_value=100,
        value=20,
        step=5
    )
    
    model_selection = st.sidebar.selectbox(
        "Analysis Model",
        ("Default similarity", "SecBERT model", "LLM model"),
        help="Select the model to use for analysis"
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
                    if model_selection == "Default similarity":
                        # Load similarity model
                        model = load_model()
                        positive_examples_df = load_positive_examples()
                        positive_texts = positive_examples_df['normalized_log'].tolist()
                        normalized_log_lines = [normalize_text(line) for line in log_lines]
                        
                        # Compute similarities
                        max_similarities, most_similar_idx = compute_similarities(
                            model, positive_texts, normalized_log_lines
                        )
                        
                        # Build results
                        results_df = build_results_df(
                            log_lines,
                            normalized_log_lines,
                            max_similarities,
                            most_similar_idx,
                            positive_examples_df
                        )
                        results_df['model'] = "Default similarity"
                        results_df['prediction'] = "Failed login"  # Default model only detects failures
                    
                    elif model_selection == "SecBERT model":
                        # Use SecBERT analyzer
                        results_df = st.session_state.secbert_analyzer.analyze_logs(
                            log_lines,
                            confidence_threshold
                        )
                    else:
                        st.warning("LLM model not yet implemented")
                        return
                    
                    # Display results
                    display_results(results_df, confidence_threshold, top_n)
                    
                    # Download results
                    st.subheader("💾 Download Results")
                    csv_all = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Full Results (CSV)",
                        data=csv_all,
                        file_name="login_analysis_results.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **How it works:**
    - **Default similarity**: Uses semantic similarity to known failed login patterns
    - **SecBERT model**: Uses cybersecurity-specific BERT model to classify login events
    - **LLM model**: Coming soon - will use large language model for advanced analysis
    """)

if __name__ == "__main__":
    main()