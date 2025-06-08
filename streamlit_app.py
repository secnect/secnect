import matplotlib.pyplot as plt

import streamlit as st

import pandas as pd


from model.model_utils.model_utils import load_model, normalize_text, compute_similarities, build_results_df, load_positive_examples, get_similarity_breakdown
from model.bert_model import get_secbert_analyzer
from streamlit_custom_utils.corrections_manager import show_corrections_management
from streamlit_custom_utils.log_feedback import save_feedback, add_feedback_ui

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

page = st.sidebar.selectbox("Choose a page", ["Log Analysis", "Corrections Management", "Named Entity Recognition"])
# Model options with descriptions
MODEL_OPTIONS = [
    "all-MiniLM-L6-v2 (Fast)",
    "all-mpnet-base-v2 (Balanced)",
    "all-MiniLM-L12-v2 (High Quality)"
]

def get_similarity_model(model_name=None):
    return load_model(model_name)

def initialize_session_state():
    """Initialize session state variables."""
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'error' not in st.session_state:
        st.session_state.error = None


def display_error(error_msg):
    """Display an error message in the UI."""
    st.error(error_msg)
    st.session_state.error = error_msg


def analyze_logs(log_lines, model_selection, confidence_threshold):
    """Analyze log lines using the selected model."""
    try:
        progress = st.progress(0)
        
        if model_selection == "BERT":
            analyzer = get_secbert_analyzer()
            df_results = analyzer.analyze_logs(log_lines, confidence_threshold)
            df_results['model'] = "SecBERT"
        else:
            model_name = model_selection.split(" (")[0]
            sim_model = get_similarity_model(model_name)
            normalized = [normalize_text(l) for l in log_lines]
            pos_texts = positive_examples_df['normalized_log'].tolist()
            
            progress.progress(45)
            sims, idxs = compute_similarities(sim_model, pos_texts, normalized)
            progress.progress(80)
            df_results = build_results_df(log_lines, normalized, sims, idxs, positive_examples_df)
            df_results['model'] = "BERT Similarity"
            
        progress.progress(100)
        return df_results
    except Exception as e:
        display_error(f"Error during analysis: {str(e)}")
        return None


def display_analysis_results(df_results, confidence_threshold, top_n):
    """Display the analysis results in the UI."""
    if df_results is None:
        return
        
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
        
        st.markdown(f"**Original Log Line:**")
        st.text(row['original_log_line'])
        st.markdown(f"**Normalized Log:**")
        st.text(row.get('normalized_log', normalize_text(row['original_log_line'])))
        
        with st.expander("Analysis Details"):
            display_analysis_details(row)
        
        add_feedback_ui(
            row['original_log_line'],
            {"label": row.get('prediction', 'N/A'), "confidence": score},
            idx
        )
        st.divider()


def display_analysis_details(row):
    """Display detailed analysis information for a log line."""
    prediction = row.get('prediction', 'N/A')
    score = row['max_similarity_score']
    
    st.markdown(f"**Prediction:** {prediction}")
    st.markdown(f"**Confidence:** {score:.4f}")
    
    if row.get('model') == "SecBERT":
        st.markdown("**Model:** SecBERT")
        if 'explanation' in row:
            st.markdown("**Explanation:**")
            st.write(row['explanation'])
            if "Matched" in row['explanation']:
                st.markdown("**Pattern Matched:**")
                st.code(row['explanation'].split(": ")[1])
    else:
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


def main():
    if page == "Corrections Management":
        show_corrections_management()
        return
        
    initialize_session_state()
    
    st.title("🔐 Login Event Detector")
    st.markdown('''
    This app uses semantic similarity to identify login events in log files.
    Upload your log file and we'll rank each line by its similarity to known failed login patterns.
    
    For testing purposes, sample log files can be found in the [LogHub repository](https://github.com/logpai/loghub) on GitHub, which contains a collection of system logs from various technologies.
    Not every system listed on LogHub repository contains login events. 
    We recommend to firstly try them out on :blue-background[Linux] or :blue-background[SSH] logs! 
    ''')
    
    positive_examples_df = load_positive_examples()
    if positive_examples_df is None:
        display_error("Failed to load positive examples")
        return

    st.sidebar.header("Configuration")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.05, help="Log lines with similarity scores above this threshold will be highlighted")
    top_n = st.sidebar.number_input("Number of top results to display", 5, 100, 20, 5)
    model_selection = st.sidebar.selectbox("Model selection", MODEL_OPTIONS)

    st.sidebar.subheader("Change log*")
    st.sidebar.markdown("*Currently available model is **:blue[BERT]**")
    st.sidebar.markdown("*We've added similarity details, and an NER model is coming soon.")
    st.sidebar.markdown("Our model is still under development, and we apologize for any inconvenience.")

    st.header("📁 Upload Log File")
    uploaded_file = st.file_uploader("Choose a log file", type=['log','txt','csv'], help="Upload a log file in .log, .txt, or .csv format")
    st.write(f"You selected: **{model_selection}**")
    
    if not uploaded_file:
        st.session_state.show_results = False
        return

    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            log_lines = df.iloc[:, -1].tolist() if df.shape[1]>1 else df.iloc[:,0].tolist()
        else:
            text = uploaded_file.read().decode('utf-8', errors='ignore')
            log_lines = [l for l in text.splitlines() if l.strip()]
    except Exception as e:
        display_error(f"Error reading file: {str(e)}")
        return

    st.success(f"Loaded {len(log_lines)} log lines")
    
    with st.expander("View first 5 log lines"):
        for i, l in enumerate(log_lines[:5]): st.text(f"{i+1}: {l}")

    if st.button("🔍 Analyze Log File", type="primary"):
        with st.spinner("Analyzing..."):
            df_results = analyze_logs(log_lines, model_selection, confidence_threshold)
            if df_results is not None:
                st.session_state.analysis_results = df_results
                st.session_state.show_results = True

    if st.session_state.show_results and st.session_state.analysis_results is not None:
        display_analysis_results(st.session_state.analysis_results, confidence_threshold, top_n)

    # Distribution plot
    if st.session_state.show_results and st.session_state.analysis_results is not None:
        st.subheader("📈 Score Distribution")
        fig, ax = plt.subplots()
        ax.hist(st.session_state.analysis_results['max_similarity_score'], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(confidence_threshold, linestyle='--', label=f'Threshold: {confidence_threshold}')
        ax.axvline(st.session_state.analysis_results['max_similarity_score'].mean(), linestyle='--', label=f"Mean: {st.session_state.analysis_results['max_similarity_score'].mean():.3f}")
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        st.pyplot(fig)

        # Download
        st.subheader("💾 Download Results")
        csv = st.session_state.analysis_results.to_csv(index=False)
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