import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import json
from datetime import datetime

from model.model_utils.model_utils import (
    load_model, normalize_text, compute_similarities, 
    build_results_df, load_positive_examples, get_similarity_breakdown
)
from model.bert_model import get_secbert_analyzer
from streamlit_custom_utils.corrections_manager import show_corrections_management
from streamlit_custom_utils.log_feedback import save_feedback, add_feedback_ui
from backend.splunk_config_generator import SplunkConfigGenerator


class StreamlitLogAnalyzer:
    """Main class for the Streamlit log analysis application."""
    
    MODEL_OPTIONS = [
        "bert-base-uncased", 
        "all-MiniLM-L6-v2 (Fast)", 
        "all-mpnet-base-v2 (Balanced)", 
        "all-MiniLM-L12-v2 (Quality)"
    ]
    
    def __init__(self):
        self._setup_page_config()
        self._setup_session_state()
        self.splunk_generator = SplunkConfigGenerator()
        
    def _setup_page_config(self):
        """Configure the Streamlit page settings."""
        st.set_page_config(
            page_title="Secnect | Making sense of security data with AI",
            page_icon="🔐",
            layout="wide"
        )
        
    def _setup_session_state(self):
        """Initialize session state variables."""
        if 'show_results' not in st.session_state:
            st.session_state.show_results = False
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'splunk_configs' not in st.session_state:
            st.session_state.splunk_configs = {}
            
    def _render_header(self):
        """Render the application header with custom CSS."""
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
        
    def _render_sidebar_config(self):
        """Render sidebar configuration options."""
        st.sidebar.header("Configuration")
        
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold", 
            0.0, 1.0, 0.6, 0.05, 
            help="Log lines with similarity scores above this threshold will be highlighted"
        )
        
        top_n = st.sidebar.number_input(
            "Number of top results to display", 
            5, 100, 20, 5
        )
        
        model_selection = st.sidebar.selectbox(
            "Model selection", 
            self.MODEL_OPTIONS
        )
        
        self._render_sidebar_changelog()
        
        return confidence_threshold, top_n, model_selection
    
    def _render_sidebar_changelog(self):
        """Render the changelog section in sidebar."""
        st.sidebar.subheader("Change log*", divider=True)
        st.sidebar.markdown("*Currently available model is **:blue[BERT]**")
        st.sidebar.markdown("*We've added similarity details, and an NER model is coming soon.")
        st.sidebar.markdown("Our model is still under development, and we apologize for any inconvenience.")
        
    def _render_introduction(self):
        """Render the main introduction section."""
        st.title("🔐 Login Event Detector")
        st.markdown('''
        This app uses semantic similarity to identify login events in log files.
        Upload your log file and we'll rank each line by its similarity to known failed login patterns.
        
        For testing purposes, sample log files can be found in the [LogHub repository](https://github.com/logpai/loghub) on GitHub, which contains a collection of system logs from various technologies.
        Not every system listed on LogHub repository contains login events. 
        We recommend to firstly try them out on :blue-background[Linux] or :blue-background[SSH] logs! 
        ''')
        
    def _load_log_file(self, uploaded_file):
        """Load and process the uploaded log file."""
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            log_lines = (df.iloc[:, -1].tolist() 
                        if df.shape[1] > 1 
                        else df.iloc[:, 0].tolist())
        else:
            text = uploaded_file.read().decode('utf-8', errors='ignore')
            log_lines = [line for line in text.splitlines() if line.strip()]
            
        return log_lines
    
    def _display_log_preview(self, log_lines):
        """Display a preview of the first few log lines."""
        with st.expander("View first 5 log lines"):
            for i, line in enumerate(log_lines[:5]):
                st.text(f"{i+1}: {line}")
                
    def _analyze_with_secbert(self, log_lines, confidence_threshold):
        """Analyze logs using SecBERT model."""
        analyzer = get_secbert_analyzer()
        df_results = analyzer.analyze_logs(log_lines, confidence_threshold)
        
        if 'model' not in df_results.columns:
            df_results['model'] = "SecBERT"
            
        return df_results
    
    def _analyze_with_similarity(self, log_lines, model_selection, positive_examples_df, progress):
        """Analyze logs using similarity-based approach."""
        progress.progress(10)
        model_name = model_selection.split(" (")[0]
        sim_model = load_model(model_name)
        
        normalized = [normalize_text(line) for line in log_lines]
        pos_texts = positive_examples_df['normalized_log'].tolist()
        
        progress.progress(45)
        sims, idxs = compute_similarities(sim_model, pos_texts, normalized)
        
        progress.progress(80)
        df_results = build_results_df(log_lines, normalized, sims, idxs, positive_examples_df)
        df_results['model'] = "BERT Similarity"
        
        return df_results, sim_model
    
    def _perform_analysis(self, log_lines, model_selection, confidence_threshold, positive_examples_df):
        """Perform the main log analysis."""
        with st.spinner("Analyzing..."):
            progress = st.progress(0)
            
            if model_selection == "BERT":
                df_results = self._analyze_with_secbert(log_lines, confidence_threshold)
                sim_model = None
            else:
                df_results, sim_model = self._analyze_with_similarity(
                    log_lines, model_selection, positive_examples_df, progress
                )
            
            progress.progress(100)
            
            # Store results in session state
            st.session_state.analysis_results = df_results
            st.session_state.show_results = True
            st.session_state.sim_model = sim_model
            
    def _display_metrics(self, df_results, confidence_threshold):
        """Display analysis metrics."""
        st.header("📊 Analysis Results")
        cols = st.columns(4)
        
        above_threshold = len(df_results[df_results['max_similarity_score'] >= confidence_threshold])
        mean_score = df_results['max_similarity_score'].mean()
        max_score = df_results['max_similarity_score'].max()
        
        cols[0].metric("Total lines", len(df_results))
        cols[1].metric("Above threshold", above_threshold)
        cols[2].metric("Mean score", f"{mean_score:.3f}")
        cols[3].metric("Max score", f"{max_score:.3f}")
        
    def _display_secbert_details(self, row):
        """Display SecBERT-specific analysis details."""
        st.markdown("**Model:** SecBERT")
        if 'explanation' in row:
            st.markdown("**Explanation:**")
            st.write(row['explanation'])
            
            if "Matched" in row['explanation']:
                st.markdown("**Pattern Matched:**")
                st.code(row['explanation'].split(": ")[1])
                
    def _display_similarity_details(self, row, sim_model):
        """Display similarity-based analysis details."""
        st.markdown("**Model:** BERT Similarity")
        if 'most_similar_positive_example' in row and sim_model:
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
            
    def _display_results_detail(self, df_results, top_n, confidence_threshold):
        """Display detailed results for top N entries."""
        st.subheader(f"Top {top_n} results")
        sim_model = st.session_state.get('sim_model')
        
        for idx, row in df_results.head(top_n).iterrows():
            score = row['max_similarity_score']
            highlight_color = 'red' if score >= confidence_threshold else 'orange'
            
            st.markdown(f"**Score:** :{highlight_color}[{score:.4f}]")
            st.markdown("**Original Log Line:**")
            st.text(row['original_log_line'])
            st.markdown("**Normalized Log:**")
            st.text(row.get('normalized_log', normalize_text(row['original_log_line'])))

            with st.expander("Analysis Details"):
                prediction = row.get('prediction', 'N/A')
                st.markdown(f"**Prediction:** {prediction}")
                st.markdown(f"**Confidence:** {score:.4f}")
                
                if row.get('model') == "SecBERT":
                    self._display_secbert_details(row)
                else:
                    self._display_similarity_details(row, sim_model)
            
            # Add feedback UI
            add_feedback_ui(
                row['original_log_line'],
                {"label": prediction, "confidence": score},
                idx
            )
            
            st.divider()
            
    def _display_score_distribution(self, df_results, confidence_threshold):
        """Display score distribution plot."""
        st.subheader("📈 Score Distribution")
        fig, ax = plt.subplots()
        
        ax.hist(df_results['max_similarity_score'], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(confidence_threshold, linestyle='--', label=f'Threshold: {confidence_threshold}')
        
        mean_score = df_results['max_similarity_score'].mean()
        ax.axvline(mean_score, linestyle='--', label=f"Mean: {mean_score:.3f}")
        
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        st.pyplot(fig)
        
    def _display_download_section(self, df_results):
        """Display download section."""
        st.subheader("💾 Download Results")
        csv = df_results.to_csv(index=False)
        st.download_button(
            "Download Full Results", 
            data=csv, 
            file_name="results.csv"
        )
        
    def _display_how_it_works(self):
        """Display the how it works section."""
        st.markdown("---")
        st.markdown('''
        **How it works:**
        1. Upload your log file
        2. The app normalizes text by removing timestamps, IPs, and numbers
        3. Uses Sentence-BERT to compute semantic embeddings
        4. Calculates cosine similarity with known failed login patterns
        5. Ranks log lines by similarity score
        ''')
        
    def run_log_analysis_page(self):
        """Run the main log analysis page."""
        self._render_introduction()
        
        positive_examples_df = load_positive_examples()
        if positive_examples_df is None:
            return

        confidence_threshold, top_n, model_selection = self._render_sidebar_config()

        # File upload section
        st.header("📁 Upload Log File")
        uploaded_file = st.file_uploader(
            "Choose a log file", 
            type=['log', 'txt', 'csv'], 
            help="Upload a log file in .log, .txt, or .csv format"
        )
        st.write(f"You selected: **{model_selection}**")
        
        if not uploaded_file:
            st.session_state.show_results = False
            return

        # Process uploaded file
        log_lines = self._load_log_file(uploaded_file)
        st.success(f"Loaded {len(log_lines)} log lines")
        self._display_log_preview(log_lines)

        # Analysis button
        if st.button("🔍 Analyze Log File", type="primary"):
            self._perform_analysis(log_lines, model_selection, confidence_threshold, positive_examples_df)

        # Display results if available
        if st.session_state.show_results and st.session_state.analysis_results is not None:
            df_results = st.session_state.analysis_results
            
            self._display_metrics(df_results, confidence_threshold)
            self._display_results_detail(df_results, top_n, confidence_threshold)
            self._display_score_distribution(df_results, confidence_threshold)
            self._display_download_section(df_results)

        self._display_how_it_works()
        
    def run_named_entity_recognition_page(self):
        """Placeholder for Named Entity Recognition page."""
        st.title("🏷️ Named Entity Recognition")
        st.info("This feature is coming soon!")
        
    def run_splunk_configuration_page(self):
        """Run the Splunk Configuration Generator page."""
        st.title("🔧 Splunk Configuration Generator")
        st.markdown('''
        Generate Splunk configuration files based on your log analysis results.
        This tool helps you create props.conf, transforms.conf, and savedsearches.conf files
        to integrate your security log analysis into Splunk.
        ''')
        
        # Check if analysis results are available
        if not st.session_state.get('analysis_results') is not None:
            st.warning("⚠️ Please run log analysis first to generate Splunk configurations.")
            if st.button("Go to Log Analysis"):
                st.rerun()
            return
            
        df_results = st.session_state.analysis_results
        
        # Configuration options
        st.header("🔧 Configuration Options")
        col1, col2 = st.columns(2)
        
        with col1:
            sourcetype = st.text_input(
                "Source Type", 
                value="security_logs",
                help="Splunk sourcetype for your logs"
            )
            
            index_name = st.text_input(
                "Index Name", 
                value="security",
                help="Splunk index where logs will be stored"
            )
            
            confidence_threshold = st.slider(
                "Alert Threshold", 
                0.0, 1.0, 0.8, 0.05,
                help="Confidence threshold for generating alerts"
            )
            
        with col2:
            app_name = st.text_input(
                "Splunk App Name", 
                value="secnect_security",
                help="Name of the Splunk app"
            )
            
            field_extractions = st.multiselect(
                "Field Extractions",
                ["timestamp", "source_ip", "user", "action", "severity"],
                default=["timestamp", "source_ip", "user"],
                help="Fields to extract from log lines"
            )
            
            enable_alerts = st.checkbox(
                "Enable Real-time Alerts", 
                value=True,
                help="Generate saved searches for real-time alerting"
            )
        
        # Generate configurations
        if st.button("🔧 Generate Splunk Configurations", type="primary"):
            with st.spinner("Generating configurations..."):
                config_data = {
                    'sourcetype': sourcetype,
                    'index_name': index_name,
                    'app_name': app_name,
                    'confidence_threshold': confidence_threshold,
                    'field_extractions': field_extractions,
                    'enable_alerts': enable_alerts,
                    'analysis_results': df_results
                }
                
                configs = self.splunk_generator.generate_all_configs(config_data)
                st.session_state.splunk_configs = configs
                
        # Display generated configurations
        if st.session_state.splunk_configs:
            self._display_splunk_configs(st.session_state.splunk_configs)
            
    def _display_splunk_configs(self, configs):
        """Display the generated Splunk configurations."""
        st.header("📋 Generated Configurations")
        
        # Tabs for different configuration files
        tab1, tab2, tab3, tab4 = st.tabs(["props.conf", "transforms.conf", "savedsearches.conf", "Summary"])
        
        with tab1:
            st.subheader("props.conf")
            st.markdown("Configuration for log parsing and field extraction:")
            st.code(configs['props_conf'], language="ini")
            st.download_button(
                "Download props.conf",
                data=configs['props_conf'],
                file_name="props.conf",
                mime="text/plain"
            )
            
        with tab2:
            st.subheader("transforms.conf")
            st.markdown("Configuration for field transformations:")
            st.code(configs['transforms_conf'], language="ini")
            st.download_button(
                "Download transforms.conf",
                data=configs['transforms_conf'],
                file_name="transforms.conf",
                mime="text/plain"
            )
            
        with tab3:
            st.subheader("savedsearches.conf")
            st.markdown("Saved searches and alerts:")
            st.code(configs['savedsearches_conf'], language="ini")
            st.download_button(
                "Download savedsearches.conf",
                data=configs['savedsearches_conf'],
                file_name="savedsearches.conf",
                mime="text/plain"
            )
            
        with tab4:
            st.subheader("Configuration Summary")
            summary = configs.get('summary', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Generated Searches", summary.get('search_count', 0))
                st.metric("Field Extractions", summary.get('field_count', 0))
                
            with col2:
                st.metric("Alert Rules", summary.get('alert_count', 0))
                st.metric("High Confidence Events", summary.get('high_confidence_count', 0))
                
            # Installation instructions
            with st.expander("📖 Installation Instructions"):
                st.markdown("""
                ### How to install these configurations in Splunk:
                
                1. **Copy configuration files** to your Splunk app directory:
                   ```
                   $SPLUNK_HOME/etc/apps/{app_name}/local/
                   ```
                
                2. **Restart Splunk** or reload the configuration:
                   ```
                   splunk restart
                   ```
                   or
                   ```
                   splunk reload deploy-server
                   ```
                
                3. **Verify the configuration** in Splunk Web:
                   - Go to Settings > Data Inputs
                   - Check Settings > Fields > Field Extractions
                   - Verify Settings > Searches, Reports, and Alerts
                
                4. **Test with sample data**:
                   - Upload a sample log file
                   - Verify field extractions are working
                   - Check that alerts are triggered for high-confidence events
                """)
                
            # Download all configs as ZIP
            if st.button("📦 Download All Configurations"):
                zip_data = self.splunk_generator.create_config_package(configs)
                st.download_button(
                    "Download Complete Package",
                    data=zip_data,
                    file_name=f"splunk_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
        
    def run(self):
        """Main application entry point."""
        self._render_header()
        
        page = st.sidebar.selectbox(
            "Choose a page", 
            ["Log Analysis", "Corrections Management", "Named Entity Recognition", "Splunk Configuration Generator"]
        )
        
        if page == "Corrections Management":
            show_corrections_management()
        elif page == "Named Entity Recognition":
            self.run_named_entity_recognition_page()
        elif page == "Splunk Configuration Generator":
            self.run_splunk_configuration_page()
        else:
            self.run_log_analysis_page()


def main():
    """Application entry point."""
    app = StreamlitLogAnalyzer()
    app.run()


if __name__ == "__main__":
    main()