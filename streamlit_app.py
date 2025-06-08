import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import sys
import os
import warnings
import logging

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*torch.classes.*")

# Optionally suppress Streamlit file watcher warnings
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from model.model_utils.model_utils import (
    load_model, normalize_text, compute_similarities, 
    build_results_df, load_positive_examples, get_similarity_breakdown
)
from model.bert_model import get_secbert_analyzer
from streamlit_custom_utils.corrections_manager import show_corrections_management
from streamlit_custom_utils.log_feedback import add_feedback_ui

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
        if 'log_lines' not in st.session_state:
            st.session_state.log_lines = None
            
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
            .splunk-section {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                border-left: 4px solid #2E8B57;
                margin: 15px 0;
            }
            .config-preview {
                background-color: #f4f4f4;
                padding: 10px;
                border-radius: 5px;
                font-family: monospace;
                font-size: 12px;
                max-height: 300px;
                overflow-y: auto;
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
        st.sidebar.markdown("*New: **:green[Splunk Configuration Generator]**")
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
            st.session_state.log_lines = log_lines
            
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
        
    def run_splunk_config_generator_page(self):
        """Run the Splunk Configuration Generator page."""
        st.title("⚙️ Splunk Configuration Generator")
        st.markdown("""
        Generate Splunk configuration files based on your log analysis results. 
        This tool creates props.conf, transforms.conf, savedsearches.conf, and other configuration files 
        to automatically parse and monitor your security logs in Splunk.
        """)
        
        # Configuration inputs
        st.header("📋 Configuration Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            sourcetype = st.text_input(
                "Sourcetype Name",
                value="custom_security_logs",
                help="Name for your custom sourcetype in Splunk"
            )
            index_name = st.text_input(
                "Index Name",
                value="security",
                help="Splunk index where logs will be stored"
            )
        
        with col2:
            app_name = st.text_input(
                "App Name",
                value="security_monitoring",
                help="Name for your Splunk app"
            )
            
        # Check if we have previous analysis results
        if st.session_state.get('analysis_results') is not None and st.session_state.get('log_lines') is not None:
            st.success("✅ Using results from previous log analysis")
            
            with st.expander("Analysis Results Summary"):
                df_results = st.session_state.analysis_results
                log_lines = st.session_state.log_lines
                
                st.write(f"**Total Log Lines:** {len(log_lines)}")
                st.write(f"**Analysis Results:** {len(df_results)} entries")
                st.write(f"**Mean Confidence:** {df_results['max_similarity_score'].mean():.3f}")
                
            # Generate configurations
            if st.button("🔧 Generate Splunk Configurations", type="primary"):
                with st.spinner("Generating Splunk configurations..."):
                    # Generate all configurations
                    config = self.splunk_generator.generate_all_configs(
                        sourcetype=sourcetype,
                        index_name=index_name,
                        log_lines=log_lines,
                        analysis_results=df_results
                    )
                    
                    # Generate SPL queries
                    patterns = self.splunk_generator.analyze_log_patterns(log_lines, df_results)
                    queries = self.splunk_generator.generate_spl_queries(sourcetype, patterns)
                    
                    st.session_state.splunk_config = config
                    st.session_state.splunk_queries = queries
                    st.session_state.splunk_patterns = patterns
                    
                st.success("✅ Splunk configurations generated successfully!")
                
        else:
            st.info("💡 To generate configurations, please first analyze a log file in the 'Log Analysis' page.")
            
            # Allow manual log upload for configuration generation
            st.subheader("Or upload a log file for configuration generation")
            uploaded_file = st.file_uploader(
                "Choose a log file for configuration generation",
                type=['log', 'txt', 'csv'],
                key="splunk_upload"
            )
            
            if uploaded_file:
                log_lines = self._load_log_file(uploaded_file)
                st.success(f"Loaded {len(log_lines)} log lines for configuration generation")
                
                if st.button("🔧 Generate Configurations from Upload", type="primary"):
                    with st.spinner("Generating Splunk configurations..."):
                        config = self.splunk_generator.generate_all_configs(
                            sourcetype=sourcetype,
                            index_name=index_name,
                            log_lines=log_lines
                        )
                        
                        patterns = self.splunk_generator.analyze_log_patterns(log_lines)
                        queries = self.splunk_generator.generate_spl_queries(sourcetype, patterns)
                        
                        st.session_state.splunk_config = config
                        st.session_state.splunk_queries = queries
                        st.session_state.splunk_patterns = patterns
                        
                    st.success("✅ Splunk configurations generated successfully!")
        
        # Display generated configurations
        if 'splunk_config' in st.session_state:
            self._display_splunk_configurations()
            
    def _display_splunk_configurations(self):
        """Display the generated Splunk configurations."""
        config = st.session_state.splunk_config
        queries = st.session_state.get('splunk_queries', {})
        patterns = st.session_state.get('splunk_patterns', [])
        
        st.header("📄 Generated Configurations")
        
        # Pattern analysis summary
        if patterns:
            with st.expander("🔍 Detected Patterns Summary"):
                for pattern in patterns[:5]:  # Show top 5 patterns
                    st.markdown(f"**{pattern.name}** (Confidence: {pattern.confidence:.2f})")
                    st.code(pattern.pattern)
                    st.write(f"Fields: {', '.join(pattern.fields)}")
                    st.divider()
        
        # Configuration files tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "props.conf", "transforms.conf", "savedsearches.conf", "macros.conf", "indexes.conf"
        ])
        
        with tab1:
            st.subheader("props.conf")
            st.markdown("Field extractions and sourcetype configuration:")
            st.markdown('<div class="config-preview">', unsafe_allow_html=True)
            st.code(config.props_conf, language="ini")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tab2:
            st.subheader("transforms.conf")
            st.markdown("Field transformations and lookups:")
            st.markdown('<div class="config-preview">', unsafe_allow_html=True)
            st.code(config.transforms_conf, language="ini")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tab3:
            st.subheader("savedsearches.conf")
            st.markdown("Security alerts and reports:")
            st.markdown('<div class="config-preview">', unsafe_allow_html=True)
            st.code(config.savedsearches_conf, language="ini")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tab4:
            st.subheader("macros.conf")
            st.markdown("Reusable search macros:")
            st.markdown('<div class="config-preview">', unsafe_allow_html=True)
            st.code(config.macros_conf, language="ini")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tab5:
            st.subheader("indexes.conf")
            st.markdown("Index configuration:")
            st.markdown('<div class="config-preview">', unsafe_allow_html=True)
            st.code(config.indexes_conf, language="ini")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # SPL Queries section
        if queries:
            st.header("🔍 Sample SPL Queries")
            
            query_names = list(queries.keys())
            selected_query = st.selectbox("Select a sample query:", query_names)
            
            if selected_query:
                st.subheader(selected_query)
                st.code(queries[selected_query], language="sql")
                
                if st.button(f"Copy {selected_query} Query"):
                    st.success("Query copied to clipboard! (Note: Clipboard access may be limited)")
        
        # Download section
        st.header("📦 Download Configuration Package")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Complete Package", type="primary"):
                try:
                    # Create the configuration package
                    zip_content = self.splunk_generator.create_config_package(
                        config, 
                        st.session_state.get('sourcetype', 'custom_security_logs'),
                        queries
                    )
                    
                    st.download_button(
                        label="📥 Download ZIP Package",
                        data=zip_content,
                        file_name=f"splunk_config_{st.session_state.get('sourcetype', 'custom')}.zip",
                        mime="application/zip"
                    )
                    
                except Exception as e:
                    st.error(f"Error creating package: {str(e)}")
        
        with col2:
            # Individual file downloads
            st.markdown("**Download Individual Files:**")
            
            files = {
                "props.conf": config.props_conf,
                "transforms.conf": config.transforms_conf,
                "savedsearches.conf": config.savedsearches_conf,
                "macros.conf": config.macros_conf,
                "indexes.conf": config.indexes_conf
            }
            
            for filename, content in files.items():
                st.download_button(
                    label=f"📄 {filename}",
                    data=content,
                    file_name=filename,
                    mime="text/plain",
                    key=f"download_{filename}"
                )
        
        # Installation instructions
        with st.expander("📖 Installation Instructions"):
            st.markdown("""
            ### How to Install in Splunk:
            
            1. **Extract the package** to your Splunk app directory:
               ```
               $SPLUNK_HOME/etc/apps/your_app_name/
               ```
            
            2. **Restart Splunk** to load the new configurations:
               ```
               $SPLUNK_HOME/bin/splunk restart
               ```
            
            3. **Configure data inputs** to use your custom sourcetype
            
            4. **Test the configuration** by searching for your data:
               ```
               sourcetype="your_sourcetype_name"
               ```
            
            ### Configuration Details:
            
            - **props.conf**: Defines how Splunk parses your logs and extracts fields
            - **transforms.conf**: Contains lookup tables and field transformations
            - **savedsearches.conf**: Pre-configured security alerts and reports
            - **macros.conf**: Reusable search components for easier query building
            - **indexes.conf**: Index settings optimized for security data
            
            ### Security Monitoring:
            
            The generated configurations include:
            - ✅ Automatic field extraction for IPs, users, timestamps
            - ✅ Pre-built security alerts for failed logins and suspicious activity
            - ✅ CIM (Common Information Model) compliance for integration
            - ✅ Performance-optimized settings
            """)
        
    def run_named_entity_recognition_page(self):
        """Placeholder for Named Entity Recognition page."""
        st.title("🏷️ Named Entity Recognition")
        st.info("This feature is coming soon!")
        
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
            self.run_splunk_config_generator_page()
        else:
            self.run_log_analysis_page()


def main():
    """Application entry point."""
    app = StreamlitLogAnalyzer()
    app.run()


if __name__ == "__main__":
    main()