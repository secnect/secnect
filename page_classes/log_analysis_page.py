# pages/log_analysis_page.py

"""
Log Analysis Page for the Streamlit application.

This page handles the main log analysis functionality including
file upload, analysis execution, and results display.
"""

import streamlit as st
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
import pandas as pd

from page_classes.base_page import BasePage
from config import app_config, ui_config, apply_matplotlib_dark_theme
from utils import AppState
from services.file_service import file_service
from components import sidebar_component

# Import model utilities
from model.model_utils.model_utils import (
    load_model, normalize_text, compute_similarities,
    build_results_df, load_positive_examples, get_similarity_breakdown
)
from model.bert_model import get_secbert_analyzer
from streamlit_custom_utils.log_feedback import add_feedback_ui


class LogAnalysisPage(BasePage):
    """
    Page class for log analysis functionality.

    This page handles file upload, analysis configuration,
    execution of analysis, and display of results.
    """

    def __init__(self):
        """Initialize the log analysis page."""
        super().__init__()
        self.file_service = file_service
        self.sidebar = sidebar_component

    def render(self) -> None:
        """Render the complete log analysis page."""
        self._render_introduction()

        # Load positive examples
        positive_examples_df = load_positive_examples()
        if positive_examples_df is None:
            self.display_error("Failed to load positive examples for analysis")
            return

        # Get configuration from sidebar
        config = self._get_analysis_config()

        # Handle file upload and analysis
        self._handle_file_upload_section(config, positive_examples_df)

        # Display results if available
        if AppState.has_analysis_results():
            self._display_analysis_results(config)

        # Show how it works section
        self._display_how_it_works()

    def _render_introduction(self) -> None:
        """Render the page introduction."""
        self.render_section_header("Security Log Analysis", ui_config.ANALYSIS_ICON)
        st.markdown(app_config.INTRODUCTION_TEXT)

    def _get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration from sidebar."""
        config = self.sidebar.render_analysis_config()
        return config

    def _handle_file_upload_section(self, config: Dict[str, Any], positive_examples_df: pd.DataFrame) -> None:
        """Handle the file upload section."""
        self.render_section_header("Upload Log File", ui_config.UPLOAD_ICON, divider=False)

        # File uploader
        uploaded_file = self.handle_file_upload("Choose a log file", key="log_upload_1")

        # Add example data button
        st.markdown("**Or try with example data:**")
        if st.button("🚀 Load Example Security Logs", help="Load 27 example security log entries for demonstration", key="load_examples"):
            self._load_example_logs(config, positive_examples_df)
            return  # Exit early since we've processed example data

        # Process uploaded file if one was selected
        if uploaded_file:
            st.write(f"You selected: **{config['model_selection']}**")
            self._process_uploaded_file(uploaded_file, config, positive_examples_df)
        else:
            AppState.set('show_results', False)

    def _process_uploaded_file(self, uploaded_file, config: Dict[str, Any], positive_examples_df: pd.DataFrame) -> None:
        """Process the uploaded file and handle analysis."""
        # Process file using file service
        result = self.file_service.process_uploaded_file(uploaded_file)

        if not result["success"]:
            self.display_error(result["error"])
            return

        log_lines = result["log_lines"]
        file_info = result["file_info"]

        # Store file info
        AppState.set('uploaded_file_name', file_info['name'])

        # Display file info and preview
        self.display_success(f"Loaded {len(log_lines)} log lines from {file_info['name']}")
        self._display_file_info(file_info, result)
        self._display_log_preview(log_lines)

        # Analysis button
        if st.button(f"{ui_config.SEARCH_ICON} Analyze Log File", type="primary", key=f"log_upload_{uploaded_file.name}"):
            self._perform_analysis(log_lines, config, positive_examples_df)
            
    def _load_example_logs(self, config: Dict[str, Any], positive_examples_df: pd.DataFrame) -> None:
        """Load and analyze example security logs."""
        # Example security log entries
        example_logs = [
            "Jun 30 20:16:26 combo sshd(pam_unix)[19208]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=195.129.24.210  user=root",
            "Jun 30 20:16:30 combo sshd(pam_unix)[19222]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=195.129.24.210  user=root", 
            "Jun 30 20:53:04 combo klogind[19272]: Authentication failed from 163.27.187.39 (163.27.187.39): Permission denied in replay cache code",
            "Jun 30 20:53:04 combo klogind[19272]: Kerberos authentication failed",
            "Jun 30 20:53:04 combo klogind[19287]: Authentication failed from 163.27.187.39 (163.27.187.39): Permission denied in replay cache code",
            "Jun 30 20:17:45 gateway sshd(pam_unix)[19234]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=192.168.1.150  user=admin",
            "Jun 30 20:18:12 webserver httpd[8912]: authentication failed for user 'dbuser' from 172.16.0.42",
            "Jun 30 20:19:33 mailserver postfix/smtpd[5678]: authentication failed: user=mail@company.com, method=PLAIN, rip=203.45.67.89",
            "Jun 30 20:20:15 firewall kernel[0]: TCP connection denied from 10.0.0.45:3389 to 192.168.1.100:22",
            "Jun 30 20:21:08 database mysqld[7412]: Access denied for user 'backup'@'10.1.1.25' (using password: YES)",
            "Jun 30 20:22:41 proxy squid[9876]: authentication failed for user 'guest' from 172.20.1.33",
            "Jun 30 20:23:19 ldapserver slapd[4567]: authentication failure; logname= uid=1001 euid=1001 user=testuser rhost=10.10.10.50",
            "Jun 30 20:24:52 vpnserver openvpn[3344]: authentication failed for user 'contractor' from 88.99.77.66",
            "Jun 30 20:25:37 ftpserver vsftpd[2211]: authentication failed for user 'anonymous' from 45.123.89.12",
            "Jun 30 20:26:14 appserver tomcat[8888]: authentication failure; user=developer rhost=192.168.10.75 method=form",
            "Jun 30 20:27:03 dns bind[1123]: authentication failed from 208.67.222.222 for zone transfer request",
            "Jun 30 20:27:48 radius radiusd[5599]: authentication failed for user 'wifiuser' from nas 172.30.1.1 port 0",
            "Jun 30 20:28:25 jenkins java[7799]: authentication failed for user 'build' from 10.5.5.100",
            "Jun 30 20:29:11 backup bacula[4433]: authentication failed for client 'workstation-01' from 192.168.2.150",
            "Jun 30 20:30:07 monitoring nagios[6677]: authentication failed for user 'monitor' from 172.25.1.200",
            "Jun 30 20:31:15 gateway sshd(pam_unix)[19456]: session opened for user 'sysadmin' by (uid=0) from 192.168.1.25",
            "Jun 30 20:32:22 webserver httpd[9134]: authentication successful for user 'webadmin' from 172.16.0.50",
            "Jun 30 20:33:08 mailserver postfix/smtpd[5890]: authentication successful: user=support@company.com, method=LOGIN, rip=203.45.67.100",
            "Jun 30 20:34:44 database mysqld[7633]: User 'analytics'@'10.1.1.30' authenticated successfully",
            "Jun 30 20:35:19 vpnserver openvpn[3567]: user 'manager' authenticated successfully from 88.99.77.80",
            "Jun 30 20:36:33 ldapserver slapd[4890]: bind successful for user 'jdoe' from 10.10.10.60",
            "Jun 30 20:37:12 appserver tomcat[9012]: login successful; user=qa_engineer rhost=192.168.10.80 method=form"
        ]
        
        # Store example file info
        AppState.set('uploaded_file_name', 'example_security_logs.txt')
        
        # Display success message and file info
        self.display_success(f"Loaded {len(example_logs)} example security log lines")
        
        # Display example file info
        example_file_info = {
            'name': 'example_security_logs.txt',
            'size': sum(len(line.encode('utf-8')) for line in example_logs),
            'type': 'text/plain'
        }
        
        example_result = {
            'log_lines': example_logs,
            'line_count': len(example_logs)
        }
        
        self._display_file_info(example_file_info, example_result)
        self._display_log_preview(example_logs)
        
        st.write(f"You selected: **{config['model_selection']}**")
        
        # Automatically perform analysis instead of showing another button
        st.info("Automatically analyzing example logs...")
        self._perform_analysis(example_logs, config, positive_examples_df)
        
    def _display_file_info(self, file_info: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Display information about the uploaded file."""
        with st.expander("📊 File Information"):
            col1, col2 = self.create_columns([1, 1])

            with col1:
                st.write(f"**Filename:** {file_info['name']}")
                st.write(f"**Size:** {file_info['size']:,} bytes")
                st.write(f"**Type:** {file_info['type']}")

            with col2:
                st.write(f"**Lines:** {result['line_count']:,}")
                stats = self.file_service.get_file_stats(result['log_lines'])
                st.write(f"**Avg Line Length:** {stats['avg_length']:.1f} chars")
                st.write(f"**Total Characters:** {stats['total_characters']:,}")

    def _display_log_preview(self, log_lines: list) -> None:
        """Display a preview of log lines."""
        preview_text = self.file_service.create_preview(log_lines)

        with st.expander("👀 View First 5 Log Lines"):
            st.text(preview_text)

    def _perform_analysis(self, log_lines: list, config: Dict[str, Any], positive_examples_df: pd.DataFrame) -> None:
        """Perform the main log analysis."""
        with self.render_loading_spinner("Analyzing log file..."):
            progress = self.create_progress_bar()

            try:
                if config['model_selection'] == "BERT":
                    df_results = self._analyze_with_secbert(log_lines, config['confidence_threshold'])
                    sim_model = None
                else:
                    df_results, sim_model = self._analyze_with_similarity(
                        log_lines, config, positive_examples_df, progress
                    )

                progress.progress(1.0)

                # Store results
                AppState.update({
                    'analysis_results': df_results,
                    'show_results': True,
                    'sim_model': sim_model,
                    'log_lines': log_lines,
                    'last_analysis_config': config
                })

                self.display_success("Analysis completed successfully!")

            except Exception as e:
                self.display_error(f"Analysis failed: {str(e)}")
                st.exception(e)

    def _analyze_with_secbert(self, log_lines: list, confidence_threshold: float) -> pd.DataFrame:
        """Analyze logs using SecBERT model."""
        analyzer = get_secbert_analyzer()
        df_results = analyzer.analyze_logs(log_lines, confidence_threshold)

        if 'model' not in df_results.columns:
            df_results['model'] = "SecBERT"

        return df_results

    def _analyze_with_similarity(self, log_lines: list, config: Dict[str, Any], positive_examples_df: pd.DataFrame,
                                 progress) -> tuple:
        """Analyze logs using similarity-based approach."""
        progress.progress(0.1)

        model_name = config['model_selection'].split(" (")[0]
        sim_model = load_model(model_name)

        normalized = [normalize_text(line) for line in log_lines]
        pos_texts = positive_examples_df['normalized_log'].tolist()

        progress.progress(0.45)
        sims, idxs = compute_similarities(sim_model, pos_texts, normalized)

        progress.progress(0.8)
        df_results = build_results_df(log_lines, normalized, sims, idxs, positive_examples_df)
        df_results['model'] = "BERT Similarity"

        return df_results, sim_model

    def _display_analysis_results(self, config: Dict[str, Any]) -> None:
        """Display all analysis results."""
        df_results = AppState.get('analysis_results')
        confidence_threshold = config['confidence_threshold']
        top_n = config['top_n']

        # Display metrics
        self._display_metrics(df_results, confidence_threshold)

        # Display detailed results
        self._display_results_detail(df_results, top_n, confidence_threshold)

        # Display score distribution
        self._display_score_distribution(df_results, confidence_threshold)

        # Display download section
        self._display_download_section(df_results)

    def _display_metrics(self, df_results: pd.DataFrame, confidence_threshold: float) -> None:
        """Display analysis metrics."""
        self.render_section_header("Analysis Results", ui_config.ANALYSIS_ICON, divider=False)

        above_threshold = len(df_results[df_results['max_similarity_score'] >= confidence_threshold])
        mean_score = df_results['max_similarity_score'].mean()
        max_score = df_results['max_similarity_score'].max()

        metrics = {
            "Total lines": len(df_results),
            "Above threshold": above_threshold,
            "Mean score": f"{mean_score:.3f}",
            "Max score": f"{max_score:.3f}"
        }

        self.create_metric_display(metrics, columns=4)

    def _display_results_detail(self, df_results: pd.DataFrame, top_n: int, confidence_threshold: float) -> None:
        """Display detailed results for top N entries."""
        self.render_subheader(f"Top {top_n} Results", divider=True)

        sim_model = AppState.get('sim_model')

        for idx, row in df_results.head(top_n).iterrows():
            score = row['max_similarity_score']

            # Score display with color coding
            color = 'red' if score >= confidence_threshold else 'orange'
            st.markdown(f'**Score:** <span style="color: {color}">{score:.4f}</span>', unsafe_allow_html=True)

            # Log line display
            st.markdown("**Original Log Line:**")
            st.text(row['original_log_line'])
            st.markdown("**Normalized Log:**")
            st.text(row.get('normalized_log', normalize_text(row['original_log_line'])))

            # Analysis details
            with st.expander("🔍 Analysis Details"):
                prediction = row.get('prediction', 'N/A')
                st.markdown(f"**Prediction:** {prediction}")
                st.markdown(f"**Confidence:** {score:.4f}")

                if row.get('model') == "SecBERT":
                    self._display_secbert_details(row)
                else:
                    self._display_similarity_details(row, sim_model)

            # Feedback UI
            add_feedback_ui(
                row['original_log_line'],
                {"label": prediction, "confidence": score},
                idx
            )

            st.divider()

    def _display_secbert_details(self, row: pd.Series) -> None:
        """Display SecBERT-specific analysis details."""
        st.markdown("**Model:** SecBERT")
        if 'explanation' in row and row['explanation']:
            st.markdown("**Explanation:**")
            st.write(row['explanation'])

            if "Matched" in str(row['explanation']):
                st.markdown("**Pattern Matched:**")
                try:
                    pattern = str(row['explanation']).split(": ")[1]
                    st.code(pattern)
                except (IndexError, AttributeError):
                    st.write(row['explanation'])

    def _display_similarity_details(self, row: pd.Series, sim_model) -> None:
        """Display similarity-based analysis details."""
        st.markdown("**Model:** BERT Similarity")

        if 'most_similar_positive_example' in row and sim_model and row['most_similar_positive_example']:
            try:
                breakdown = get_similarity_breakdown(
                    sim_model,
                    row['original_log_line'],
                    row['most_similar_positive_example']
                )

                st.markdown("**Common Tokens:**")
                st.write(breakdown.get('common_tokens', 'N/A'))
                st.markdown("**Unique to Log:**")
                st.write(breakdown.get('unique_to_log', 'N/A'))
                st.markdown("**Unique to Example:**")
                st.write(breakdown.get('unique_to_example', 'N/A'))

            except Exception as e:
                st.warning(f"Could not generate similarity breakdown: {str(e)}")

    def _display_score_distribution(self, df_results: pd.DataFrame, confidence_threshold: float) -> None:
        """Display score distribution plot."""
        self.render_subheader("Score Distribution", ui_config.CHART_ICON, divider=True)

        # Get theme configuration
        theme_config = apply_matplotlib_dark_theme()

        fig, ax = plt.subplots(facecolor=theme_config['facecolor'])
        ax.set_facecolor(theme_config['facecolor'])

        # Create histogram
        ax.hist(
            df_results['max_similarity_score'],
            bins=app_config.DEFAULT_HISTOGRAM_BINS,
            edgecolor=theme_config['edgecolor'],
            alpha=0.7,
            color=theme_config['color']
        )

        # Add threshold line
        ax.axvline(
            confidence_threshold,
            linestyle='--',
            color=theme_config['threshold_color'],
            label=f'Threshold: {confidence_threshold}'
        )

        # Add mean line
        mean_score = df_results['max_similarity_score'].mean()
        ax.axvline(
            mean_score,
            linestyle='--',
            color=theme_config['mean_color'],
            label=f"Mean: {mean_score:.3f}"
        )

        # Style the plot
        ax.set_xlabel('Score', color=theme_config['text_color'])
        ax.set_ylabel('Frequency', color=theme_config['text_color'])
        ax.tick_params(colors=theme_config['text_color'])
        ax.legend()

        st.pyplot(fig)

    def _display_download_section(self, df_results: pd.DataFrame) -> None:
        """Display download section."""
        self.render_subheader("Download Results", ui_config.DOWNLOAD_ICON, divider=True)

        # Main results download
        csv = df_results.to_csv(index=False)
        self.create_download_button(
            data=csv,
            filename="analysis_results.csv",
            label="Download Full Results",
            mime="text/csv"
        )

        # Additional download options
        col1, col2 = self.create_columns([1, 1])

        with col1:
            # High confidence results only
            high_conf_results = df_results[df_results['max_similarity_score'] >= 0.8]
            if not high_conf_results.empty:
                high_conf_csv = high_conf_results.to_csv(index=False)
                self.create_download_button(
                    data=high_conf_csv,
                    filename="high_confidence_results.csv",
                    label="Download High Confidence Results",
                    mime="text/csv"
                )

        with col2:
            # Summary statistics
            summary_stats = self._generate_summary_stats(df_results)
            self.create_download_button(
                data=summary_stats,
                filename="analysis_summary.txt",
                label="Download Summary Report",
                mime="text/plain"
            )

    def _generate_summary_stats(self, df_results: pd.DataFrame) -> str:
        """Generate summary statistics as text."""
        stats = [
            "Log Analysis Summary Report",
            "=" * 40,
            f"Generated: {AppState.get('uploaded_file_name', 'Unknown file')}",
            f"Total log lines analyzed: {len(df_results)}",
            f"Mean confidence score: {df_results['max_similarity_score'].mean():.4f}",
            f"Max confidence score: {df_results['max_similarity_score'].max():.4f}",
            f"Min confidence score: {df_results['max_similarity_score'].min():.4f}",
            f"Standard deviation: {df_results['max_similarity_score'].std():.4f}",
            "",
            "Confidence Distribution:",
            f"  High (>= 0.8): {len(df_results[df_results['max_similarity_score'] >= 0.8])}",
            f"  Medium (0.6-0.8): {len(df_results[(df_results['max_similarity_score'] >= 0.6) & (df_results['max_similarity_score'] < 0.8)])}",
            f"  Low (< 0.6): {len(df_results[df_results['max_similarity_score'] < 0.6])}",
        ]

        return "\n".join(stats)

    def _display_how_it_works(self) -> None:
        """Display the how it works section."""
        st.markdown("---")
        st.markdown(app_config.HOW_IT_WORKS_TEXT)

    def get_page_title(self) -> str:
        """Get the title for this page."""
        return "Log Analysis"


# Create page instance for easy importing
log_analysis_page = LogAnalysisPage()