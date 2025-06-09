# pages/splunk_config_page.py

"""
Splunk Configuration Page for the Streamlit application.

This page handles the generation of Splunk configuration files
based on log analysis results.
"""

import streamlit as st
from typing import Dict, Any, Optional
from datetime import datetime

from page_classes.base_page import BasePage
from config import app_config, ui_config
from utils import AppState
from services import file_service
from backend.splunk_config_generator import SplunkConfigGenerator


class SplunkConfigPage(BasePage):
    """
    Page class for Splunk configuration generation.

    This page handles the creation of Splunk configuration files
    including props.conf, transforms.conf, and savedsearches.conf.
    """

    def __init__(self):
        """Initialize the Splunk configuration page."""
        super().__init__()
        self.splunk_generator = SplunkConfigGenerator()
        self.file_service = file_service

    def render(self) -> None:
        """Render the complete Splunk configuration page."""
        self._render_introduction()

        # Get configuration settings
        config = self._render_configuration_section()

        # Handle configuration generation
        self._handle_configuration_generation(config)

        # Display generated configurations if available
        if AppState.has_splunk_config():
            self._display_generated_configurations()

    def _render_introduction(self) -> None:
        """Render the page introduction."""
        self.render_section_header("Splunk Configuration Generator")

        st.markdown("""
        Generate Splunk configuration files based on your log analysis results. 
        This tool creates props.conf, transforms.conf, savedsearches.conf, and other configuration files 
        to automatically parse and monitor your security logs in Splunk.
        """)

    def _render_configuration_section(self) -> Dict[str, Any]:
        """Render the configuration input section."""
        self.render_section_header("Configuration Settings", divider=False)

        col1, col2 = self.create_columns([1, 1])

        with col1:
            sourcetype = st.text_input(
                "Sourcetype Name",
                value=app_config.DEFAULT_SOURCETYPE,
                help="Name for your custom sourcetype in Splunk"
            )

            index_name = st.text_input(
                "Index Name",
                value=app_config.DEFAULT_INDEX_NAME,
                help="Splunk index where logs will be stored"
            )

            confidence_threshold = st.slider(
                "Alert Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="Confidence threshold for generating alerts"
            )

        with col2:
            app_name = st.text_input(
                "App Name",
                value=app_config.DEFAULT_APP_NAME,
                help="Name for your Splunk app"
            )

            field_extractions = st.multiselect(
                "Field Extractions",
                ["timestamp", "source_ip", "user", "action", "severity", "event_type"],
                default=["timestamp", "source_ip", "user"],
                help="Fields to extract from log lines"
            )

            enable_alerts = st.checkbox(
                "Enable Real-time Alerts",
                value=True,
                help="Generate saved searches for real-time alerting"
            )

        return {
            'sourcetype': sourcetype,
            'index_name': index_name,
            'app_name': app_name,
            'confidence_threshold': confidence_threshold,
            'field_extractions': field_extractions,
            'enable_alerts': enable_alerts
        }

    def _handle_configuration_generation(self, config: Dict[str, Any]) -> None:
        """Handle the configuration generation process."""
        # Check if we have analysis results
        if AppState.has_analysis_results() and AppState.has_log_lines():
            self._handle_generation_from_analysis(config)
        else:
            self._handle_generation_from_upload(config)

    def _handle_generation_from_analysis(self, config: Dict[str, Any]) -> None:
        """Handle generation using existing analysis results."""
        self.display_success("Using results from previous log analysis")

        # Show analysis summary
        self._display_analysis_summary()

        # Generation button
        if st.button("🔧 Generate Splunk Configurations", type="primary"):
            self._generate_configurations_from_analysis(config)

    def _handle_generation_from_upload(self, config: Dict[str, Any]) -> None:
        """Handle generation from manual file upload."""
        self.display_info("To generate configurations, please first analyze a log file in the 'Log Analysis' page.")

        # Alternative: manual upload
        self.render_subheader("Or upload a log file for configuration generation")

        uploaded_file = self.handle_file_upload(
            "Choose a log file for configuration generation",
            key="splunk_upload"
        )

        if uploaded_file:
            self._handle_manual_upload_generation(uploaded_file, config)

    def _display_analysis_summary(self) -> None:
        """Display summary of existing analysis results."""
        with st.expander("📊 Analysis Results Summary"):
            summary = AppState.get_analysis_summary()

            col1, col2 = self.create_columns([1, 1])

            with col1:
                st.write(f"**Total Log Lines:** {summary.get('total_log_lines', 0):,}")
                st.write(f"**Analysis Results:** {summary.get('analysis_entries', 0):,} entries")
                st.write(f"**Source File:** {summary.get('uploaded_file', 'Unknown')}")

            with col2:
                if 'mean_confidence' in summary:
                    st.write(f"**Mean Confidence:** {summary['mean_confidence']:.3f}")
                    st.write(f"**Max Confidence:** {summary.get('max_confidence', 0):.3f}")
                    st.write(f"**Min Confidence:** {summary.get('min_confidence', 0):.3f}")

    def _handle_manual_upload_generation(self, uploaded_file, config: Dict[str, Any]) -> None:
        """Handle configuration generation from manual upload."""
        result = self.file_service.process_uploaded_file(uploaded_file)

        if not result["success"]:
            self.display_error(result["error"])
            return

        log_lines = result["log_lines"]
        self.display_success(f"Loaded {len(log_lines)} log lines for configuration generation")

        if st.button("🔧 Generate Configurations from Upload", type="primary"):
            self._generate_configurations_from_upload(log_lines, config)

    def _generate_configurations_from_analysis(self, config: Dict[str, Any]) -> None:
        """Generate configurations using existing analysis results."""
        with self.render_loading_spinner("Generating Splunk configurations..."):
            try:
                df_results = AppState.get('analysis_results')
                log_lines = AppState.get('log_lines')

                # Generate configurations
                splunk_config = self.splunk_generator.generate_all_configs(
                    sourcetype=config['sourcetype'],
                    index_name=config['index_name'],
                    log_lines=log_lines,
                    analysis_results=df_results
                )

                # Generate patterns and queries
                patterns = self.splunk_generator.analyze_log_patterns(log_lines, df_results)
                queries = self.splunk_generator.generate_spl_queries(config['sourcetype'], patterns)

                # Store results
                AppState.update({
                    'splunk_config': splunk_config,
                    'splunk_queries': queries,
                    'splunk_patterns': patterns,
                    'splunk_config_settings': config
                })

                self.display_success("Splunk configurations generated successfully!")

            except Exception as e:
                self.display_error(f"Configuration generation failed: {str(e)}")
                st.exception(e)

    def _generate_configurations_from_upload(self, log_lines: list, config: Dict[str, Any]) -> None:
        """Generate configurations from uploaded file without analysis."""
        with self.render_loading_spinner("Generating Splunk configurations..."):
            try:
                # Generate configurations without analysis results
                splunk_config = self.splunk_generator.generate_all_configs(
                    sourcetype=config['sourcetype'],
                    index_name=config['index_name'],
                    log_lines=log_lines
                )

                # Generate patterns and queries
                patterns = self.splunk_generator.analyze_log_patterns(log_lines)
                queries = self.splunk_generator.generate_spl_queries(config['sourcetype'], patterns)

                # Store results
                AppState.update({
                    'splunk_config': splunk_config,
                    'splunk_queries': queries,
                    'splunk_patterns': patterns,
                    'splunk_config_settings': config
                })

                self.display_success("Splunk configurations generated successfully!")

            except Exception as e:
                self.display_error(f"Configuration generation failed: {str(e)}")
                st.exception(e)

    def _display_generated_configurations(self) -> None:
        """Display the generated Splunk configurations."""
        config = AppState.get('splunk_config')
        queries = AppState.get('splunk_queries', {})
        patterns = AppState.get('splunk_patterns', [])

        self.render_section_header("Generated Configurations", "📄")

        # Pattern analysis summary
        if patterns:
            self._display_patterns_summary(patterns)

        # Configuration files in tabs
        self._display_configuration_tabs(config)

        # SPL queries section
        if queries:
            self._display_spl_queries(queries)

        # Download section
        self._display_download_section(config, queries)

    def _display_patterns_summary(self, patterns: list) -> None:
        """Display detected patterns summary."""
        with st.expander("🔍 Detected Patterns Summary"):
            for i, pattern in enumerate(patterns[:5], 1):  # Show top 5 patterns
                st.markdown(f"**Pattern {i}: {pattern.name}** (Confidence: {pattern.confidence:.2f})")
                st.code(pattern.pattern)
                if hasattr(pattern, 'fields') and pattern.fields:
                    st.write(f"Fields: {', '.join(pattern.fields)}")
                st.divider()

    def _display_configuration_tabs(self, config) -> None:
        """Display configuration files in tabs."""
        tab_names = ["props.conf", "transforms.conf", "savedsearches.conf", "macros.conf", "indexes.conf"]
        tabs = self.create_tabs(tab_names)

        config_files = [
            ("props.conf", "Field extractions and sourcetype configuration", config.props_conf),
            ("transforms.conf", "Field transformations and lookups", config.transforms_conf),
            ("savedsearches.conf", "Security alerts and reports", config.savedsearches_conf),
            ("macros.conf", "Reusable search macros", config.macros_conf),
            ("indexes.conf", "Index configuration", config.indexes_conf)
        ]

        for tab, (filename, description, content) in zip(tabs, config_files):
            with tab:
                st.subheader(filename)
                st.markdown(description)

                # Display with custom styling
                st.markdown(f'<div class="{ui_config.CONFIG_PREVIEW_CLASS}">', unsafe_allow_html=True)
                st.code(content, language="ini")
                st.markdown('</div>', unsafe_allow_html=True)

                # Individual download button
                self.create_download_button(
                    data=content,
                    filename=filename,
                    label=f"📄 Download {filename}",
                    mime="text/plain"
                )

    def _display_spl_queries(self, queries: Dict[str, str]) -> None:
        """Display SPL queries section."""
        self.render_section_header("Sample SPL Queries", "🔍")

        if not queries:
            self.display_info("No sample queries generated")
            return

        # Query selector
        query_names = list(queries.keys())
        selected_query = st.selectbox("Select a sample query:", query_names)

        if selected_query and selected_query in queries:
            st.subheader(selected_query)
            st.code(queries[selected_query], language="sql")

            # Copy button simulation (since actual clipboard access is limited)
            if st.button(f"📋 Copy {selected_query} Query"):
                st.success("Query text displayed above - copy from the code block")

    def _display_download_section(self, config, queries: Dict[str, str]) -> None:
        """Display download section for all configurations."""
        self.render_section_header("Download Configuration Package", ui_config.PACKAGE_ICON)

        col1, col2 = self.create_columns([1, 1])

        with col1:
            # Complete package download
            if st.button("📥 Download Complete Package", type="primary"):
                self._create_complete_package_download(config, queries)

        with col2:
            # Individual file downloads
            st.markdown("**Individual File Downloads:**")

            files = {
                "props.conf": config.props_conf,
                "transforms.conf": config.transforms_conf,
                "savedsearches.conf": config.savedsearches_conf,
                "macros.conf": config.macros_conf,
                "indexes.conf": config.indexes_conf
            }

            for filename, content in files.items():
                self.create_download_button(
                    data=content,
                    filename=filename,
                    label=f"📄 {filename}",
                    mime="text/plain",
                    key=f"download_{filename.replace('.', '_')}"
                )

        # Installation instructions
        self._display_installation_instructions()

    def _create_complete_package_download(self, config, queries: Dict[str, str]) -> None:
        """Create and offer complete package download."""
        try:
            settings = AppState.get('splunk_config_settings', {})
            sourcetype = settings.get('sourcetype', app_config.DEFAULT_SOURCETYPE)

            # Create the configuration package
            zip_content = self.splunk_generator.create_config_package(
                config,
                sourcetype,
                queries
            )

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"splunk_config_{sourcetype}_{timestamp}.zip"

            st.download_button(
                label="📥 Download ZIP Package",
                data=zip_content,
                file_name=filename,
                mime="application/zip"
            )

        except Exception as e:
            self.display_error(f"Error creating package: {str(e)}")

    def _display_installation_instructions(self) -> None:
        """Display installation instructions."""
        with st.expander(f"{ui_config.INSTRUCTIONS_ICON} Installation Instructions"):
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

            ### Troubleshooting:

            If configurations don't work:
            1. Check Splunk logs for syntax errors
            2. Verify file permissions are correct
            3. Ensure sourcetype matches your data inputs
            4. Test field extractions with sample data
            """)

    def get_page_title(self) -> str:
        """Get the title for this page."""
        return "Splunk Configuration Generator"


# Create page instance for easy importing
splunk_config_page = SplunkConfigPage()