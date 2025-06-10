# components/sidebar.py

import streamlit as st
from typing import Dict, Any, List, Optional

from config import app_config, page_config, get_default_thresholds, get_model_options


class SidebarComponent:
    """
    Sidebar component responsible for rendering configuration and navigation.

    This component provides a consistent sidebar interface across all pages
    with configuration controls and navigation elements.
    """

    def __init__(self):
        """Initialize the sidebar component."""
        self.config = app_config
        self.page_config = page_config
        self.defaults = get_default_thresholds()

    def render_navigation(self) -> str:
        """
        Render the main navigation selector.

        Returns:
            Selected page name
        """
        return st.sidebar.selectbox(
            "Select a module",
            [
                self.page_config.LOG_ANALYSIS_PAGE,
                self.page_config.CORRECTIONS_PAGE,
                self.page_config.NER_PAGE,
                self.page_config.SPLUNK_CONFIG_PAGE
            ]
        )

    def render_analysis_config(self) -> Dict[str, Any]:
        """
        Render the analysis configuration controls.

        Returns:
            Dictionary containing all configuration values
        """
        st.sidebar.header("Configuration")

        # Confidence threshold slider
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=self.defaults["confidence_threshold"],
            step=0.05,
            help="Log lines with similarity scores above this threshold will be highlighted"
        )

        # Top N results input
        top_n = st.sidebar.number_input(
            "Number of top results to display",
            min_value=self.defaults["min_top_n"],
            max_value=self.defaults["max_top_n"],
            value=self.defaults["top_n"],
            step=self.defaults["top_n_step"]
        )

        # Model selection
        model_selection = st.sidebar.selectbox(
            "Model selection",
            get_model_options()
        )

        return {
            'confidence_threshold': confidence_threshold,
            'top_n': top_n,
            'model_selection': model_selection
        }

    def render_splunk_config(self) -> Dict[str, Any]:
        """
        Render Splunk configuration controls.

        Returns:
            Dictionary containing Splunk configuration values
        """
        st.sidebar.header("Splunk Configuration")

        sourcetype = st.sidebar.text_input(
            "Sourcetype Name",
            value=self.config.DEFAULT_SOURCETYPE,
            help="Name for your custom sourcetype in Splunk"
        )

        index_name = st.sidebar.text_input(
            "Index Name",
            value=self.config.DEFAULT_INDEX_NAME,
            help="Splunk index where logs will be stored"
        )

        app_name = st.sidebar.text_input(
            "App Name",
            value=self.config.DEFAULT_APP_NAME,
            help="Name for your Splunk app"
        )

        return {
            'sourcetype': sourcetype,
            'index_name': index_name,
            'app_name': app_name
        }

    def render_changelog(self) -> None:
        """Render the changelog section."""
        st.sidebar.subheader("Change log*", divider=True)

        for item in self.config.CHANGELOG_ITEMS:
            st.sidebar.markdown(item)

    def render_info_section(self, title: str, content: List[str]) -> None:
        """
        Render a custom information section.

        Args:
            title: Section title
            content: List of content items to display
        """
        st.sidebar.subheader(title, divider=True)

        for item in content:
            st.sidebar.markdown(item)

    def render_model_info(self, selected_model: str) -> None:
        """
        Render information about the selected model.

        Args:
            selected_model: Currently selected model name
        """
        st.sidebar.write(f"You selected: **{selected_model}**")

        # Could add more detailed model information here
        model_info = self._get_model_info(selected_model)
        if model_info:
            with st.sidebar.expander("Model Details"):
                for key, value in model_info.items():
                    st.write(f"**{key}:** {value}")

    def _get_model_info(self, model_name: str) -> Optional[Dict[str, str]]:
        """
        Get detailed information about a model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model information or None
        """
        model_info_map = {
            "bert-base-uncased": {
                "Type": "Base BERT model",
                "Speed": "Medium",
                "Accuracy": "Good",
                "Use Case": "General purpose"
            },
            "all-MiniLM-L6-v2 (Fast)": {
                "Type": "Sentence Transformer",
                "Speed": "Fast",
                "Accuracy": "Good",
                "Use Case": "Quick analysis"
            },
            "all-mpnet-base-v2 (Balanced)": {
                "Type": "Sentence Transformer",
                "Speed": "Medium",
                "Accuracy": "High",
                "Use Case": "Balanced performance"
            },
            "all-MiniLM-L12-v2 (Quality)": {
                "Type": "Sentence Transformer",
                "Speed": "Slower",
                "Accuracy": "High",
                "Use Case": "Best quality results"
            }
        }

        return model_info_map.get(model_name)

    def render_analysis_status(self, has_results: bool, has_logs: bool) -> None:
        """
        Render the current analysis status.

        Args:
            has_results: Whether analysis results are available
            has_logs: Whether log lines are loaded
        """
        st.sidebar.subheader("Analysis Status")

        if has_logs:
            st.sidebar.success("✅ Log file loaded")
        else:
            st.sidebar.info("📄 No log file loaded")

        if has_results:
            st.sidebar.success("✅ Analysis complete")
        else:
            st.sidebar.info("⏳ No analysis results")

    def render_debug_info(self, debug_data: Dict[str, Any]) -> None:
        """
        Render debug information (only in development mode).

        Args:
            debug_data: Debug information to display
        """
        from config import env_config

        if env_config.is_development():
            with st.sidebar.expander("🔧 Debug Info"):
                for key, value in debug_data.items():
                    st.write(f"**{key}:** {value}")

    def render_help_section(self) -> None:
        """Render a help section with usage instructions."""
        with st.sidebar.expander("❓ Help"):
            st.markdown("""
            **How to use this application:**

            1. **Upload** a log file in the Log Analysis page
            2. **Configure** the analysis parameters
            3. **Run** the analysis to get results
            4. **Download** or **Generate** Splunk configs

            **Supported file formats:**
            - .log files
            - .txt files  
            - .csv files

            **Tips:**
            - Higher confidence thresholds show fewer, more relevant results
            - Try different models for varying speed/accuracy tradeoffs
            - Use the Splunk generator to automate configuration
            """)

    def render_full_sidebar(self, page_type: str = "analysis") -> Dict[str, Any]:
        """
        Render the complete sidebar for a given page type.

        Args:
            page_type: Type of page ("analysis", "splunk", "general")

        Returns:
            Dictionary with all configuration values
        """
        config = {}

        # Always render navigation
        selected_page = self.render_navigation()
        config['selected_page'] = selected_page

        # Render appropriate configuration based on page type
        if page_type == "analysis":
            analysis_config = self.render_analysis_config()
            config.update(analysis_config)
            self.render_changelog()

        elif page_type == "splunk":
            splunk_config = self.render_splunk_config()
            config.update(splunk_config)

        # Always render help section
        self.render_help_section()

        return config


# Convenience instance for easy importing
sidebar_component = SidebarComponent()