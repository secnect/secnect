"""
Centralized state management for the Streamlit application.

This module provides a clean interface for managing Streamlit session state,
eliminating the scattered session state management throughout the original code.
"""

import streamlit as st
from typing import Any, Dict, Optional, List
import pandas as pd


class AppState:
    """
    Centralized state management class for the Streamlit application.

    This class provides a clean interface for managing all session state
    variables, making it easier to track and debug state changes.
    """

    # Define all session state keys as class constants
    SHOW_RESULTS = 'show_results'
    ANALYSIS_RESULTS = 'analysis_results'
    LOG_LINES = 'log_lines'
    SIM_MODEL = 'sim_model'
    SPLUNK_CONFIG = 'splunk_config'
    SPLUNK_QUERIES = 'splunk_queries'
    SPLUNK_PATTERNS = 'splunk_patterns'
    CURRENT_PAGE = 'current_page'
    USER_FEEDBACK = 'user_feedback'
    UPLOADED_FILE_NAME = 'uploaded_file_name'
    LAST_ANALYSIS_CONFIG = 'last_analysis_config'

    @staticmethod
    def initialize():
        """
        Initialize all session state variables with their default values.

        This should be called once at the start of the application to ensure
        all required session state variables exist.
        """
        defaults = {
            AppState.SHOW_RESULTS: False,
            AppState.ANALYSIS_RESULTS: None,
            AppState.LOG_LINES: None,
            AppState.SIM_MODEL: None,
            AppState.SPLUNK_CONFIG: None,
            AppState.SPLUNK_QUERIES: None,
            AppState.SPLUNK_PATTERNS: None,
            AppState.CURRENT_PAGE: "Security Log Analysis",
            AppState.USER_FEEDBACK: {},
            AppState.UPLOADED_FILE_NAME: None,
            AppState.LAST_ANALYSIS_CONFIG: None
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """
        Get a value from session state.

        Args:
            key: The session state key
            default: Default value if key doesn't exist

        Returns:
            The value from session state or the default value
        """
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any) -> None:
        """
        Set a value in session state.

        Args:
            key: The session state key
            value: The value to set
        """
        st.session_state[key] = value

    @staticmethod
    def update(updates: Dict[str, Any]) -> None:
        """
        Update multiple values in session state.

        Args:
            updates: Dictionary of key-value pairs to update
        """
        for key, value in updates.items():
            st.session_state[key] = value

    @staticmethod
    def clear_key(key: str) -> None:
        """
        Clear a specific key from session state.

        Args:
            key: The session state key to clear
        """
        if key in st.session_state:
            del st.session_state[key]

    @staticmethod
    def clear_analysis_results() -> None:
        """Clear all analysis-related session state."""
        keys_to_clear = [
            AppState.SHOW_RESULTS,
            AppState.ANALYSIS_RESULTS,
            AppState.LOG_LINES,
            AppState.SIM_MODEL,
            AppState.LAST_ANALYSIS_CONFIG
        ]

        for key in keys_to_clear:
            AppState.clear_key(key)

    @staticmethod
    def clear_splunk_config() -> None:
        """Clear all Splunk configuration-related session state."""
        keys_to_clear = [
            AppState.SPLUNK_CONFIG,
            AppState.SPLUNK_QUERIES,
            AppState.SPLUNK_PATTERNS
        ]

        for key in keys_to_clear:
            AppState.clear_key(key)

    @staticmethod
    def has_analysis_results() -> bool:
        """
        Check if analysis results are available.

        Returns:
            True if analysis results exist and show_results is True
        """
        return (AppState.get(AppState.SHOW_RESULTS, False) and
                AppState.get(AppState.ANALYSIS_RESULTS) is not None)

    @staticmethod
    def has_log_lines() -> bool:
        """
        Check if log lines are available.

        Returns:
            True if log lines exist
        """
        log_lines = AppState.get(AppState.LOG_LINES)
        return log_lines is not None and len(log_lines) > 0

    @staticmethod
    def has_splunk_config() -> bool:
        """
        Check if Splunk configuration is available.

        Returns:
            True if Splunk configuration exists
        """
        return AppState.get(AppState.SPLUNK_CONFIG) is not None

    @staticmethod
    def get_analysis_summary() -> Dict[str, Any]:
        """
        Get a summary of the current analysis state.

        Returns:
            Dictionary containing analysis summary information
        """
        if not AppState.has_analysis_results():
            return {"status": "no_results"}

        results = AppState.get(AppState.ANALYSIS_RESULTS)
        log_lines = AppState.get(AppState.LOG_LINES, [])

        summary = {
            "status": "has_results",
            "total_log_lines": len(log_lines),
            "analysis_entries": len(results) if results is not None else 0,
            "uploaded_file": AppState.get(AppState.UPLOADED_FILE_NAME, "Unknown")
        }

        if isinstance(results, pd.DataFrame) and not results.empty:
            summary.update({
                "mean_confidence": results['max_similarity_score'].mean(),
                "max_confidence": results['max_similarity_score'].max(),
                "min_confidence": results['max_similarity_score'].min()
            })

        return summary

    @staticmethod
    def add_user_feedback(line_index: int, feedback: Dict[str, Any]) -> None:
        """
        Add user feedback for a specific log line.

        Args:
            line_index: Index of the log line
            feedback: Feedback dictionary
        """
        current_feedback = AppState.get(AppState.USER_FEEDBACK, {})
        current_feedback[line_index] = feedback
        AppState.set(AppState.USER_FEEDBACK, current_feedback)

    @staticmethod
    def get_user_feedback(line_index: int) -> Optional[Dict[str, Any]]:
        """
        Get user feedback for a specific log line.

        Args:
            line_index: Index of the log line

        Returns:
            Feedback dictionary or None if no feedback exists
        """
        feedback = AppState.get(AppState.USER_FEEDBACK, {})
        return feedback.get(line_index)

    @staticmethod
    def debug_state() -> Dict[str, Any]:
        """
        Get debug information about the current session state.

        Returns:
            Dictionary containing debug information
        """
        debug_info = {}

        # Check which state variables are set
        state_keys = [
            AppState.SHOW_RESULTS,
            AppState.ANALYSIS_RESULTS,
            AppState.LOG_LINES,
            AppState.SIM_MODEL,
            AppState.SPLUNK_CONFIG,
            AppState.CURRENT_PAGE,
            AppState.UPLOADED_FILE_NAME
        ]

        for key in state_keys:
            value = AppState.get(key)
            if value is not None:
                if isinstance(value, (pd.DataFrame, list)):
                    debug_info[key] = f"Type: {type(value).__name__}, Length: {len(value)}"
                elif isinstance(value, dict):
                    debug_info[key] = f"Dict with {len(value)} keys"
                else:
                    debug_info[key] = f"Type: {type(value).__name__}, Value: {str(value)[:50]}..."
            else:
                debug_info[key] = "None"

        return debug_info


# Create a convenience instance for easy importing
state_manager = AppState()