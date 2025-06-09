# config/app_config.py

from dataclasses import dataclass
from typing import List, Dict, Any
import os


@dataclass
class AppConfig:
    """Application configuration class containing all app constants."""

    # Page Configuration
    PAGE_TITLE: str = "Secnect | Making sense of security data with AI"
    PAGE_ICON: str = "🔐"
    LAYOUT: str = "wide"

    # Model Options
    MODEL_OPTIONS: List[str] = None

    # Default Values for UI Components
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.6
    DEFAULT_TOP_N_RESULTS: int = 20
    DEFAULT_MIN_TOP_N: int = 5
    DEFAULT_MAX_TOP_N: int = 100
    DEFAULT_TOP_N_STEP: int = 5

    # File Upload Configuration
    ALLOWED_FILE_TYPES: List[str] = None

    # Asset Paths
    LOGO_PATH: str = "assets/logo.png"

    # Analysis Configuration
    DEFAULT_HISTOGRAM_BINS: int = 50
    DEFAULT_LOG_PREVIEW_LINES: int = 5

    # Splunk Configuration Defaults
    DEFAULT_SOURCETYPE: str = "custom_security_logs"
    DEFAULT_INDEX_NAME: str = "security"
    DEFAULT_APP_NAME: str = "security_monitoring"

    # UI Text Content
    INTRODUCTION_TEXT: str = """
    This app uses semantic similarity to identify login events in log files.
    Upload your log file and we'll rank each line by its similarity to known failed login patterns.

    For testing purposes, sample log files can be found in the [LogHub repository](https://github.com/logpai/loghub) on GitHub, which contains a collection of system logs from various technologies.
    Not every system listed on LogHub repository contains login events. 
    We recommend to firstly try them out on :blue-background[Linux] or :blue-background[SSH] logs! 
    """

    HOW_IT_WORKS_TEXT: str = """
    **How it works:**
    1. Upload your log file
    2. The app normalizes text by removing timestamps, IPs, and numbers
    3. Uses Sentence-BERT to compute semantic embeddings
    4. Calculates cosine similarity with known failed login patterns
    5. Ranks log lines by similarity score
    """

    # Changelog Content
    CHANGELOG_ITEMS: List[str] = None

    def __post_init__(self):
        """Initialize fields that need to be set after instantiation."""
        if self.MODEL_OPTIONS is None:
            self.MODEL_OPTIONS = [
                "bert-base-uncased",
                "all-MiniLM-L6-v2 (Fast)",
                "all-mpnet-base-v2 (Balanced)",
                "all-MiniLM-L12-v2 (Quality)"
            ]

        if self.ALLOWED_FILE_TYPES is None:
            self.ALLOWED_FILE_TYPES = ['log', 'txt', 'csv']

        if self.CHANGELOG_ITEMS is None:
            self.CHANGELOG_ITEMS = [
                "*Currently available model is **:blue[BERT]**",
                "*We've added similarity details, and an NER model is coming soon.",
                "*New: **:green[Splunk Configuration Generator]**",
                "Our model is still under development, and we apologize for any inconvenience."
            ]

    def get_file_upload_help_text(self) -> str:
        """Get help text for file upload component."""
        file_types = ", ".join([f".{ext}" for ext in self.ALLOWED_FILE_TYPES])
        return f"Upload a log file in {file_types} format"


@dataclass
class PageConfig:
    """Configuration for individual pages."""

    # Page identifiers
    LOG_ANALYSIS_PAGE: str = "Security Log Analysis"
    CORRECTIONS_PAGE: str = "Corrections Management"
    NER_PAGE: str = "Named Entity Recognition"
    SPLUNK_CONFIG_PAGE: str = "Splunk Configuration Generator"

    def get_all_pages(self) -> List[str]:
        """Get list of all available pages."""
        return [
            self.LOG_ANALYSIS_PAGE,
            self.CORRECTIONS_PAGE,
            self.NER_PAGE,
            self.SPLUNK_CONFIG_PAGE
        ]


@dataclass
class UIConfig:
    """UI-specific configuration."""

    # Styling classes (referenced in your existing CSS)
    COMPANY_HEADER_CLASS: str = "company-header"
    COMPANY_LOGO_CLASS: str = "company-logo"
    BETA_TAG_CLASS: str = "beta-tag"
    COMPANY_NAME_CLASS: str = "company-name"
    HIGHLIGHT_RED_CLASS: str = "highlight-red"
    CONFIG_PREVIEW_CLASS: str = "config-preview"

    # Icons for different sections
    ANALYSIS_ICON: str = "📊"
    UPLOAD_ICON: str = "📁"
    SEARCH_ICON: str = "🔍"
    DOWNLOAD_ICON: str = "💾"
    CONFIG_ICON: str = "⚙️"
    CHART_ICON: str = "📈"
    PACKAGE_ICON: str = "📦"
    INSTRUCTIONS_ICON: str = "📖"
    NER_ICON: str = "🏷️"


class EnvironmentConfig:
    """Environment-specific configuration."""

    @staticmethod
    def is_development() -> bool:
        """Check if running in development mode."""
        return os.getenv("STREAMLIT_ENV", "production").lower() == "development"

    @staticmethod
    def get_log_level() -> str:
        """Get logging level from environment."""
        return os.getenv("LOG_LEVEL", "INFO").upper()

    @staticmethod
    def get_backend_path() -> str:
        """Get backend path configuration."""
        return os.getenv("BACKEND_PATH", os.path.join(os.path.dirname(__file__), '..', 'backend'))


# Singleton instances for easy access
app_config = AppConfig()
page_config = PageConfig()
ui_config = UIConfig()
env_config = EnvironmentConfig()


# Helper functions for easy access to configuration
def get_model_options() -> List[str]:
    """Get available model options."""
    return app_config.MODEL_OPTIONS


def get_page_list() -> List[str]:
    """Get list of all pages."""
    return page_config.get_all_pages()


def get_file_types() -> List[str]:
    """Get allowed file types for upload."""
    return app_config.ALLOWED_FILE_TYPES


def get_default_thresholds() -> Dict[str, Any]:
    """Get default threshold values for UI components."""
    return {
        "confidence_threshold": app_config.DEFAULT_CONFIDENCE_THRESHOLD,
        "top_n": app_config.DEFAULT_TOP_N_RESULTS,
        "min_top_n": app_config.DEFAULT_MIN_TOP_N,
        "max_top_n": app_config.DEFAULT_MAX_TOP_N,
        "top_n_step": app_config.DEFAULT_TOP_N_STEP
    }