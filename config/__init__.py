# config/__init__.py

from .app_config import (
    app_config,
    page_config,
    ui_config,
    env_config,
    AppConfig,
    PageConfig,
    UIConfig,
    EnvironmentConfig,
    get_model_options,
    get_page_list,
    get_file_types,
    get_default_thresholds
)

# Import theme configuration if it exists
try:
    from .theme import apply_complete_theme, apply_matplotlib_dark_theme
except ImportError:
    # Handle case where theme.py doesn't exist yet
    def apply_complete_theme():
        """Placeholder theme function."""
        pass


    def apply_matplotlib_dark_theme():
        """Placeholder matplotlib theme function."""
        return {}

__all__ = [
    'app_config',
    'page_config',
    'ui_config',
    'env_config',
    'AppConfig',
    'PageConfig',
    'UIConfig',
    'EnvironmentConfig',
    'get_model_options',
    'get_page_list',
    'get_file_types',
    'get_default_thresholds',
    'apply_complete_theme',
    'apply_matplotlib_dark_theme'
]