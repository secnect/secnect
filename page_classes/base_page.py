# pages/base_page.py

from abc import ABC, abstractmethod
import streamlit as st
from typing import Any, Dict, Optional

from config import app_config, ui_config
from utils import AppState


class BasePage(ABC):
    """
    Abstract base class for all application pages.

    This class provides common functionality that all pages need,
    such as configuration access, state management, and standard UI methods.
    """

    def __init__(self):
        """Initialize the base page."""
        self.config = app_config
        self.ui_config = ui_config
        self.state = AppState

    @abstractmethod
    def render(self) -> None:
        """
        Render the page content.

        This method must be implemented by all subclasses to define
        the specific content and behavior of each page.
        """
        pass

    def display_error(self, message: str, icon: str = "🚨") -> None:
        """
        Display an error message with consistent formatting.

        Args:
            message: The error message to display
            icon: Optional icon to display with the message
        """
        st.error(f"{icon} {message}")

    def display_success(self, message: str, icon: str = "✅") -> None:
        """
        Display a success message with consistent formatting.

        Args:
            message: The success message to display
            icon: Optional icon to display with the message
        """
        st.success(f"{icon} {message}")

    def display_info(self, message: str, icon: str = "ℹ️") -> None:
        """
        Display an info message with consistent formatting.

        Args:
            message: The info message to display
            icon: Optional icon to display with the message
        """
        st.info(f"{icon} {message}")

    def display_warning(self, message: str, icon: str = "⚠️") -> None:
        """
        Display a warning message with consistent formatting.

        Args:
            message: The warning message to display
            icon: Optional icon to display with the message
        """
        st.warning(f"{icon} {message}")

    def create_columns(self, specs: list, gap: str = "medium") -> tuple:
        """
        Create columns with consistent spacing.

        Args:
            specs: List of column specifications (integers or fractions)
            gap: Gap size between columns

        Returns:
            Tuple of column objects
        """
        return st.columns(specs, gap=gap)

    def create_tabs(self, tab_names: list) -> tuple:
        """
        Create tabs with consistent styling.

        Args:
            tab_names: List of tab names

        Returns:
            Tuple of tab objects
        """
        return st.tabs(tab_names)

    def render_section_header(self, title: str, icon: str = "", divider: bool = True) -> None:
        """
        Render a consistent section header.

        Args:
            title: The section title
            icon: Optional icon to display
            divider: Whether to add a divider after the header
        """
        header_text = f"{icon} {title}" if icon else title
        st.header(header_text)
        if divider:
            st.divider()

    def render_subheader(self, title: str, icon: str = "", divider: bool = False) -> None:
        """
        Render a consistent subheader.

        Args:
            title: The subheader title
            icon: Optional icon to display
            divider: Whether to add a divider after the subheader
        """
        header_text = f"{icon} {title}" if icon else title
        st.subheader(header_text)
        if divider:
            st.divider()

    def create_metric_display(self, metrics: Dict[str, Any], columns: int = 4) -> None:
        """
        Create a consistent metric display.

        Args:
            metrics: Dictionary of metric name to value pairs
            columns: Number of columns to display metrics in
        """
        cols = st.columns(columns)

        for i, (name, value) in enumerate(metrics.items()):
            col_index = i % columns

            if isinstance(value, dict):
                # Handle complex metric with delta
                cols[col_index].metric(
                    name,
                    value.get('value', 'N/A'),
                    value.get('delta'),
                    value.get('delta_color', 'normal')
                )
            else:
                # Simple metric
                cols[col_index].metric(name, value)

    def create_download_button(
            self,
            data: Any,
            filename: str,
            label: str = None,
            mime: str = "text/plain",
            icon: str = "📥"
    ) -> bool:
        """
        Create a consistent download button.

        Args:
            data: The data to download
            filename: The filename for the download
            label: The button label (defaults to filename)
            mime: The MIME type of the file
            icon: Icon to display on the button

        Returns:
            True if the button was clicked
        """
        if label is None:
            label = f"{icon} Download {filename}"

        return st.download_button(
            label=label,
            data=data,
            file_name=filename,
            mime=mime
        )

    def create_expandable_section(self, title: str, content_func, expanded: bool = False) -> None:
        """
        Create an expandable section with consistent styling.

        Args:
            title: The section title
            content_func: Function to call to render the content
            expanded: Whether the section should be expanded by default
        """
        with st.expander(title, expanded=expanded):
            content_func()

    def check_prerequisites(self, required_state_keys: list) -> bool:
        """
        Check if required state keys exist before rendering content.

        Args:
            required_state_keys: List of session state keys that must exist

        Returns:
            True if all prerequisites are met
        """
        missing_keys = []

        for key in required_state_keys:
            if self.state.get(key) is None:
                missing_keys.append(key)

        if missing_keys:
            self.display_warning(
                f"Missing required data: {', '.join(missing_keys)}. "
                "Please complete the previous steps first."
            )
            return False

        return True

    def render_loading_spinner(self, message: str = "Processing...") -> Any:
        """
        Create a consistent loading spinner.

        Args:
            message: Message to display while loading

        Returns:
            Streamlit spinner context manager
        """
        return st.spinner(message)

    def create_progress_bar(self, value: float = 0.0) -> Any:
        """
        Create a progress bar.

        Args:
            value: Initial progress value (0.0 to 1.0)

        Returns:
            Streamlit progress bar object
        """
        return st.progress(value)

    def handle_file_upload(
            self,
            label: str,
            accepted_types: list = None,
            help_text: str = None,
            key: str = None
    ) -> Optional[Any]:
        """
        Create a consistent file upload widget.

        Args:
            label: Label for the file uploader
            accepted_types: List of accepted file extensions
            help_text: Help text to display
            key: Unique key for the widget

        Returns:
            Uploaded file object or None
        """
        if accepted_types is None:
            accepted_types = self.config.ALLOWED_FILE_TYPES

        if help_text is None:
            help_text = self.config.get_file_upload_help_text()

        return st.file_uploader(
            label,
            type=accepted_types,
            help=help_text,
            key=key
        )

    def get_page_title(self) -> str:
        """
        Get the title for this page.

        Returns:
            The page title (should be overridden by subclasses)
        """
        return "Base Page"

    def on_page_load(self) -> None:
        """
        Called when the page is loaded.

        Override this method to perform page-specific initialization.
        """
        pass

    def on_page_unload(self) -> None:
        """
        Called when the page is about to be unloaded.

        Override this method to perform page-specific cleanup.
        """
        pass