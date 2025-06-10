# components/header.py

import streamlit as st
import base64
import os
from typing import Optional

from config import app_config, ui_config


class HeaderComponent:
    """
    Header component responsible for rendering the application header.

    This component handles logo loading, fallback display, and consistent
    branding across all pages of the application.
    """

    def __init__(self):
        """Initialize the header component."""
        self.config = app_config
        self.ui_config = ui_config
        self.logo_path = self.config.LOGO_PATH

    def render(self) -> None:
        """
        Render the complete header with logo and branding.

        This method attempts to load and display the logo image,
        falling back to a text-based header if the logo is not found.
        """
        logo_base64 = self._load_logo()

        if logo_base64:
            self._render_logo_header(logo_base64)
        else:
            self._render_fallback_header()

    def _load_logo(self) -> Optional[str]:
        """
        Load the logo image and convert it to base64.

        Returns:
            Base64 encoded logo string or None if loading fails
        """
        try:
            if not os.path.exists(self.logo_path):
                return None

            with open(self.logo_path, "rb") as f:
                logo_data = f.read()

            return base64.b64encode(logo_data).decode()

        except (FileNotFoundError, IOError, OSError) as e:
            # Log the error if needed (could be added later)
            return None

    def _render_logo_header(self, logo_base64: str) -> None:
        """
        Render header with logo image.

        Args:
            logo_base64: Base64 encoded logo image string
        """
        st.markdown(f"""
        <div class="{self.ui_config.COMPANY_HEADER_CLASS}">
            <img src="data:image/png;base64,{logo_base64}" 
                 class="{self.ui_config.COMPANY_LOGO_CLASS}" 
                 alt="Secnect Logo">
            <div class="{self.ui_config.BETA_TAG_CLASS}">Beta</div>
        </div>
        """, unsafe_allow_html=True)

    def _render_fallback_header(self) -> None:
        """
        Render fallback header without logo image.

        This method is called when the logo image cannot be loaded,
        providing a text-based alternative.
        """
        st.markdown(f"""
        <div class="{self.ui_config.COMPANY_HEADER_CLASS}">
            <div class="{self.ui_config.COMPANY_NAME_CLASS}">Secnect</div>
            <div class="{self.ui_config.BETA_TAG_CLASS}">Beta</div>
        </div>
        """, unsafe_allow_html=True)

        # Show warning only in development mode or if explicitly enabled
        if self._should_show_logo_warning():
            st.warning(f"Logo file not found at {self.logo_path}")

    def _should_show_logo_warning(self) -> bool:
        """
        Determine whether to show the logo missing warning.

        Returns:
            True if the warning should be displayed
        """
        # Could be based on environment config
        return True  # For now, always show the warning

    def render_custom_header(self, title: str, subtitle: str = None, show_beta: bool = True) -> None:
        """
        Render a custom header with specified title and subtitle.

        Args:
            title: Main title to display
            subtitle: Optional subtitle
            show_beta: Whether to show the beta tag
        """
        beta_tag = f'<div class="{self.ui_config.BETA_TAG_CLASS}">Beta</div>' if show_beta else ""
        subtitle_html = f'<div class="header-subtitle">{subtitle}</div>' if subtitle else ""

        st.markdown(f"""
        <div class="{self.ui_config.COMPANY_HEADER_CLASS}">
            <div class="{self.ui_config.COMPANY_NAME_CLASS}">{title}</div>
            {subtitle_html}
            {beta_tag}
        </div>
        """, unsafe_allow_html=True)

    def render_minimal_header(self, title: str) -> None:
        """
        Render a minimal header with just the title.

        Args:
            title: Title to display
        """
        st.markdown(f"""
        <div class="minimal-header">
            <h1>{title}</h1>
        </div>
        """, unsafe_allow_html=True)


# Convenience instance for easy importing
header_component = HeaderComponent()