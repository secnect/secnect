# services/file_service.py

import pandas as pd
import streamlit as st
from typing import List, Optional, Dict, Any, Union
import io
import csv
from pathlib import Path

from config import app_config


class FileService:
    """
    Service class for handling file operations and processing.

    This service provides methods for processing different file types,
    extracting log lines, and validating file formats.
    """

    def __init__(self):
        """Initialize the file service."""
        self.config = app_config
        self.supported_extensions = self.config.ALLOWED_FILE_TYPES

    def process_uploaded_file(self, uploaded_file) -> Dict[str, Any]:
        """
        Process an uploaded file and return structured data.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            Dictionary containing file info and extracted log lines
        """
        if uploaded_file is None:
            return {"success": False, "error": "No file uploaded"}

        try:
            # Get file information
            file_info = self._get_file_info(uploaded_file)

            # Validate file
            validation_result = self._validate_file(uploaded_file)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": validation_result["error"],
                    "file_info": file_info
                }

            # Process file based on type
            log_lines = self._extract_log_lines(uploaded_file)

            return {
                "success": True,
                "log_lines": log_lines,
                "file_info": file_info,
                "line_count": len(log_lines)
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing file: {str(e)}",
                "file_info": self._get_file_info(uploaded_file) if uploaded_file else None
            }

    def _get_file_info(self, uploaded_file) -> Dict[str, Any]:
        """
        Extract information about the uploaded file.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            Dictionary with file information
        """
        return {
            "name": uploaded_file.name,
            "size": uploaded_file.size,
            "type": uploaded_file.type,
            "extension": Path(uploaded_file.name).suffix.lower()
        }

    def _validate_file(self, uploaded_file) -> Dict[str, Any]:
        """
        Validate the uploaded file.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            Dictionary with validation results
        """
        file_info = self._get_file_info(uploaded_file)
        extension = file_info["extension"].lstrip('.')

        # Check file extension
        if extension not in self.supported_extensions:
            return {
                "valid": False,
                "error": f"Unsupported file type: {extension}. "
                         f"Supported types: {', '.join(self.supported_extensions)}"
            }

        # Check file size (optional - could add limits)
        max_size = 100 * 1024 * 1024  # 100MB limit
        if file_info["size"] > max_size:
            return {
                "valid": False,
                "error": f"File too large: {file_info['size']} bytes. Maximum size: {max_size} bytes"
            }

        return {"valid": True}

    def _extract_log_lines(self, uploaded_file) -> List[str]:
        """
        Extract log lines from the uploaded file.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            List of log line strings
        """
        file_info = self._get_file_info(uploaded_file)
        extension = file_info["extension"].lstrip('.')

        if extension == 'csv':
            return self._process_csv_file(uploaded_file)
        else:
            return self._process_text_file(uploaded_file)

    def _process_csv_file(self, uploaded_file) -> List[str]:
        """
        Process a CSV file and extract log lines.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            List of log line strings
        """
        try:
            # Reset file pointer
            uploaded_file.seek(0)

            # Try to read as CSV
            df = pd.read_csv(uploaded_file)

            # Strategy: Use the last column if multiple columns exist,
            # otherwise use the first (and only) column
            if df.shape[1] > 1:
                log_lines = df.iloc[:, -1].tolist()
            else:
                log_lines = df.iloc[:, 0].tolist()

            # Filter out empty/null values
            log_lines = [str(line) for line in log_lines if pd.notna(line) and str(line).strip()]

            return log_lines

        except Exception as e:
            # If CSV parsing fails, try to process as text
            st.warning(f"CSV parsing failed ({str(e)}), trying as text file")
            return self._process_text_file(uploaded_file)

    def _process_text_file(self, uploaded_file) -> List[str]:
        """
        Process a text file and extract log lines.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            List of log line strings
        """
        # Reset file pointer
        uploaded_file.seek(0)

        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']

        for encoding in encodings:
            try:
                uploaded_file.seek(0)
                text = uploaded_file.read().decode(encoding)

                # Split into lines and filter empty lines
                log_lines = [
                    line.strip()
                    for line in text.splitlines()
                    if line.strip()
                ]

                return log_lines

            except UnicodeDecodeError:
                continue

        # If all encodings fail, raise an error
        raise ValueError("Could not decode file with any supported encoding")

    def create_preview(self, log_lines: List[str], num_lines: int = None) -> str:
        """
        Create a preview of the log lines.

        Args:
            log_lines: List of log line strings
            num_lines: Number of lines to include in preview

        Returns:
            Formatted preview string
        """
        if num_lines is None:
            num_lines = self.config.DEFAULT_LOG_PREVIEW_LINES

        preview_lines = log_lines[:num_lines]

        formatted_lines = []
        for i, line in enumerate(preview_lines, 1):
            formatted_lines.append(f"{i}: {line}")

        return "\n".join(formatted_lines)

    def get_file_stats(self, log_lines: List[str]) -> Dict[str, Any]:
        """
        Get statistics about the processed log lines.

        Args:
            log_lines: List of log line strings

        Returns:
            Dictionary with file statistics
        """
        if not log_lines:
            return {"total_lines": 0, "empty_lines": 0, "avg_length": 0}

        line_lengths = [len(line) for line in log_lines]

        return {
            "total_lines": len(log_lines),
            "avg_length": sum(line_lengths) / len(line_lengths),
            "min_length": min(line_lengths),
            "max_length": max(line_lengths),
            "total_characters": sum(line_lengths)
        }

    def save_processed_data(self, log_lines: List[str], filename: str) -> bytes:
        """
        Save processed log lines to a downloadable format.

        Args:
            log_lines: List of log line strings
            filename: Desired filename

        Returns:
            Bytes data for download
        """
        # Create a simple text format
        content = "\n".join(log_lines)
        return content.encode('utf-8')

    def export_to_csv(self, log_lines: List[str]) -> bytes:
        """
        Export log lines to CSV format.

        Args:
            log_lines: List of log line strings

        Returns:
            CSV data as bytes
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(['line_number', 'log_line'])

        # Write data
        for i, line in enumerate(log_lines, 1):
            writer.writerow([i, line])

        return output.getvalue().encode('utf-8')

    def validate_log_format(self, log_lines: List[str]) -> Dict[str, Any]:
        """
        Analyze and validate the format of log lines.

        Args:
            log_lines: List of log line strings

        Returns:
            Dictionary with format analysis results
        """
        if not log_lines:
            return {"valid": False, "reason": "No log lines found"}

        # Basic format validation
        analysis = {
            "total_lines": len(log_lines),
            "has_timestamps": self._check_timestamps(log_lines),
            "has_ip_addresses": self._check_ip_addresses(log_lines),
            "avg_line_length": sum(len(line) for line in log_lines) / len(log_lines),
            "suspected_format": self._detect_log_format(log_lines)
        }

        analysis["valid"] = analysis["total_lines"] > 0

        return analysis

    def _check_timestamps(self, log_lines: List[str], sample_size: int = 10) -> bool:
        """Check if log lines appear to contain timestamps."""
        import re

        # Common timestamp patterns
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}:\d{2}:\d{2}',  # HH:MM:SS
            r'\w{3}\s+\d{1,2}',  # Mon DD
        ]

        sample_lines = log_lines[:min(sample_size, len(log_lines))]

        for pattern in timestamp_patterns:
            matches = sum(1 for line in sample_lines if re.search(pattern, line))
            if matches > len(sample_lines) * 0.5:  # More than 50% match
                return True

        return False

    def _check_ip_addresses(self, log_lines: List[str], sample_size: int = 10) -> bool:
        """Check if log lines appear to contain IP addresses."""
        import re

        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        sample_lines = log_lines[:min(sample_size, len(log_lines))]

        matches = sum(1 for line in sample_lines if re.search(ip_pattern, line))
        return matches > 0

    def _detect_log_format(self, log_lines: List[str]) -> str:
        """Attempt to detect the log format."""
        if not log_lines:
            return "unknown"

        sample_line = log_lines[0].lower()

        if 'apache' in sample_line or 'httpd' in sample_line:
            return "apache"
        elif 'nginx' in sample_line:
            return "nginx"
        elif 'ssh' in sample_line or 'sshd' in sample_line:
            return "ssh"
        elif 'login' in sample_line or 'auth' in sample_line:
            return "authentication"
        elif 'kernel' in sample_line:
            return "system"
        else:
            return "generic"


# Convenience instance for easy importing
file_service = FileService()