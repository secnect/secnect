# page_classes/named_entity_recognition_page.py

"""
Named Entity Recognition Page for the Streamlit application.
Simplified version to avoid NumPy compatibility issues.
"""

import streamlit as st
import pandas as pd
import re
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional

from page_classes.base_page import BasePage
from config import app_config, ui_config
from utils import AppState
from services.file_service import file_service


class NamedEntityRecognitionPage(BasePage):
    """
    Named Entity Recognition page class.

    Provides comprehensive entity extraction from security logs including:
    - IP addresses, usernames, hostnames, email addresses, file paths, timestamps
    """

    def __init__(self):
        """Initialize the NER page."""
        super().__init__()
        self.file_service = file_service
        self.entity_patterns = self._initialize_patterns()

    def render(self) -> None:
        """Render the NER page focused on entity extraction."""
        self.render_section_header("Named Entity Recognition - If you don't see the Demo Button, refresh the site")

        st.markdown("""
        Automatically extract and identify entities from your security logs including IP addresses, 
        usernames, hostnames, email addresses, file paths, and timestamps.
        """)

        # Check if we have log data to analyze
        if not self._check_data_availability():
            self._render_upload_section()
            return

        # Direct entity extraction - no tabs needed
        self._render_entity_extraction()

    def _check_data_availability(self) -> bool:
        """Check if log data is available for analysis."""
        return AppState.has_log_lines() or AppState.get('ner_log_lines') is not None

    def _render_upload_section(self) -> None:
        """Render file upload section for NER analysis."""
        self.display_info("Upload log files to begin Named Entity Recognition analysis.")

        # Check if we can use existing analysis data
        if AppState.has_log_lines():
            if st.button("Use Existing Log Data", type="primary"):
                log_lines = AppState.get('log_lines')
                AppState.set('ner_log_lines', log_lines)
                st.rerun()
                return

        # Upload new file
        st.subheader("Upload Log File for NER Analysis")
        uploaded_file = self.handle_file_upload("Choose a log file for entity recognition")
        
        # Add example data button
        st.markdown("**Or try with example data:**")
        if st.button("🚀 Load Example Security Logs for NER", help="Load 5 example security log entries for entity recognition"):
            self._load_example_logs_for_ner()
            return  # Exit early since we've processed example data

        if uploaded_file:
            result = self.file_service.process_uploaded_file(uploaded_file)
            if result["success"]:
                AppState.set('ner_log_lines', result["log_lines"])
                self.display_success(f"Loaded {len(result['log_lines'])} log lines")
                st.rerun()
            else:
                self.display_error(result["error"])
                
    def _load_example_logs_for_ner(self) -> None:
        """Load example security logs for NER analysis."""
        # Example security log entries with rich entity content
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
        
        # Store the example logs
        AppState.set('ner_log_lines', example_logs)
        
        # Display success message
        self.display_success(f"Loaded {len(example_logs)} example security log lines for entity recognition")
        
        # Display basic info about loaded data
        st.write("**Example data loaded successfully!**")
        st.write("The logs contain various entities like IP addresses, usernames, hostnames, and timestamps.")
        
        # Show preview of the loaded data
        with st.expander("👀 View Example Log Lines"):
            for i, line in enumerate(example_logs, 1):
                st.text(f"{i}. {line}")
        
        # Automatically rerun to show the entity extraction interface
        st.rerun()

    def _render_entity_extraction(self) -> None:
        """Render entity extraction functionality."""
        st.header("Entity Extraction")

        log_lines = self._get_log_lines()

        st.write(f"**Analyzing {len(log_lines):,} log lines**")
        st.write("Extracting: IP Addresses, Usernames, Hostnames, Email Addresses, File Paths, and Timestamps")

        if st.button("Extract All Entities", type="primary"):
            with self.render_loading_spinner("Extracting entities from logs..."):
                # Extract all entity types by default
                entities = self._extract_entities(log_lines)
                AppState.set('extracted_entities', entities)

        # Display results
        entities = AppState.get('extracted_entities')
        if entities:
            self._display_entity_results(entities)

    def _render_pattern_recognition(self) -> None:
        """Render attack pattern recognition."""
        st.header("Pattern Recognition")

        log_lines = self._get_log_lines()

        # Pattern categories
        pattern_categories = st.multiselect(
            "Select Attack Patterns to Detect",
            ["Failed Logins", "Brute Force", "SQL Injection", "XSS", "Port Scanning"],
            default=["Failed Logins", "Brute Force"]
        )

        confidence_threshold = st.slider(
            "Detection Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1
        )

        if st.button("Detect Attack Patterns", type="primary"):
            with self.render_loading_spinner("Analyzing attack patterns..."):
                patterns = self._detect_attack_patterns(log_lines, pattern_categories, confidence_threshold)
                AppState.set('detected_patterns', patterns)

        # Display pattern results
        patterns = AppState.get('detected_patterns')
        if patterns:
            self._display_pattern_results(patterns)

    def _render_timeline_analysis(self) -> None:
        """Render basic timeline analysis."""
        st.header("Timeline Analysis")

        log_lines = self._get_log_lines()

        if st.button("Generate Timeline", type="primary"):
            with self.render_loading_spinner("Creating timeline..."):
                timeline_data = self._create_basic_timeline(log_lines)
                AppState.set('timeline_data', timeline_data)

        # Display timeline
        timeline_data = AppState.get('timeline_data')
        if timeline_data:
            self._display_basic_timeline(timeline_data)

    def _get_log_lines(self) -> List[str]:
        """Get log lines for analysis."""
        return AppState.get('ner_log_lines', [])

    def _initialize_patterns(self) -> Dict[str, str]:
        """Initialize regex patterns for entity extraction."""
        return {
            'ip': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'hostname': r'\b[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*\b',
            'username': r'\b(?:user|username|login|account)[:\s=]+([a-zA-Z0-9_\-\.]+)\b',
            'filepath': r'[/\\](?:[^/\\:\*\?"<>\|]+[/\\])*[^/\\:\*\?"<>\|]*',
            'timestamp': r'\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}'
        }

    def _initialize_attack_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for attack detection."""
        return {
            'Failed Logins': [
                r'failed\s+login',
                r'authentication\s+failed',
                r'invalid\s+user',
                r'login\s+failed'
            ],
            'Brute Force': [
                r'multiple\s+failed\s+attempts',
                r'repeated\s+login\s+failures',
                r'brute\s+force'
            ],
            'SQL Injection': [
                r'union\s+select',
                r'or\s+1\s*=\s*1',
                r'drop\s+table',
                r'insert\s+into'
            ],
            'XSS': [
                r'<script>',
                r'javascript:',
                r'onerror\s*=',
                r'onload\s*='
            ],
            'Port Scanning': [
                r'port\s+scan',
                r'connection\s+attempt',
                r'probe\s+detected'
            ]
        }

    def _extract_entities(self, log_lines: List[str]) -> Dict[str, List[Dict]]:
        """Extract all entity types from log lines."""
        entities = {
            'ips': [],
            'users': [],
            'hosts': [],
            'emails': [],
            'files': [],
            'timestamps': []
        }

        for i, line in enumerate(log_lines):
            if not isinstance(line, str):
                continue  # Skip non-string lines

            line_info = {'line_number': i + 1, 'line': line}

            # Extract IP addresses
            try:
                ips = re.findall(self.entity_patterns['ip'], line)
                for ip in ips:
                    if isinstance(ip, str):
                        entities['ips'].append({**line_info, 'entity': ip, 'type': 'IP'})
            except Exception:
                pass

            # Extract usernames
            try:
                users = re.findall(self.entity_patterns['username'], line, re.IGNORECASE)
                for user in users:
                    if isinstance(user, str) and user.strip():
                        entities['users'].append({**line_info, 'entity': user, 'type': 'User'})
            except Exception:
                pass

            # Extract hostnames (excluding IPs)
            try:
                hosts = re.findall(self.entity_patterns['hostname'], line)
                for host in hosts:
                    if isinstance(host, str) and host.strip():
                        # Check if it's not an IP address
                        if not re.match(self.entity_patterns['ip'], host):
                            entities['hosts'].append({**line_info, 'entity': host, 'type': 'Host'})
            except Exception:
                pass

            # Extract email addresses
            try:
                emails = re.findall(self.entity_patterns['email'], line)
                for email in emails:
                    if isinstance(email, str) and email.strip():
                        entities['emails'].append({**line_info, 'entity': email, 'type': 'Email'})
            except Exception:
                pass

            # Extract file paths
            try:
                files = re.findall(self.entity_patterns['filepath'], line)
                for file_path in files:
                    if isinstance(file_path, str) and len(file_path.strip()) > 3:
                        entities['files'].append({**line_info, 'entity': file_path.strip(), 'type': 'File'})
            except Exception:
                pass

            # Extract timestamps
            try:
                timestamps = re.findall(self.entity_patterns['timestamp'], line)
                for timestamp in timestamps:
                    if isinstance(timestamp, str) and timestamp.strip():
                        entities['timestamps'].append({**line_info, 'entity': timestamp, 'type': 'Timestamp'})
            except Exception:
                pass

        return entities


    def _display_entity_results(self, entities: Dict[str, List[Dict]]) -> None:
        """Display entity extraction results."""
        st.subheader("Extracted Entities")

        # Summary metrics
        unique_entities = {}
        for entity_type, entity_list in entities.items():
            unique_entities[entity_type] = len(set(item['entity'] for item in entity_list))

        # Filter out entity types with zero counts
        non_zero_entities = {k: v for k, v in unique_entities.items() if v > 0}

        # Check if we have any entities to display
        if not non_zero_entities:
            st.info("No entities were extracted from the log data.")
            return

        # Display metrics columns
        # Ensure we don't exceed reasonable column limits (max 6 columns for readability)
        num_cols = min(len(non_zero_entities), 6)
        if num_cols > 0:  # Safety check to ensure we have at least 1 column
            cols = st.columns(num_cols)
            col_idx = 0

            for entity_type, count in non_zero_entities.items():
                if col_idx < num_cols:  # Safety check
                    cols[col_idx].metric(f"Unique {entity_type.title()}", count)
                    col_idx += 1

        # Entity details
        for entity_type, entity_list in entities.items():
            if entity_list:
                st.subheader(f"{entity_type.title()} Found")

                # Count occurrences
                entity_counts = Counter(item['entity'] for item in entity_list)
                top_entities = entity_counts.most_common(10)

                # Display as table
                if top_entities:
                    df = pd.DataFrame(top_entities, columns=['Entity', 'Count'])
                    st.dataframe(df, use_container_width=True)

                # Show detailed view in expander
                with st.expander(f"View All {entity_type.title()}"):
                    df_full = pd.DataFrame(entity_list)
                    st.dataframe(df_full, use_container_width=True)

    def _detect_attack_patterns(self, log_lines: List[str], pattern_categories: List[str],
                                threshold: float) -> List[Dict]:
        """Detect attack patterns in log lines."""
        detected_patterns = []

        for category in pattern_categories:
            if category in self.attack_patterns:
                patterns = self.attack_patterns[category]

                for i, line in enumerate(log_lines):
                    for pattern in patterns:
                        matches = re.findall(pattern, line, re.IGNORECASE)
                        if matches:
                            confidence = min(1.0, len(matches) * 0.3 + 0.4)

                            if confidence >= threshold:
                                detected_patterns.append({
                                    'line_number': i + 1,
                                    'line': line,
                                    'category': category,
                                    'pattern': pattern,
                                    'matches': matches,
                                    'confidence': confidence
                                })

        return detected_patterns

    def _display_pattern_results(self, patterns: List[Dict]) -> None:
        """Display attack pattern detection results."""
        st.subheader("Detected Attack Patterns")

        if not patterns:
            self.display_info("No attack patterns detected with the current settings.")
            return

        # Summary
        df = pd.DataFrame(patterns)
        category_counts = df['category'].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Patterns Detected", len(patterns))
            st.subheader("Pattern Categories")
            st.dataframe(category_counts.reset_index(), use_container_width=True)

        with col2:
            st.subheader("Confidence Distribution")
            confidence_bins = pd.cut(df['confidence'], bins=5).value_counts()
            st.dataframe(confidence_bins.reset_index(), use_container_width=True)

        # Detailed results
        st.subheader("Pattern Details")
        for pattern in patterns[:10]:  # Show top 10
            with st.expander(
                    f"Line {pattern['line_number']}: {pattern['category']} (Confidence: {pattern['confidence']:.2f})"):
                st.write(f"**Line:** {pattern['line']}")
                st.write(f"**Pattern:** `{pattern['pattern']}`")
                st.write(f"**Matches:** {pattern['matches']}")

    def _create_basic_timeline(self, log_lines: List[str]) -> Dict:
        """Create basic timeline analysis."""
        timeline_data = {
            'events_by_hour': Counter(),
            'total_events': 0,
            'timestamp_patterns': []
        }

        # Look for basic timestamp patterns
        for line in log_lines:
            # Simple hour extraction from common formats
            hour_matches = re.findall(r'(\d{2}):\d{2}:\d{2}', line)
            for hour in hour_matches:
                timeline_data['events_by_hour'][int(hour)] += 1
                timeline_data['total_events'] += 1

        return timeline_data

    def _display_basic_timeline(self, timeline_data: Dict) -> None:
        """Display basic timeline analysis."""
        st.subheader("Event Timeline")

        if timeline_data['total_events'] == 0:
            self.display_info("No timestamp patterns found in the logs.")
            return

        st.metric("Total Events with Timestamps", timeline_data['total_events'])

        # Events by hour
        if timeline_data['events_by_hour']:
            st.subheader("Events by Hour of Day")

            hours = list(range(24))
            counts = [timeline_data['events_by_hour'].get(hour, 0) for hour in hours]

            chart_data = pd.DataFrame({
                'Hour': hours,
                'Events': counts
            })

            st.bar_chart(chart_data.set_index('Hour'))

            # Peak hours
            peak_hour = max(timeline_data['events_by_hour'], key=timeline_data['events_by_hour'].get)
            peak_count = timeline_data['events_by_hour'][peak_hour]
            st.write(f"**Peak Activity:** Hour {peak_hour} with {peak_count} events")

    def get_page_title(self) -> str:
        """Get the title for this page."""
        return "Named Entity Recognition"


# Create page instance for easy importing
named_entity_recognition_page = NamedEntityRecognitionPage()
