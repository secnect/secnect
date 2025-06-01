import streamlit as st
import pandas as pd
import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import io

class LogParser:
    """
    A flexible log parser that can handle multiple log formats commonly found in cybersecurity contexts.
    Designed to extract structured features from raw log lines for normalization.
    """
    
    def __init__(self):
        # Define regex patterns for different log formats
        self.patterns = {
            'hdfs': {
                'pattern': r'^(\d{6}\s\d{6})\s+(\w+)\s+(\S+):\s+(.*)$',
                'fields': ['timestamp', 'log_level', 'component', 'message']
            },
            'windows': {
                'pattern': r'^(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\s+(\w+)\s+(\S+)\s+(.*)$',
                'fields': ['timestamp', 'log_level', 'component', 'message']
            },
            'apache': {
                'pattern': r'^(\S+)\s+\S+\s+\S+\s+\[([^\]]+)\]\s+"([^"]+)"\s+(\d+)\s+(\S+)(?:\s+"([^"]*)")?(?:\s+"([^"]*)")?',
                'fields': ['ip_address', 'timestamp', 'request', 'status_code', 'response_size', 'referrer', 'user_agent']
            },
            'linux_syslog': {
                'pattern': r'^(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+([^:]+):\s+(.*)$',
                'fields': ['timestamp', 'hostname', 'component', 'message']
            },
            'generic': {
                'pattern': r'^(\S+)\s+(\S+)\s+(.*)$',
                'fields': ['timestamp', 'component', 'message']
            }
        }
    
    def detect_log_format(self, sample_lines: List[str]) -> str:
        """
        Automatically detect the log format based on sample lines.
        Returns the most likely format type.
        """
        format_scores = {}
        
        for format_name, format_info in self.patterns.items():
            matches = 0
            for line in sample_lines[:10]:  # Test first 10 lines
                if re.match(format_info['pattern'], line.strip()):
                    matches += 1
            format_scores[format_name] = matches
        
        # Return format with highest match score
        best_format = max(format_scores, key=format_scores.get)
        return best_format if format_scores[best_format] > 0 else 'generic'
    
    def extract_features(self, log_line: str, format_type: str) -> Dict:
        """
        Extract structured features from a single log line based on the specified format.
        """
        if format_type not in self.patterns:
            format_type = 'generic'
        
        pattern_info = self.patterns[format_type]
        match = re.match(pattern_info['pattern'], log_line.strip())
        
        if match:
            features = dict(zip(pattern_info['fields'], match.groups()))
            
            # Add common derived features
            features['raw_log'] = log_line.strip()
            features['log_length'] = len(log_line.strip())
            features['format_type'] = format_type
            
            # Extract additional security-relevant features
            features.update(self._extract_security_features(log_line, features))
            
            return features
        else:
            # Fallback for unparseable lines
            return {
                'raw_log': log_line.strip(),
                'log_length': len(log_line.strip()),
                'format_type': 'unparsed',
                'message': log_line.strip()
            }
    
    def _extract_security_features(self, log_line: str, base_features: Dict) -> Dict:
        """
        Extract additional security-relevant features from log content.
        """
        security_features = {}
        
        # Extract IP addresses
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        ips = re.findall(ip_pattern, log_line)
        security_features['ip_addresses'] = ips if ips else []
        
        # Extract potential error indicators
        error_keywords = ['error', 'failed', 'denied', 'unauthorized', 'forbidden', 'exception']
        security_features['has_error'] = any(keyword in log_line.lower() for keyword in error_keywords)
        
        # Extract potential security event indicators
        security_keywords = ['login', 'logout', 'authentication', 'access', 'permission', 'firewall', 'intrusion']
        security_features['security_event'] = any(keyword in log_line.lower() for keyword in security_keywords)
        
        # Extract severity level if present in message
        severity_pattern = r'\b(CRITICAL|HIGH|MEDIUM|LOW|INFO|WARN|ERROR|DEBUG)\b'
        severity_match = re.search(severity_pattern, log_line.upper())
        security_features['severity'] = severity_match.group(1) if severity_match else None
        
        # Count suspicious patterns
        suspicious_patterns = [r'\.\./', r'<script', r'SELECT.*FROM', r'DROP\s+TABLE']
        security_features['suspicious_patterns'] = sum(1 for pattern in suspicious_patterns 
                                                     if re.search(pattern, log_line, re.IGNORECASE))
        
        return security_features
    
    def parse_logs(self, log_content: str, format_type: Optional[str] = None) -> List[Dict]:
        """
        Parse multiple log lines and return structured features for each.
        """
        lines = [line for line in log_content.split('\n') if line.strip()]
        
        if not format_type:
            format_type = self.detect_log_format(lines)
        
        parsed_logs = []
        for line in lines:
            if line.strip():  # Skip empty lines
                features = self.extract_features(line, format_type)
                features['line_number'] = len(parsed_logs) + 1
                parsed_logs.append(features)
        
        return parsed_logs

def create_sample_logs() -> Dict[str, str]:
    """
    Create sample log data for demonstration purposes.
    These represent common log formats found in cybersecurity environments.
    """
    samples = {
        "HDFS Logs": """081109 203615 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-5911576436174369298 terminating
081109 203615 INFO dfs.DataNode$DataXceiver: Receiving block blk_-5911576436174369298 src: /10.250.10.6:54106 dest: /10.250.10.6:50010
081109 203616 ERROR dfs.FSNamesystem: Failed to process request from 192.168.1.100 due to authentication failure
081109 203617 WARN dfs.DataNode: Disk error on /data/hdfs occurred
081109 203618 INFO dfs.DataNode$BlockReceiver: Receiving block from /10.250.19.40:33660""",
        
        "Windows Event Logs": """2024-01-15 10:30:25 ERROR Security.Audit User login failed for account DOMAIN\\user123 from 192.168.1.45
2024-01-15 10:30:30 INFO Application.Service Service 'WebServer' started successfully
2024-01-15 10:31:15 WARN Security.Policy Policy violation detected: unauthorized access attempt
2024-01-15 10:31:45 ERROR System.Authentication Authentication failed for user admin from 10.0.0.50
2024-01-15 10:32:00 INFO Security.Audit Successful login for DOMAIN\\administrator from 192.168.1.10""",
        
        "Apache Access Logs": """192.168.1.100 - - [15/Jan/2024:10:30:25 +0000] "GET /admin/login HTTP/1.1" 403 2326 "-" "Mozilla/5.0"
10.0.0.50 - - [15/Jan/2024:10:30:30 +0000] "POST /api/data HTTP/1.1" 200 1543 "https://example.com" "curl/7.68.0"
192.168.1.200 - - [15/Jan/2024:10:31:15 +0000] "GET /../../../etc/passwd HTTP/1.1" 404 209 "-" "Nikto/2.1.6"
172.16.0.10 - - [15/Jan/2024:10:31:45 +0000] "DELETE /users/1 HTTP/1.1" 401 87 "-" "PostmanRuntime/7.29.2"
192.168.1.150 - - [15/Jan/2024:10:32:00 +0000] "GET /dashboard HTTP/1.1" 200 5432 "https://app.example.com" "Mozilla/5.0""",
        
        "Linux Syslog": """Jan 15 10:30:25 webserver01 sshd[12345]: Failed password for user from 192.168.1.100 port 22 ssh2
Jan 15 10:30:30 webserver01 kernel: firewall: denied connection from 10.0.0.50 to port 443
Jan 15 10:31:15 webserver01 apache2[54321]: authentication failure for user admin
Jan 15 10:31:45 webserver01 systemd[1]: Started security monitoring service
Jan 15 10:32:00 webserver01 sshd[12346]: Accepted publickey for root from 192.168.1.10 port 22"""
    }
    return samples

def main():
    st.title("🔒 Cybersecurity Log Feature Extraction Prototype")
    st.markdown("""
    This prototype demonstrates how to extract structured features from raw security logs for normalization and ingestion into SIEM tools like Splunk.
    
    **Features:**
    - Automatic log format detection
    - Structured feature extraction
    - Security-relevant pattern identification
    - Export capabilities for downstream processing
    """)
    
    # Initialize the log parser
    parser = LogParser()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["Sample Data", "Upload File"]
    )
    
    log_content = ""
    
    if data_source == "Sample Data":
        samples = create_sample_logs()
        selected_sample = st.sidebar.selectbox("Choose Sample Log Type:", list(samples.keys()))
        log_content = samples[selected_sample]
        
        st.subheader(f"📄 Sample Data: {selected_sample}")
        with st.expander("View Raw Log Content", expanded=True):
            st.code(log_content, language="text")
    
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload Log File",
            type=['txt', 'log'],
            help="Upload a text file containing log entries"
        )
        
        if uploaded_file is not None:
            # Read the uploaded file
            log_content = str(uploaded_file.read(), "utf-8")
            st.subheader("📄 Uploaded Log File")
            with st.expander("View Raw Log Content"):
                st.code(log_content[:2000] + "..." if len(log_content) > 2000 else log_content, language="text")
    
    # Process logs if content is available
    if log_content:
        st.subheader("🔍 Log Analysis")
        
        # Parse the logs
        with st.spinner("Parsing logs and extracting features..."):
            parsed_logs = parser.parse_logs(log_content)
        
        if parsed_logs:
            # Create DataFrame for structured view
            df = pd.DataFrame(parsed_logs)
            
            # Display summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Log Lines", len(parsed_logs))
            with col2:
                error_count = sum(1 for log in parsed_logs if log.get('has_error', False))
                st.metric("Error Events", error_count)
            with col3:
                security_count = sum(1 for log in parsed_logs if log.get('security_event', False))
                st.metric("Security Events", security_count)
            with col4:
                unique_ips = set()
                for log in parsed_logs:
                    unique_ips.update(log.get('ip_addresses', []))
                st.metric("Unique IP Addresses", len(unique_ips))
            
            # Display format detection results
            detected_format = parser.detect_log_format(log_content.split('\n')[:10])
            st.info(f"🎯 **Detected Log Format:** {detected_format.upper()}")
            
            # Feature extraction results
            st.subheader("📊 Extracted Features")
            
            # Select columns to display
            available_columns = list(df.columns)
            
            # Create safe default columns by checking which ones actually exist
            preferred_defaults = ['line_number', 'timestamp', 'log_level', 'component', 'message', 'has_error']
            safe_defaults = [col for col in preferred_defaults if col in available_columns]
            
            # If no preferred defaults exist, use the first few available columns
            if not safe_defaults:
                safe_defaults = available_columns[:min(6, len(available_columns))]
            
            selected_columns = st.multiselect(
                "Select columns to display:",
                available_columns,
                default=safe_defaults
            )
            
            if selected_columns:
                st.dataframe(df[selected_columns], use_container_width=True)
            
            # Security analysis section
            st.subheader("🛡️ Security Analysis")
            
            # Security event summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Error Distribution:**")
                error_df = df.groupby('has_error').size().reset_index(name='count')
                st.bar_chart(error_df.set_index('has_error'))
            
            with col2:
                st.write("**Security Events:**")
                security_df = df.groupby('security_event').size().reset_index(name='count')
                st.bar_chart(security_df.set_index('security_event'))
            
            # Display suspicious activities
            suspicious_logs = [log for log in parsed_logs if log.get('suspicious_patterns', 0) > 0]
            if suspicious_logs:
                st.warning(f"⚠️ **{len(suspicious_logs)} suspicious patterns detected!**")
                suspicious_df = pd.DataFrame(suspicious_logs)
                st.dataframe(suspicious_df[['line_number', 'raw_log', 'suspicious_patterns']], use_container_width=True)
            
            # Export functionality
            st.subheader("💾 Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export as JSON
                json_data = json.dumps(parsed_logs, indent=2, default=str)
                st.download_button(
                    label="📄 Download as JSON",
                    data=json_data,
                    file_name="extracted_log_features.json",
                    mime="application/json"
                )
            
            with col2:
                # Export as CSV
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download as CSV",
                    data=csv_data,
                    file_name="extracted_log_features.csv",
                    mime="text/csv"
                )
            
            # Technical details
            with st.expander("🔧 Technical Details"):
                st.write("**Parser Configuration:**")
                st.json({
                    "supported_formats": list(parser.patterns.keys()),
                    "detected_format": detected_format,
                    "total_features_extracted": len(df.columns),
                    "parsing_success_rate": f"{len([log for log in parsed_logs if log['format_type'] != 'unparsed'])/len(parsed_logs)*100:.1f}%"
                })
        
        else:
            st.error("No valid log entries found. Please check your log format.")
    
    else:
        st.info("👈 Please select a data source from the sidebar to begin analysis.")
    
    # Footer with usage instructions
    st.markdown("---")
    st.markdown("""
    ### 📋 Usage Instructions:
    
    1. **Select a data source** from the sidebar (sample data or upload your own)
    2. **Review the raw log content** to understand the format
    3. **Analyze extracted features** in the structured table view
    4. **Review security analysis** for potential threats and anomalies
    5. **Export the results** as JSON or CSV for integration with SIEM tools
    
    ### 🎯 Next Steps for Production:
    - Integrate with real-time log streams
    - Implement advanced anomaly detection
    - Add support for additional log formats
    - Create automated alerting for security events
    - Develop custom parsing rules for organization-specific formats
    """)

if __name__ == "__main__":
    main()