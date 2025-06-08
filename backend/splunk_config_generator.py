
"""
Splunk Configuration Generator
Extracted from Jupyter notebook and simplified for Streamlit integration.
"""

import os
import re
import json
import zipfile
from datetime import datetime
from typing import Dict, List, Any, Optional
from io import BytesIO
from dataclasses import dataclass


@dataclass
class SplunkConfig:
    """Data class to hold all Splunk configuration files."""
    props_conf: str = ""
    transforms_conf: str = ""
    savedsearches_conf: str = ""
    macros_conf: str = ""
    indexes_conf: str = ""


class SplunkConfigGenerator:
    """Generate Splunk configuration files based on log analysis."""

    def __init__(self, app_name: str = "custom_app"):
        self.app_name = app_name
        self.configs = {}

    def analyze_log_patterns(self, log_samples: List[str], analysis_results=None) -> List[Any]:
        """Analyze log samples to identify patterns and fields."""
        # Create simple pattern objects for compatibility
        patterns = []

        # Analyze timestamp patterns
        timestamp_regexes = [
            r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}',  # YYYY-MM-DD HH:MM:SS
            r'\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}:\d{2}',   # MM/DD/YYYY HH:MM:SS
            r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}',    # Mon DD HH:MM:SS
        ]

        # Identify common security patterns
        security_patterns = [
            {"name": "Failed Login", "pattern": r"(failed|invalid|unsuccessful).*(login|authentication|sign)", "confidence": 0.8, "fields": ["user", "src_ip", "timestamp"]},
            {"name": "Successful Login", "pattern": r"(success|successful|accepted).*(login|authentication|sign)", "confidence": 0.7, "fields": ["user", "src_ip", "timestamp"]},
            {"name": "User Activity", "pattern": r"user\s*[:=]\s*\w+", "confidence": 0.6, "fields": ["user", "action"]},
        ]

        # Create pattern objects that match what's expected
        class Pattern:
            def __init__(self, name, pattern, confidence, fields):
                self.name = name
                self.pattern = pattern
                self.confidence = confidence
                self.fields = fields

        for sp in security_patterns:
            patterns.append(Pattern(sp["name"], sp["pattern"], sp["confidence"], sp["fields"]))

        return patterns

    def generate_all_configs(self, sourcetype: str, index_name: str, log_lines: List[str], analysis_results=None) -> SplunkConfig:
        """Generate all Splunk configuration files."""
        # Analyze patterns
        patterns = self.analyze_log_patterns(log_lines[:100], analysis_results)

        # Generate each configuration file
        config = SplunkConfig()
        config.props_conf = self._generate_props_conf(sourcetype, patterns)
        config.transforms_conf = self._generate_transforms_conf(sourcetype)
        config.savedsearches_conf = self._generate_savedsearches_conf(sourcetype)
        config.macros_conf = self._generate_macros_conf(sourcetype)
        config.indexes_conf = self._generate_indexes_conf(index_name)

        return config

    def _generate_props_conf(self, sourcetype: str, patterns: List[Any]) -> str:
        """Generate props.conf content for log parsing and field extraction."""
        props_content = f"""[{sourcetype}]
SHOULD_LINEMERGE = false
LINE_BREAKER = ([\\r\\n]+)
TRUNCATE = 10000
TIME_PREFIX =
MAX_TIMESTAMP_LOOKAHEAD = 32
KV_MODE = auto

# Field extractions based on detected patterns
EXTRACT-user = user[\\s]*[:=][\\s]*(?P<user>\\w+)
EXTRACT-src_ip = (?P<src_ip>\\b\\d{{1,3}}\\.\\d{{1,3}}\\.\\d{{1,3}}\\.\\d{{1,3}}\\b)
EXTRACT-action = (?P<action>failed|success|login|logout|authentication)

# Timestamp extraction
TIME_FORMAT = %Y-%m-%d %H:%M:%S

# Calculated fields
EVAL-app_name = "{self.app_name}"
EVAL-log_level = case(match(_raw, "ERROR"), "ERROR", match(_raw, "WARN"), "WARNING", match(_raw, "INFO"), "INFO", 1=1, "DEBUG")
EVAL-event_category = case(match(_raw, "(?i)failed.*login"), "authentication_failure", match(_raw, "(?i)success.*login"), "authentication_success", 1=1, "unknown")
"""
        return props_content

    def _generate_transforms_conf(self, sourcetype: str) -> str:
        """Generate transforms.conf content for field transformations."""
        transforms_content = f"""# Transforms for {sourcetype}

[{sourcetype}_severity_lookup]
REGEX = (ERROR|WARN|INFO|DEBUG)
FORMAT = severity::$1
DEST_KEY = _meta

[{sourcetype}_extract_session]
REGEX = session[_\\-]?id[=:]\\s*(?P<session_id>[A-Za-z0-9\\-_]+)
FORMAT = session_id::$1
DEST_KEY = _meta

[{sourcetype}_normalize_user]
REGEX = user[_\\-]?(?:name|id)?[=:]\\s*(?P<user>[A-Za-z0-9\\-_@.]+)
FORMAT = user::$1
DEST_KEY = _meta
"""
        return transforms_content

    def _generate_savedsearches_conf(self, sourcetype: str) -> str:
        """Generate savedsearches.conf content with saved searches and alerts."""
        savedsearches_content = f"""# Saved searches for {sourcetype}

[{sourcetype}_failed_login_detection]
search = sourcetype={sourcetype} event_category="authentication_failure" | stats count by src_ip, user | where count > 5
dispatch.earliest_time = -1h
dispatch.latest_time = now
enableSched = 1
cron_schedule = */30 * * * *
alert.track = 1
alert.condition = search count > 0
action.email = 1
action.email.subject = Security Alert: Multiple Failed Logins Detected

[{sourcetype}_security_summary]
search = sourcetype={sourcetype} | stats count by event_category | sort -count
dispatch.earliest_time = -24h
dispatch.latest_time = now
enableSched = 1
cron_schedule = 0 9 * * *

[{sourcetype}_hourly_activity]
search = sourcetype={sourcetype} | bucket _time span=1h | stats count by _time, event_category | sort _time
dispatch.earliest_time = -24h
dispatch.latest_time = now
enableSched = 1
cron_schedule = 0 * * * *
"""
        return savedsearches_content

    def _generate_macros_conf(self, sourcetype: str) -> str:
        """Generate macros.conf content with reusable search macros."""
        macros_content = f"""# Macros for {sourcetype}

[{sourcetype}_base_search]
definition = sourcetype={sourcetype}
iseval = 0

[{sourcetype}_failed_logins]
definition = `{sourcetype}_base_search` event_category="authentication_failure"
iseval = 0

[{sourcetype}_successful_logins]
definition = `{sourcetype}_base_search` event_category="authentication_success"
iseval = 0

[{sourcetype}_by_user(1)]
args = user_field
definition = `{sourcetype}_base_search` | stats count by $user_field$ | sort -count
iseval = 0
"""
        return macros_content

    def _generate_indexes_conf(self, index_name: str) -> str:
        """Generate indexes.conf content for index configuration."""
        indexes_content = f"""# Index configuration for {index_name}

[{index_name}]
homePath = $SPLUNK_DB/{index_name}/db
coldPath = $SPLUNK_DB/{index_name}/colddb
thawedPath = $SPLUNK_DB/{index_name}/thaweddb
maxDataSize = auto_high_volume
maxHotBuckets = 10
maxWarmDBCount = 300
repFactor = auto
"""
        return indexes_content

    def generate_spl_queries(self, sourcetype: str, patterns: List[Any]) -> Dict[str, str]:
        """Generate sample SPL queries."""
        queries = {
            "Basic Search": f'sourcetype="{sourcetype}"',
            "Failed Logins": f'sourcetype="{sourcetype}" event_category="authentication_failure"',
            "Successful Logins": f'sourcetype="{sourcetype}" event_category="authentication_success"',
            "Top Users": f'sourcetype="{sourcetype}" | stats count by user | sort -count | head 10',
            "Activity by Hour": f'sourcetype="{sourcetype}" | bucket _time span=1h | stats count by _time | sort _time',
            "Security Events Summary": f'sourcetype="{sourcetype}" | stats count by event_category | sort -count',
            "Failed Login Sources": f'sourcetype="{sourcetype}" event_category="authentication_failure" | stats count by src_ip | sort -count | head 20',
        }
        return queries

    def create_config_package(self, config: SplunkConfig, sourcetype: str, queries: Dict[str, str]) -> bytes:
        """Create a ZIP package with all configuration files."""
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add configuration files
            zip_file.writestr('local/props.conf', config.props_conf)
            zip_file.writestr('local/transforms.conf', config.transforms_conf)
            zip_file.writestr('local/savedsearches.conf', config.savedsearches_conf)
            zip_file.writestr('local/macros.conf', config.macros_conf)
            zip_file.writestr('local/indexes.conf', config.indexes_conf)

            # Add sample queries
            queries_content = "\n".join([f"# {name}\n{query}\n" for name, query in queries.items()])
            zip_file.writestr('sample_queries.spl', queries_content)

            # Add README
            readme_content = f"""# Splunk Configuration Package

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Sourcetype: {sourcetype}

## Installation Instructions:
1. Extract this package to $SPLUNK_HOME/etc/apps/your_app_name/
2. Restart Splunk
3. Configure your data inputs to use the sourcetype: {sourcetype}

## Files Included:
- local/props.conf: Field extractions and parsing rules
- local/transforms.conf: Field transformations
- local/savedsearches.conf: Saved searches and alerts
- local/macros.conf: Reusable search macros
- local/indexes.conf: Index configuration
- sample_queries.spl: Sample SPL queries for testing

## Usage:
After installation, test with: sourcetype="{sourcetype}"
"""
            zip_file.writestr('README.md', readme_content)

        zip_buffer.seek(0)
        return zip_buffer.getvalue()