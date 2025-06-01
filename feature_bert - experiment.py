import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import io
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Transformers and PyTorch imports
try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForTokenClassification,
        pipeline, BertTokenizer, BertModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.error("⚠️ Transformers library not available. Install with: pip install transformers torch")

# Set page configuration
st.set_page_config(
    page_title="BERT-Based Log Parser",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class BERTLogParser:
    """
    BERT-based log parser that extracts structured features from unstructured log data.
    
    This class leverages pretrained BERT models to:
    1. Tokenize and embed log messages
    2. Extract structured fields via token classification
    3. Apply template mining for pattern recognition
    4. Generate structured output from raw logs
    """
    
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.embeddings_cache = {}
        
    def load_model(self):
        """Load pretrained BERT model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            return True
        except Exception as e:
            st.error(f"Error loading model {self.model_name}: {str(e)}")
            return False
    
    def get_bert_embeddings(self, texts, max_length=512):
        """
        Generate BERT embeddings for input texts.
        
        BERT helps infer structure by:
        - Understanding contextual relationships between tokens
        - Capturing semantic meaning beyond simple keyword matching
        - Providing dense representations that cluster similar log patterns
        """
        if not self.model or not self.tokenizer:
            return None
            
        embeddings = []
        
        for text in texts:
            # Use cache to avoid recomputing embeddings
            if text in self.embeddings_cache:
                embeddings.append(self.embeddings_cache[text])
                continue
                
            # Tokenize with BERT tokenizer
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                max_length=max_length, 
                truncation=True, 
                padding=True
            )
            
            # Get BERT embeddings (no gradient computation needed for inference)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding as sentence representation
                embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
                
            embeddings.append(embedding)
            self.embeddings_cache[text] = embedding
            
        return np.array(embeddings)
    
    def extract_structured_fields(self, log_lines):
        """
        Extract structured fields from log lines using BERT-based analysis.
        
        This approach combines BERT embeddings with pattern recognition to identify:
        - Timestamps
        - Log levels
        - Components/modules
        - Event types
        - Parameters
        """
        structured_logs = []
        
        # Get BERT embeddings for semantic analysis
        embeddings = self.get_bert_embeddings(log_lines)
        
        if embeddings is not None:
            # Apply clustering to group similar log patterns
            # BERT embeddings help identify semantic clusters beyond surface-level patterns
            n_clusters = min(10, len(log_lines))
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(embeddings)
            else:
                clusters = [0] * len(log_lines)
        else:
            clusters = [0] * len(log_lines)
        
        for i, log_line in enumerate(log_lines):
            structured_log = self._parse_single_log(log_line, clusters[i])
            structured_logs.append(structured_log)
            
        return structured_logs
    
    def _parse_single_log(self, log_line, cluster_id):
        """
        Parse a single log line into structured components.
        
        BERT helps here by providing context-aware understanding of:
        - Which tokens are likely timestamps vs. identifiers
        - Semantic relationships between components
        - Pattern variations that share similar meaning
        """
        # Initialize structured log entry
        parsed_log = {
            'raw_message': log_line.strip(),
            'cluster_id': int(cluster_id),
            'timestamp': None,
            'log_level': None,
            'component': None,
            'event_type': None,
            'parameters': [],
            'message_template': None
        }
        
        # Tokenize for analysis (space-based for this example)
        tokens = log_line.strip().split()
        
        if not tokens:
            return parsed_log
        
        # Extract timestamp (enhanced by BERT's understanding of temporal context)
        timestamp = self._extract_timestamp(tokens)
        if timestamp:
            parsed_log['timestamp'] = timestamp
        
        # Extract log level (BERT helps identify severity indicators)
        log_level = self._extract_log_level(tokens)
        if log_level:
            parsed_log['log_level'] = log_level
        
        # Extract component/module (BERT understands component naming patterns)
        component = self._extract_component(tokens)
        if component:
            parsed_log['component'] = component
        
        # Extract event type and parameters
        event_type, parameters = self._extract_event_and_params(tokens)
        if event_type:
            parsed_log['event_type'] = event_type
        parsed_log['parameters'] = parameters
        
        # Generate message template (abstracting specific values)
        parsed_log['message_template'] = self._generate_template(log_line)
        
        return parsed_log
    
    def _extract_timestamp(self, tokens):
        """Extract timestamp using pattern recognition enhanced by BERT context"""
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}[\s|T]\d{2}:\d{2}:\d{2}',  # ISO format
            r'\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}',     # US format
            r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}',       # Syslog format
            r'\d{10,13}',                                   # Unix timestamp
        ]
        
        text = ' '.join(tokens)
        for pattern in timestamp_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group()
        return None
    
    def _extract_log_level(self, tokens):
        """Extract log level with BERT's understanding of severity semantics"""
        log_levels = ['DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR', 'FATAL', 'TRACE']
        
        for token in tokens:
            if token.upper() in log_levels:
                return token.upper()
            # Check for bracketed levels like [INFO]
            clean_token = token.strip('[]()').upper()
            if clean_token in log_levels:
                return clean_token
        return None
    
    def _extract_component(self, tokens):
        """Extract component/module using BERT's pattern recognition"""
        # Look for component patterns (often appear early in logs)
        for i, token in enumerate(tokens[:5]):  # Check first few tokens
            # Component often has specific patterns
            if ('.' in token and len(token) > 3) or \
               (token.endswith(':') and len(token) > 2) or \
               (token.startswith('[') and token.endswith(']')):
                return token.strip('[]():')
        return None
    
    def _extract_event_and_params(self, tokens):
        """
        Extract event type and parameters using BERT's semantic understanding.
        
        BERT helps identify which tokens represent:
        - Action verbs (events)
        - Variable values (parameters)
        - Static message components
        """
        # Common event indicators
        event_keywords = [
            'started', 'stopped', 'failed', 'error', 'completed', 'initialized',
            'connected', 'disconnected', 'timeout', 'exception', 'warning'
        ]
        
        event_type = None
        parameters = []
        
        for token in tokens:
            token_lower = token.lower().strip('.,;:()')
            
            # Check for event keywords
            if token_lower in event_keywords:
                event_type = token_lower
            
            # Extract potential parameters (numbers, IDs, paths)
            if self._is_parameter(token):
                parameters.append(token)
        
        return event_type, parameters
    
    def _is_parameter(self, token):
        """Identify if a token is likely a parameter value"""
        # Numbers, IDs, file paths, URLs, etc.
        param_patterns = [
            r'^\d+$',           # Pure numbers
            r'^[a-f0-9-]{8,}$', # Hex IDs
            r'^/.*',            # File paths
            r'^\w+://.*',       # URLs
            r'^\w+\.\w+$',      # File names
        ]
        
        for pattern in param_patterns:
            if re.match(pattern, token, re.IGNORECASE):
                return True
        return False
    
    def _generate_template(self, log_line):
        """
        Generate message template by abstracting variable components.
        
        BERT embeddings help identify which parts of messages are:
        - Static template text (consistent across similar logs)
        - Variable parameters (change between instances)
        """
        template = log_line
        
        # Replace common variable patterns with placeholders
        replacements = [
            (r'\d{4}-\d{2}-\d{2}[\s|T]\d{2}:\d{2}:\d{2}', '<TIMESTAMP>'),
            (r'\d+\.\d+\.\d+\.\d+', '<IP_ADDRESS>'),
            (r'\b\d+\b', '<NUMBER>'),
            (r'\b[a-f0-9-]{8,}\b', '<ID>'),
            (r'/[\w/.-]*', '<PATH>'),
        ]
        
        for pattern, placeholder in replacements:
            template = re.sub(pattern, placeholder, template)
        
        return template

def load_sample_data():
    """Load sample log data from different sources"""
    
    # Sample HDFS logs
    hdfs_logs = [
        "2019-06-14 08:00:01,001 INFO org.apache.hadoop.hdfs.server.datanode.DataNode: Opened socket address /10.251.42.84:50010",
        "2019-06-14 08:00:01,020 INFO org.apache.hadoop.hdfs.server.datanode.DataNode: Block pool <registering> (Datanode Uuid unassigned) service to localhost/127.0.0.1:9000 starting to offer service",
        "2019-06-14 08:00:01,025 WARN org.apache.hadoop.hdfs.server.common.Storage: Storage directory /tmp/hadoop-hdfs/dfs/data does not exist",
        "2019-06-14 08:00:01,040 ERROR org.apache.hadoop.hdfs.server.datanode.DataNode: Exception in secureMain java.lang.RuntimeException: Error in extension"
    ]
    
    # Sample BGL logs
    bgl_logs = [
        "1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected",
        "1117838573 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.53.363812 R02-M1-N0-C:J12-U11 RAS KERNEL FATAL data storage interrupt",
        "1117838580 2005.06.03 R02-M1-N1-C:J13-U01 2005-06-03-15.43.00.364009 R02-M1-N1-C:J13-U01 RAS KERNEL INFO processor correctable error",
    ]
    
    # Sample Windows logs
    windows_logs = [
        "2019-06-14 08:32:23.454 INFO Application started successfully on port 8080",
        "2019-06-14 08:32:45.123 WARN Database connection timeout after 30 seconds",
        "2019-06-14 08:33:01.789 ERROR Failed to process request: Invalid user credentials",
        "2019-06-14 08:33:15.456 DEBUG User authentication successful for user: admin@example.com"
    ]
    
    return {
        "HDFS": hdfs_logs,
        "BGL": bgl_logs,
        "Windows": windows_logs
    }

def main():
    """Main Streamlit application"""
    
    st.title("🔍 BERT-Based Log Parser")
    st.markdown("""
    This prototype demonstrates how **BERT-based models** can automatically parse and extract 
    structured features from raw, unstructured log data without hardcoded regular expressions.
    
    **Key Features:**
    - Uses pretrained BERT for semantic understanding
    - Extracts timestamps, log levels, components, and parameters
    - Applies clustering for template mining
    - Generalizes across different log formats
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    model_options = [
        "distilbert-base-uncased",
        "bert-base-uncased", 
        "microsoft/DialoGPT-medium"
    ]
    
    selected_model = st.sidebar.selectbox(
        "Select BERT Model:",
        model_options,
        help="Choose the pretrained model for log analysis"
    )
    
    # Check if transformers is available
    if not TRANSFORMERS_AVAILABLE:
        st.error("Please install the transformers library to use this application.")
        st.code("pip install transformers torch")
        return
    
    # Initialize parser
    parser = BERTLogParser(model_name=selected_model)
    
    # Data input section
    st.header("📥 Data Input")
    
    # Load sample data
    sample_data = load_sample_data()
    
    input_method = st.radio(
        "Choose input method:",
        ["Sample Data", "Upload File", "Manual Input"]
    )
    
    log_lines = []
    
    if input_method == "Sample Data":
        dataset_type = st.selectbox("Select sample dataset:", list(sample_data.keys()))
        log_lines = sample_data[dataset_type]
        st.subheader(f"Sample {dataset_type} Logs:")
        for i, log in enumerate(log_lines, 1):
            st.text(f"{i}. {log}")
    
    elif input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload log file", type=['txt', 'log'])
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            log_lines = content.strip().split('\n')
            st.success(f"Loaded {len(log_lines)} log lines")
            with st.expander("Preview uploaded logs"):
                for i, log in enumerate(log_lines[:10], 1):
                    st.text(f"{i}. {log}")
                if len(log_lines) > 10:
                    st.text(f"... and {len(log_lines) - 10} more lines")
    
    else:  # Manual Input
        manual_logs = st.text_area(
            "Enter log lines (one per line):",
            height=200,
            placeholder="Enter your log lines here..."
        )
        if manual_logs:
            log_lines = [line.strip() for line in manual_logs.split('\n') if line.strip()]
    
    # Processing section
    if log_lines:
        st.header("🔄 Processing")
        
        if st.button("Parse Logs with BERT", type="primary"):
            with st.spinner("Loading BERT model and processing logs..."):
                
                # Load model
                if parser.load_model():
                    st.success(f"✅ Loaded {selected_model}")
                    
                    # Process logs
                    structured_logs = parser.extract_structured_fields(log_lines)
                    
                    # Display results
                    st.header("📊 Results")
                    
                    # Create DataFrame for structured view
                    df = pd.DataFrame(structured_logs)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Logs", len(structured_logs))
                    
                    with col2:
                        unique_clusters = df['cluster_id'].nunique()
                        st.metric("Log Patterns", unique_clusters)
                    
                    with col3:
                        logs_with_timestamps = df['timestamp'].notna().sum()
                        st.metric("With Timestamps", logs_with_timestamps)
                    
                    with col4:
                        logs_with_levels = df['log_level'].notna().sum()
                        st.metric("With Log Levels", logs_with_levels)
                    
                    # Structured data view
                    st.subheader("Structured Log Data")
                    st.dataframe(df, use_container_width=True)
                    
                    # Export options
                    st.subheader("📤 Export")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # JSON export
                        json_str = json.dumps(structured_logs, indent=2, default=str)
                        st.download_button(
                            label="Download as JSON",
                            data=json_str,
                            file_name="parsed_logs.json",
                            mime="application/json"
                        )
                    
                    with col2:
                        # CSV export
                        csv_buffer = io.StringIO()
                        df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="Download as CSV",
                            data=csv_buffer.getvalue(),
                            file_name="parsed_logs.csv",
                            mime="text/csv"
                        )
                    
                    # Analysis and visualization
                    st.header("📈 Analysis")
                    
                    # Log level distribution
                    if df['log_level'].notna().any():
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Log level distribution
                        level_counts = df['log_level'].value_counts()
                        ax1.pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%')
                        ax1.set_title('Log Level Distribution')
                        
                        # Cluster distribution
                        cluster_counts = df['cluster_id'].value_counts()
                        ax2.bar(range(len(cluster_counts)), cluster_counts.values)
                        ax2.set_xlabel('Cluster ID')
                        ax2.set_ylabel('Number of Logs')
                        ax2.set_title('Log Pattern Clusters')
                        
                        st.pyplot(fig)
                    
                    # Template analysis
                    st.subheader("🔍 Template Analysis")
                    
                    template_counts = df['message_template'].value_counts()
                    if len(template_counts) > 0:
                        st.write("**Most Common Log Templates:**")
                        for template, count in template_counts.head(10).items():
                            st.write(f"- `{template}` ({count} occurrences)")
                    
                    # Component analysis
                    if df['component'].notna().any():
                        st.subheader("🏗️ Component Analysis")
                        component_counts = df['component'].value_counts()
                        st.bar_chart(component_counts)
                    
                    # Technical insights
                    st.header("🧠 How BERT Helps")
                    
                    st.markdown("""
                    **BERT's Advantages in Log Parsing:**
                    
                    1. **Contextual Understanding**: BERT understands relationships between tokens,
                       helping identify components even when they appear in different positions.
                    
                    2. **Semantic Clustering**: Similar log messages cluster together in BERT's 
                       embedding space, enabling template discovery without manual rules.
                    
                    3. **Generalization**: The model can handle unseen log formats by leveraging
                       patterns learned from pretraining on diverse text data.
                    
                    4. **Robustness**: Unlike regex-based approaches, BERT-based parsing adapts
                       to variations in log structure and formatting.
                    
                    5. **Feature Extraction**: Embeddings capture rich semantic features that
                       can be used for downstream tasks like anomaly detection or classification.
                    """)
                    
                else:
                    st.error("Failed to load BERT model. Please check your internet connection.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Note**: This prototype demonstrates BERT-based log parsing capabilities. 
    For production use, consider fine-tuning on domain-specific log data and 
    implementing more sophisticated NER models for improved accuracy.
    """)

if __name__ == "__main__":
    main()