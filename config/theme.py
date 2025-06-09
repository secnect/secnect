# config/theme.py

import streamlit as st


def apply_dark_theme():
    """Apply dark gradient theme with custom colors."""
    st.markdown("""
    <style>
        /* Import and set primary font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global font family - using Inter as fallback since Aptos may not be available */
        * {
            font-family: 'Aptos Light', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif !important;
        }
        
        /* Streamlit specific font overrides */
        .stApp, .stApp * {
            font-family: 'Aptos Light', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif !important;
        }
        
        /* Main app background with gradient */
        .stApp {
            background: linear-gradient(135deg, #0C293A 0%, #081325 100%);
            color: #ffffff;
            font-family: 'Aptos Light', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif !important;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #0C293A 0%, #081325 100%);
            border-right: 2px solid #2AC3CA;
        }
        
        /* Additional sidebar selectors for different Streamlit versions */
        .css-1d391kg, .css-6qob1r, .css-17lntkn {
            background: linear-gradient(180deg, #0C293A 0%, #081325 100%) !important;
            border-right: 2px solid #2AC3CA !important;
        }
        
        /* Sidebar container */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0C293A 0%, #081325 100%) !important;
            border-right: 2px solid #2AC3CA !important;
        }
        
        /* Sidebar content */
        [data-testid="stSidebar"] > div {
            background: transparent !important;
        }
        
        /* Header styling */
        .css-1rs6os, .css-17eq0hr {
            background: transparent;
            color: #ffffff;
        }
        
        /* Text color and font */
        .stMarkdown, .stText, p, span, div, h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
            font-family: 'Aptos Light', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif !important;
        }
        
        /* Headers with proper font weight */
        h1, h2, h3, h4, h5, h6 {
            font-weight: 300 !important;
        }
        
        /* Button text */
        .stButton > button {
            font-family: 'Aptos Light', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif !important;
            font-weight: 400 !important;
        }
        
        /* Links styling */
        a, a:link, a:visited, a:hover, a:active {
            color: #2AC3CA !important;
            text-decoration: none;
        }
        
        a:hover {
            color: #20A8B5 !important;
            text-decoration: underline;
        }
        
        /* Markdown links */
        .stMarkdown a {
            color: #2AC3CA !important;
        }
        
        .stMarkdown a:hover {
            color: #20A8B5 !important;
        }
        
        /* Divider styling */
        hr {
            border-color: #2AC3CA !important;
            background-color: #2AC3CA !important;
        }
        
        /* Streamlit divider elements */
        .stMarkdown hr {
            border-color: #2AC3CA !important;
            background-color: #2AC3CA !important;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(45deg, #2AC3CA, #20A8B5);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(45deg, #20A8B5, #2AC3CA);
            box-shadow: 0 4px 12px rgba(42, 195, 202, 0.3);
            transform: translateY(-2px);
        }
        
        /* Primary button */
        .stButton > button[kind="primary"] {
            background: linear-gradient(45deg, #2AC3CA, #20A8B5);
            border: 2px solid #2AC3CA;
        }
        
        /* Input fields */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select,
        .stNumberInput > div > div > input {
            background: rgba(42, 195, 202, 0.1);
            color: #ffffff;
            border: 1px solid #2AC3CA;
            border-radius: 6px;
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stSelectbox > div > div > select:focus,
        .stNumberInput > div > div > input:focus {
            border-color: #2AC3CA;
            box-shadow: 0 0 0 2px rgba(42, 195, 202, 0.3);
        }
        
        /* Selectbox dropdown styling */
        .stSelectbox [data-baseweb="select"] {
            background: rgba(42, 195, 202, 0.1);
            border: 1px solid #2AC3CA;
        }
        
        .stSelectbox [data-baseweb="select"] > div {
            background: rgba(12, 41, 58, 0.9);
            border: 1px solid #2AC3CA;
        }
        
        /* Selectbox options */
        .stSelectbox [role="option"] {
            background: rgba(12, 41, 58, 0.9);
            color: #ffffff;
        }
        
        .stSelectbox [role="option"]:hover,
        .stSelectbox [role="option"][aria-selected="true"] {
            background: rgba(42, 195, 202, 0.3) !important;
            color: #ffffff !important;
        }
        
        /* Selectbox dropdown menu */
        .stSelectbox [data-baseweb="menu"] {
            background: rgba(12, 41, 58, 0.95);
            border: 1px solid #2AC3CA;
            border-radius: 6px;
        }
        
        /* File uploader */
        .stFileUploader > div {
            background: rgba(8, 19, 37, 0.8) !important;
            border: 2px dashed #2AC3CA;
            border-radius: 8px;
        }
        
        .stFileUploader label {
            color: #2AC3CA !important;
        }
        
        /* File uploader drag area */
        .stFileUploader [data-testid="stFileUploaderDropzone"] {
            background: rgba(8, 19, 37, 0.8) !important;
            border: 2px dashed #2AC3CA;
            border-radius: 8px;
        }
        
        /* File uploader content area */
        .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] {
            background: transparent !important;
            color: #ffffff !important;
        }
        
        /* File uploader browse button */
        .stFileUploader button {
            background: transparent;
            border: 1px solid #2AC3CA;
            color: #2AC3CA;
            border-radius: 4px;
        }
        
        .stFileUploader button:hover {
            background: rgba(42, 195, 202, 0.2) !important;
            border-color: #2AC3CA !important;
            color: #ffffff !important;
        }
        
        .stFileUploader button:focus {
            background: rgba(42, 195, 202, 0.2) !important;
            border-color: #2AC3CA !important;
            color: #ffffff !important;
            box-shadow: 0 0 0 2px rgba(42, 195, 202, 0.3) !important;
        }
        
        /* Slider */
        .stSlider > div > div > div > div {
            background: #2AC3CA;
        }
        
        /* Slider track background (unfilled portion) */
        .stSlider [data-baseweb="slider"] > div > div {
            background: rgba(42, 195, 202, 0.3) !important;
            height: 4px !important;
        }
        
        /* Slider track fill (filled portion - left side) */
        .stSlider [data-baseweb="slider"] > div > div > div {
            background: #2AC3CA !important;
            height: 4px !important;
        }
        
        /* Slider thumb/handle */
        .stSlider [data-baseweb="slider"] [role="slider"] {
            background: #2AC3CA !important;
            border: 2px solid #2AC3CA !important;
            width: 16px !important;
            height: 16px !important;
        }
        
        /* Additional slider track styling */
        .stSlider [data-baseweb="slider"] [data-baseweb="slider-track"] {
            background: rgba(42, 195, 202, 0.3) !important;
        }
        
        .stSlider [data-baseweb="slider"] [data-baseweb="slider-track"] > div {
            background: #2AC3CA !important;
        }
        
        /* Tabs */
        button[data-baseweb="tab"] {
            background: transparent;
            color: #ffffff;
            border-bottom: 2px solid transparent;
        }
        
        button[data-baseweb="tab"][aria-selected="true"] {
            border-bottom-color: #2AC3CA;
            color: #2AC3CA;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: rgba(42, 195, 202, 0.15);
            border: 1px solid #2AC3CA;
            border-radius: 6px;
            color: #ffffff;
        }
        
        .streamlit-expanderContent {
            background: rgba(12, 41, 58, 0.5);
            border: 1px solid #2AC3CA;
            border-top: none;
            border-radius: 0 0 6px 6px;
        }
        
        /* Code blocks */
        .stCode {
            background: rgba(8, 19, 37, 0.8);
            border: 1px solid #2AC3CA;
            border-radius: 6px;
        }
        
        /* Metrics */
        .metric-container {
            background: rgba(42, 195, 202, 0.1);
            border: 1px solid #2AC3CA;
            border-radius: 8px;
            padding: 1rem;
        }
        
        [data-testid="metric-container"] {
            background: rgba(42, 195, 202, 0.1);
            border: 1px solid #2AC3CA;
            border-radius: 8px;
            padding: 1rem;
        }
        
        /* Success/Info/Warning messages */
        .stSuccess {
            background: rgba(42, 195, 202, 0.2);
            border-left: 4px solid #2AC3CA;
            color: #ffffff;
        }
        
        .stInfo {
            background: rgba(42, 195, 202, 0.15);
            border-left: 4px solid #2AC3CA;
            color: #ffffff;
        }
        
        .stWarning {
            background: rgba(42, 195, 202, 0.15);
            border-left: 4px solid #2AC3CA;
            color: #ffffff;
        }
        
        .stError {
            background: rgba(42, 195, 202, 0.15);
            border-left: 4px solid #2AC3CA;
            color: #ffffff;
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background: #2AC3CA;
        }
        
        /* Divider */
        .stDivider > div {
            border-color: #2AC3CA;
        }
        
        /* Download button */
        .stDownloadButton > button {
            background: linear-gradient(45deg, #2AC3CA, #20A8B5);
            color: white;
            border: 1px solid #2AC3CA;
            border-radius: 6px;
        }
        
        /* Spinner */
        .stSpinner > div {
            border-top-color: #2AC3CA;
        }
        
        /* Table styling */
        .stDataFrame {
            background: rgba(12, 41, 58, 0.5);
            border: 1px solid #2AC3CA;
            border-radius: 6px;
        }
        
        /* Data grid and data frame styling */
        .st-dg {
            border: 1px solid #2AC3CA;
            border-radius: 6px;
            background: rgba(12, 41, 58, 0.5);
        }
        
        .st-df {
            border: 1px solid #2AC3CA;
            border-radius: 6px;
            background: rgba(12, 41, 58, 0.5);
        }
        
        /* Sidebar header styling */
        .css-1544g2n {
            color: #2AC3CA;
            font-weight: bold;
        }
        
        /* st-cu component styling */
        .st-cu {
            background: rgba(42, 195, 202, 0.1);
            border: 1px solid #2AC3CA;
            border-radius: 6px;
            color: #ffffff;
        }
        
        .st-cu:hover {
            background: rgba(42, 195, 202, 0.2);
            border-color: #2AC3CA;
        }
    </style>
    """, unsafe_allow_html=True)


def get_custom_css_classes():
    """Return custom CSS classes for specific components."""
    return """
    <style>
        /* Custom classes */
        .company-header { 
            display: flex; 
            align-items: center; 
            margin-bottom: 20px; 
        }
        .company-name { 
            font-size: 24px; 
            font-weight: bold; 
            color: #2AC3CA; 
            margin-right: 10px; 
        }
        .company-logo {
            height: 80px;
            width: auto;
            max-width: 200px;
            margin-right: 15px;
            filter: brightness(1.1);
        }
        
        /* Mobile responsive logo */
        @media (max-width: 768px) {
            .company-logo {
                height: 60px;
                max-width: 150px;
            }
        }
        .beta-tag { 
            background: linear-gradient(45deg, #2AC3CA, #20A8B5);
            color: white; 
            font-size: 12px; 
            font-weight: bold;
            padding: 3px 8px; 
            border-radius: 4px; 
            text-transform: uppercase; 
            box-shadow: 0 2px 4px rgba(42, 195, 202, 0.3);
        }
        .splunk-section {
            background: rgba(42, 195, 202, 0.1);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #2AC3CA;
            margin: 15px 0;
        }
        .config-preview {
            background: rgba(8, 19, 37, 0.8);
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 12px;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #2AC3CA;
            color: #ffffff;
        }
        
        /* Custom highlight colors */
        .highlight-red {
            color: #2AC3CA;
            font-weight: bold;
        }
        
        .highlight-orange {
            color: #2AC3CA;
            font-weight: bold;
        }
    </style>
    """


def apply_matplotlib_dark_theme():
    """Configure matplotlib to use dark theme that matches the app."""
    import matplotlib.pyplot as plt

    # Set matplotlib to use dark theme
    plt.style.use('dark_background')

    # Return configuration for consistent theming
    return {
        'facecolor': '#0C293A',
        'edgecolor': '#2AC3CA',
        'color': '#2AC3CA',
        'text_color': 'white',
        'threshold_color': '#2AC3CA',
        'mean_color': '#2AC3CA'
    }


def apply_complete_theme():
    """Apply the complete theme including all CSS and configurations."""
    apply_dark_theme()
    st.markdown(get_custom_css_classes(), unsafe_allow_html=True)