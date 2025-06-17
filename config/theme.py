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
        
        /* Main app background with black background */
        .stApp {
            background: #000000;
            color: #ffffff;
            font-family: 'Aptos Light', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif !important;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: #000000;
            border-right: 2px solid #606060;
        }
        
        /* Additional sidebar selectors for different Streamlit versions */
        .css-1d391kg, .css-6qob1r, .css-17lntkn {
            background: #000000 !important;
            border-right: 2px solid #AAAAAA !important;
        }
        
        /* Sidebar container */
        [data-testid="stSidebar"] {
            background: #000000 !important;
            border-right: 2px solid #AAAAAA !important;
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
            color: #AAAAAA !important;
            text-decoration: none;
        }
        
        a:hover {
            color: #555555 !important;
            text-decoration: underline;
        }
        
        /* Markdown links */
        .stMarkdown a {
            color: #AAAAAA !important;
        }
        
        .stMarkdown a:hover {
            color: #555555 !important;
        }
        
        /* Divider styling */
        hr {
            border-color: #AAAAAA !important;
            background-color: #AAAAAA !important;
        }
        
        /* Streamlit divider elements */
        .stMarkdown hr {
            border-color: #AAAAAA !important;
            background-color: #AAAAAA !important;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(45deg, #606060, #555555);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(45deg, #555555, #606060);
            box-shadow: 0 4px 12px rgba(42, 195, 202, 0.3);
            transform: translateY(-2px);
        }
        
        /* Primary button */
        .stButton > button[kind="primary"] {
            background: linear-gradient(45deg, #606060, #555555);
            border: 2px solid #606060;
        }
        
        /* Input fields */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select,
        .stNumberInput > div > div > input {
            background: rgba(42, 195, 202, 0.1);
            color: #ffffff;
            border: 1px solid #606060;
            border-radius: 6px;
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stSelectbox > div > div > select:focus,
        .stNumberInput > div > div > input:focus {
            border-color: #AAAAAA;
            box-shadow: 0 0 0 2px rgba(42, 195, 202, 0.3);
        }
        
        /* Selectbox dropdown styling */
        .stSelectbox [data-baseweb="select"] {
            background: rgba(42, 195, 202, 0.1);
            border: 1px solid #606060;
        }
        
        .stSelectbox [data-baseweb="select"] > div {
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid #606060;
        }
        
        /* Selectbox options */
        .stSelectbox [role="option"] {
            background: rgba(0, 0, 0, 0.9);
            color: #ffffff;
        }
        
        .stSelectbox [role="option"]:hover,
        .stSelectbox [role="option"][aria-selected="true"] {
            background: rgba(195, 195, 195, 0.3) !important;
            color: #ffffff !important;
        }
        
        /* Selectbox dropdown menu */
        .stSelectbox [data-baseweb="menu"] {
            background: rgba(0, 0, 0, 0.95);
            border: 1px solid #606060;
            border-radius: 6px;
        }
        
        /* File uploader */
        .stFileUploader > div {
            background: rgba(0, 0, 0, 0.8) !important;
            border: 2px dashed #606060;
            border-radius: 8px;
        }
        
        .stFileUploader label {
            color: #AAAAAA !important;
        }
        
        /* File uploader drag area */
        .stFileUploader [data-testid="stFileUploaderDropzone"] {
            background: rgba(0, 0, 0, 0.8) !important;
            border: 2px dashed #606060;
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
            border: 1px solid #606060;
            color: #AAAAAA;
            border-radius: 4px;
        }
        
        .stFileUploader button:hover {
            background: rgba(42, 195, 202, 0.2) !important;
            border-color: #AAAAAA !important;
            color: #ffffff !important;
        }
        
        .stFileUploader button:focus {
            background: rgba(42, 195, 202, 0.2) !important;
            border-color: #AAAAAA !important;
            color: #ffffff !important;
            box-shadow: 0 0 0 2px rgba(195, 195, 195, 0.3) !important;
        }
        
        /* Slider */
        .stSlider > div > div > div > div {
            background: #606060;
        }
        
        /* Slider track background (unfilled portion) */
        .stSlider [data-baseweb="slider"] > div > div {
            background: rgba(195, 195, 195, 0.3) !important;
            height: 4px !important;
        }
        
        /* Slider track fill (filled portion - left side) */
        .stSlider [data-baseweb="slider"] > div > div > div {
            background: #AAAAAA !important;
            height: 4px !important;
        }
        
        /* Slider thumb/handle */
        .stSlider [data-baseweb="slider"] [role="slider"] {
            background: #AAAAAA !important;
            border: 2px solid #AAAAAA !important;
            width: 16px !important;
            height: 16px !important;
        }
        
        /* Additional slider track styling */
        .stSlider [data-baseweb="slider"] [data-baseweb="slider-track"] {
            background: rgba(195, 195, 195, 0.3) !important;
        }
        
        .stSlider [data-baseweb="slider"] [data-baseweb="slider-track"] > div {
            background: #AAAAAA !important;
        }
        
        /* Tabs */
        button[data-baseweb="tab"] {
            background: transparent;
            color: #ffffff;
            border-bottom: 2px solid transparent;
        }
        
        button[data-baseweb="tab"][aria-selected="true"] {
            border-bottom-color: #AAAAAA;
            color: #AAAAAA;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: rgba(42, 195, 202, 0.15);
            border: 1px solid #606060;
            border-radius: 6px;
            color: #ffffff;
        }
        
        .streamlit-expanderContent {
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #606060;
            border-top: none;
            border-radius: 0 0 6px 6px;
        }
        
        /* Code blocks */
        .stCode {
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid #606060;
            border-radius: 6px;
        }
        
        /* Metrics */
        .metric-container {
            background: rgba(42, 195, 202, 0.1);
            border: 1px solid #606060;
            border-radius: 8px;
            padding: 1rem;
        }
        
        [data-testid="metric-container"] {
            background: rgba(42, 195, 202, 0.1);
            border: 1px solid #606060;
            border-radius: 8px;
            padding: 1rem;
        }
        
        /* Success/Info/Warning messages */
        .stSuccess {
            background: rgba(42, 195, 202, 0.2);
            border-left: 4px solid #606060;
            color: #ffffff;
        }
        
        .stInfo {
            background: rgba(42, 195, 202, 0.15);
            border-left: 4px solid #606060;
            color: #ffffff;
        }
        
        .stWarning {
            background: rgba(42, 195, 202, 0.15);
            border-left: 4px solid #606060;
            color: #ffffff;
        }
        
        .stError {
            background: rgba(42, 195, 202, 0.15);
            border-left: 4px solid #606060;
            color: #ffffff;
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background: #606060;
        }
        
        /* Divider */
        .stDivider > div {
            border-color: #AAAAAA;
        }
        
        /* Download button */
        .stDownloadButton > button {
            background: linear-gradient(45deg, #606060, #555555);
            color: white;
            border: 1px solid #606060;
            border-radius: 6px;
        }
        
        /* Spinner */
        .stSpinner > div {
            border-top-color: #AAAAAA;
        }
        
        /* Table styling */
        .stDataFrame {
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #606060;
            border-radius: 6px;
        }
        
        /* Data grid and data frame styling */
        .st-dg {
            border: 1px solid #606060;
            border-radius: 6px;
            background: rgba(0, 0, 0, 0.5);
        }
        
        .st-df {
            border: 1px solid #606060;
            border-radius: 6px;
            background: rgba(0, 0, 0, 0.5);
        }
        
        /* Sidebar header styling */
        .css-1544g2n {
            color: #AAAAAA;
            font-weight: bold;
        }
        
        /* st-cu component styling */
        .st-cu {
            background: rgba(42, 195, 202, 0.1);
            border: 1px solid #606060;
            border-radius: 6px;
            color: #ffffff;
        }
        
        .st-cu:hover {
            background: rgba(42, 195, 202, 0.2);
            border-color: #AAAAAA;
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
            color: #AAAAAA; 
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
            background: linear-gradient(45deg, #606060, #555555);
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
            border-left: 4px solid #606060;
            margin: 15px 0;
        }
        .config-preview {
            background: rgba(0, 0, 0, 0.8);
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 12px;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #606060;
            color: #ffffff;
        }
        
        /* Custom highlight colors */
        .highlight-red {
            color: #AAAAAA;
            font-weight: bold;
        }
        
        .highlight-orange {
            color: #AAAAAA;
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
        'facecolor': '#000000',
        'edgecolor': '#606060',
        'color': '#606060',
        'text_color': 'white',
        'threshold_color': '#606060',
        'mean_color': '#606060'
    }


def apply_complete_theme():
    """Apply the complete theme including all CSS and configurations."""
    apply_dark_theme()
    st.markdown(get_custom_css_classes(), unsafe_allow_html=True)
