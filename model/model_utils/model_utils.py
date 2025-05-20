import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st  # only for @st.cache_resource / @st.cache_data decorators
from sentence_transformers import util
from sklearn.feature_extraction.text import TfidfVectorizer
from Levenshtein import ratio as levenshtein_ratio

@st.cache_resource
def load_model(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    return SentenceTransformer(model_name)

def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    # strip timestamps
    text = re.sub(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}', '', text)
    text = re.sub(r'\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}:\d{2}', '', text)
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', text)
    text = re.sub(r'\d{2}:\d{2}:\d{2}', '', text)
    # strip IPs, ports, standalone numbers
    text = re.sub(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', '', text)
    text = re.sub(r':\d{1,5}\b', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    return ' '.join(text.split())

@st.cache_data
def load_positive_examples(csv_path: str = 'data/sample-logs/failed_login_logs.csv') -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"{csv_path} not found.")
        return pd.DataFrame(columns=['Log', 'normalized_log'])
    df['normalized_log'] = df['Log'].apply(normalize_text)
    return df

def compute_similarities(model: SentenceTransformer, positive_texts: list[str], target_texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (max_similarities, argmax_indices)."""
    pos_emb = model.encode(positive_texts)
    tgt_emb = model.encode(target_texts)
    sims = cosine_similarity(tgt_emb, pos_emb)
    max_sim = np.max(sims, axis=1)
    argmax = np.argmax(sims, axis=1)
    return max_sim, argmax

def build_results_df(original_lines: list[str], normalized_lines: list[str], max_sim: np.ndarray, argmax: np.ndarray, positive_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        'original_log_line': original_lines,
        'normalized_log_line': normalized_lines,
        'max_similarity_score': max_sim,
        'most_similar_positive_idx': argmax,
        'most_similar_positive_example': [
            positive_df.iloc[i]['Log'] for i in argmax
        ]
    })
    
    return df.sort_values('max_similarity_score', ascending=False)
#
# Na tomhle ještě zapracovat
#
def get_similarity_breakdown(model, log_line, positive_example):
    """Return a dictionary explaining similarity components"""
    tokens1 = set(normalize_text(log_line).split())
    tokens2 = set(normalize_text(positive_example).split())
    
    common_tokens = tokens1 & tokens2
    unique_to_log = tokens1 - tokens2
    unique_to_example = tokens2 - tokens1
    
    # Get embeddings for important phrases
    important_phrases = list(common_tokens) + [
        ' '.join(pair) for pair in zip(
            log_line.split()[:-1], 
            log_line.split()[1:]
        ) if ' '.join(pair) in positive_example
    ]
    
    return {
        'common_tokens': common_tokens,
        'unique_to_log': unique_to_log,
        'unique_to_example': unique_to_example,
        'important_phrases': important_phrases
    }
    
#
#
#



def enhanced_similarity(text1, text2):
    # Semantic similarity
    model = load_model()
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    semantic_sim = util.cos_sim(emb1, emb2)
    
    # Lexical similarity
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    lexical_sim = (tfidf * tfidf.T).A[0,1]
    
    # String similarity
    string_sim = levenshtein_ratio(text1, text2)
    
    return {
        'semantic': semantic_sim,
        'lexical': lexical_sim,
        'string': string_sim,
        'combined': 0.5*semantic_sim + 0.3*lexical_sim + 0.2*string_sim
    }
    
    

def extract_log_fields(log_line):
    """Identify common log components"""
    patterns = {
        'timestamp': r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}',
        'ip': r'\b\d{1,3}(?:\.\d{1,3}){3}\b',
        'user': r'user\s*[=:]\s*[\w\-]+',
        'status': r'status\s*[=:]\s*\d{3}',
        'method': r'(GET|POST|PUT|DELETE|HEAD)',
    }
    
    fields = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, log_line)
        if match:
            fields[field] = match.group()
    
    return fields

def highlight_text(text: str, keywords: set) -> str:
    """
    Highlight matching keywords in text with HTML markup
    Args:
        text: Original text to highlight
        keywords: Set of keywords/phrases to highlight
    Returns:
        HTML string with highlighted portions
    """
    if not keywords:
        return text
    
    # Sort keywords by length (longest first) to handle multi-word phrases
    sorted_keywords = sorted(keywords, key=len, reverse=True)
    
    # Create a copy to modify
    highlighted = text
    
    # Highlight each keyword
    for keyword in sorted_keywords:
        if keyword in highlighted:
            highlighted = highlighted.replace(
                keyword,
                f'<span style="background-color: #FFFF00; font-weight: bold;">{keyword}</span>'
            )
    
    return highlighted
