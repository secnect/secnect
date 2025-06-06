import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from sentence_transformers import util
from sklearn.feature_extraction.text import TfidfVectorizer
from Levenshtein import ratio as levenshtein_ratio
from model.bert_model import get_secbert_analyzer  # Import the new SecBERT analyzer

@st.cache_resource
def load_model(model_name: str = 'bert-base-uncased') -> SentenceTransformer:
    if model_name.lower() == 'secbert':
        return get_secbert_analyzer()
    return SentenceTransformer(model_name)

def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    # strip timestamps
    text = re.sub(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}Z?', '', text)
    text = re.sub(r'\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}:\d{2}', '', text)
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', text)
    text = re.sub(r'\d{2}:\d{2}:\d{2}', '', text)

    # Remove IP addresses and ports
    text = re.sub(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', '', text)
    text = re.sub(r':\d{1,5}\b', '', text)

    # Remove standalone numbers
    text = re.sub(r'\b\d+\b', '', text)

    # Remove braces, brackets, quotes, colons, commas
    text = re.sub(r'[{}[\]":,]', ' ', text)
    text = re.sub(r'(?<=[a-zA-Z])(?=[A-Z])', ' ', text)
    text = text.lower()
    # Collapse multiple spaces and strip
    return ' '.join(text.split())

@st.cache_data
def load_positive_examples(csv_path: str = 'data/sample-logs/log_samples.csv') -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"{csv_path} not found.")
        return pd.DataFrame(columns=['Log', 'normalized_log'])
    df['normalized_log'] = df['Log'].apply(normalize_text)
    return df

def compute_similarities(model: SentenceTransformer, positive_texts: list[str], target_texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (max_similarities, argmax_indices)."""
    if isinstance(model, SentenceTransformer):
        # Original similarity computation
        pos_emb = model.encode(positive_texts)
        tgt_emb = model.encode(target_texts)
        sims = cosine_similarity(tgt_emb, pos_emb)
        max_sim = np.max(sims, axis=1)
        argmax = np.argmax(sims, axis=1)
        return max_sim, argmax
    else:
        return np.zeros(len(target_texts)), np.zeros(len(target_texts))

def build_results_df(original_lines: list[str], normalized_lines: list[str], max_sim: np.ndarray, argmax: np.ndarray, positive_df: pd.DataFrame) -> pd.DataFrame:
    """Build results dataframe, handling both similarity and SecBERT cases"""
    if isinstance(positive_df, str) and positive_df == "SecBERT":
        # For SecBERT, the input is already a DataFrame with analysis results
        return normalized_lines  # In SecBERT case, normalized_lines is actually the results_df
    
    # Original similarity-based processing
    df = pd.DataFrame({
        'original_log_line': original_lines,
        'normalized_log_line': normalized_lines,
        'max_similarity_score': max_sim,
        'most_similar_positive_idx': argmax,
        'most_similar_positive_example': [
            positive_df.iloc[i]['Log'] for i in argmax
        ],
        'prediction': 'Failed login',  # Default for similarity model
        'model': 'SentenceTransformer'
    })
    
    return df.sort_values('max_similarity_score', ascending=False)

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
    
def enhanced_similarity(text1, text2):
    
    model = load_model()
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    semantic_sim = util.cos_sim(emb1, emb2)
    

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    lexical_sim = (tfidf * tfidf.T).A[0,1]
    
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
    
    if not keywords:
        return text
    sorted_keywords = sorted(keywords, key=len, reverse=True)
    
    highlighted = text
    
    for keyword in sorted_keywords:
        if keyword in highlighted:
            highlighted = highlighted.replace(
                keyword,
                f'<span style="background-color: #FFFF00; font-weight: bold;">{keyword}</span>'
            )
    
    return highlighted