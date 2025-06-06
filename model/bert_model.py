import torch
import numpy as np
import pandas as pd
import re
import string
import json
import os
from transformers import AutoTokenizer, AutoModel
from lime.lime_text import LimeTextExplainer
import gc
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

class DynamicBERTAnalyzer:
    def __init__(self, config_path=None, config_override=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load configuration from file or use defaults
        self.config = self._load_config(config_path, config_override)
        
        # Load BERT model dynamically
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        self.model = AutoModel.from_pretrained(self.config['model_name']).to(self.device)
        
        self.class_names = self.config['class_names']
        self.explainer = LimeTextExplainer(class_names=self.class_names)
        
        # Dynamic prototype generation
        self.semantic_prototypes = self._generate_prototypes()
        self._compute_prototype_embeddings()
    
    def _load_config(self, config_path=None, config_override=None):
        """Load configuration from JSON file with fallback to defaults"""
        # Start with default configuration
        config = self._get_default_config()
        
        # Try to load from file
        if config_path is None:
            config_path = "config.json"  # Default config file name
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    # Deep merge file config with defaults
                    config = self._deep_merge_config(config, file_config)
                    print(f"Configuration loaded from: {config_path}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
                print("Using default configuration")
        else:
            print(f"Config file {config_path} not found. Using default configuration")
        
        # Apply any runtime overrides
        if config_override:
            config = self._deep_merge_config(config, config_override)
            print("Applied runtime configuration overrides")
        
        return config
    
    def _deep_merge_config(self, base_config, override_config):
        """Deep merge two configuration dictionaries"""
        result = base_config.copy()
        
        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_config(result[key], value)
            else:
                result[key] = value
        
        return result
    def _get_default_config(self):
        """Default configuration - fallback when config.json is not available"""
        return {
            'model_name': 'bert-base-uncased',
            'class_names': ["Non-login event", "Failed login", "Successful login"],
            'embedding_dim': 768,
            'max_length': 128,
            'thresholds': {
                'login_detection': 0.15,
                'classification': 0.3,
                'high_confidence': 0.6,
                'medium_confidence': 0.4,
                'model_confidence': 0.8
            },
            'prototype_categories': ['successful_login', 'failed_login', 'non_login'],
            'prototype_templates': {
                'successful_login': ['authenticated', 'accepted', 'successful', 'verified', 'granted', 'established'],
                'failed_login': ['failed', 'denied', 'invalid', 'incorrect', 'error', 'failure'],
                'non_login': ['started', 'running', 'established', 'completed', 'normal', 'transfer']
            },
            'normalization_patterns': [
                (r'\d+', ''),
                (r'[{}[\]":,]', ' '),
                (r'(?<=[a-zA-Z])(?=[A-Z])', ' ')
            ],
            'term_normalizations': {
                r'\bssh\w*': 'ssh',
                r'\blogin\w*': 'login', 
                r'\bauth\w*': 'authentication',
                r'\bfail\w*': 'failed',
                r'\bsuccess\w*': 'successful'
            }
        }
    
    def _generate_prototypes(self):
        """Dynamically generate semantic prototypes from templates"""
        prototypes = {}
        templates = self.config['prototype_templates']
        
        for category in self.config['prototype_categories']:
            if category in templates:
                base_terms = templates[category]
                prototypes[category] = []
                
                # Generate combinations dynamically
                if 'login' in category:
                    login_terms = ['user', 'login', 'authentication', 'password', 'access', 'session']
                    for base in base_terms:
                        for login_term in login_terms:
                            prototypes[category].append(f"{login_term} {base}")
                else:
                    system_terms = ['service', 'process', 'connection', 'operation', 'system', 'file']
                    for base in base_terms:
                        for sys_term in system_terms:
                            prototypes[category].append(f"{sys_term} {base}")
                
                # Limit to avoid over-representation
                prototypes[category] = prototypes[category][:7]
        
        return prototypes
    
    def _compute_prototype_embeddings(self):
        """Pre-compute embeddings for semantic prototypes"""
        self.prototype_embeddings = {}
        for category, prototypes in self.semantic_prototypes.items():
            embeddings = [self._get_embedding(p, preprocess=True) for p in prototypes]
            self.prototype_embeddings[category] = np.mean(embeddings, axis=0)
    
    def _preprocess(self, text):
        """Dynamic preprocessing based on configuration"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Apply normalization patterns from config
        for pattern, replacement in self.config['normalization_patterns']:
            text = re.sub(pattern, replacement, text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Apply dynamic timestamp/identifier removal
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}Z?',
            r'\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}:\d{2}',
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{2}:\d{2}:\d{2}',
            r'\b\d{1,3}(?:\.\d{1,3}){3}\b',  # IP addresses
            r':\d{1,5}\b',  # Ports
            r'\b\d+\b'  # Standalone numbers
        ]
        
        for pattern in timestamp_patterns:
            text = re.sub(pattern, '', text)
        
        # Apply term normalizations from config
        for pattern, replacement in self.config['term_normalizations'].items():
            text = re.sub(pattern, replacement, text)
        
        return ' '.join(text.split())

    def _get_embedding(self, text, preprocess=False):
        """Extract BERT embedding with dynamic configuration"""
        try:
            if preprocess:
                text = self._preprocess(text)
            
            if not text.strip():
                return np.zeros(self.config['embedding_dim'])
            
            tokens = self.tokenizer(text, padding=True, truncation=True, 
                                  max_length=self.config['max_length'], 
                                  return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**tokens)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embedding.flatten()
        except Exception as e:
            print(f"Embedding error: {str(e)}")
            return np.zeros(self.config['embedding_dim'])
    
    def _compute_similarities(self, log_embedding):
        """Compute cosine similarities with prototypes"""
        similarities = {}
        for category, prototype_embedding in self.prototype_embeddings.items():
            if np.allclose(log_embedding, 0) or np.allclose(prototype_embedding, 0):
                similarities[category] = 0.0
            else:
                sim = cosine_similarity(log_embedding.reshape(1, -1), 
                                      prototype_embedding.reshape(1, -1))[0][0]
                similarities[category] = sim
        return similarities
    
    def _is_login_related(self, log_line):
        """Dynamic login detection using configurable threshold"""
        embedding = self._get_embedding(log_line, preprocess=True)
        similarities = self._compute_similarities(embedding)
        
        login_categories = [cat for cat in similarities.keys() if 'login' in cat]
        non_login_categories = [cat for cat in similarities.keys() if 'login' not in cat]
        
        login_sim = max([similarities[cat] for cat in login_categories]) if login_categories else 0
        non_login_sim = max([similarities[cat] for cat in non_login_categories]) if non_login_categories else 0
        
        threshold = self.config['thresholds']['login_detection']
        return login_sim > non_login_sim and login_sim > threshold
    
    def _classify_semantically(self, log_line):
        """Dynamic semantic classification using configurable thresholds"""
        embedding = self._get_embedding(log_line, preprocess=True)
        similarities = self._compute_similarities(embedding)
        
        best_category = max(similarities, key=similarities.get)
        best_score = similarities[best_category]
        
        # Dynamic category mapping
        category_map = {}
        for i, category in enumerate(self.config['prototype_categories']):
            category_map[category] = self.class_names[i] if i < len(self.class_names) else "Unknown"
        
        prediction = category_map.get(best_category, "Unknown")
        threshold = self.config['thresholds']['classification']
        
        if best_score < threshold:
            prediction = "Ambiguous event"
        
        explanation = f"Semantic similarity: {best_score:.3f} to {best_category}"
        return prediction, best_score, explanation
    
    def predict_proba(self, texts):
        """Dynamic probability simulation based on similarities"""
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        for text in texts:
            embedding = self._get_embedding(text, preprocess=True)
            similarities = self._compute_similarities(embedding)
            
            # Map similarities to class order
            scores = []
            for class_name in self.class_names:
                # Find corresponding category
                category = None
                for cat in self.config['prototype_categories']:
                    if any(word in class_name.lower() for word in cat.split('_')):
                        category = cat
                        break
                scores.append(similarities.get(category, 0.0))
            
            # Normalize to probabilities
            exp_scores = np.exp(np.array(scores) * 3)
            probabilities = exp_scores / np.sum(exp_scores)
            results.append(probabilities)
        
        return np.array(results)
    
    def _combine_predictions(self, semantic_pred, semantic_conf, semantic_exp, 
                           model_prediction, model_confidence):
        """Dynamic prediction combination using configurable thresholds"""
        thresholds = self.config['thresholds']
        
        if semantic_conf > thresholds['high_confidence']:
            return semantic_pred, semantic_conf, f"{semantic_exp} (high confidence)"
        elif (semantic_conf > thresholds['medium_confidence'] and 
              semantic_pred == model_prediction):
            return (semantic_pred, max(semantic_conf, model_confidence),
                   f"Agreement: semantic {semantic_conf:.3f}, model {model_confidence:.3f}")
        elif model_confidence > thresholds['model_confidence']:
            return (model_prediction, model_confidence,
                   f"Model confidence: {model_confidence:.3f}")
        else:
            return ("Ambiguous event", max(semantic_conf, model_confidence),
                   f"Low confidence: {semantic_conf:.3f}/{model_confidence:.3f}")
    
    def analyze_logs(self, log_lines, confidence_threshold=0.5):
        """Dynamic log analysis with configurable behavior"""
        results = []
        
        for log in tqdm(log_lines, desc="Analyzing logs"):
            try:
                if not self._is_login_related(log):
                    results.append({
                        "original_log_line": log,
                        "normalized_log": self.normalize_text(log),
                        "prediction": "Non-login event",
                        "max_similarity_score": 0.0,
                        "explanation": "Semantically not login-related",
                        "model": "BERT"
                    })
                else:
                    # Primary classification via semantic similarity
                    semantic_pred, semantic_conf, semantic_exp = self._classify_semantically(log)
                    
                    # Secondary validation via probability estimation
                    probabilities = self.predict_proba(log)[0]
                    model_confidence = np.max(probabilities)
                    model_prediction = self.class_names[np.argmax(probabilities)]
                    
                    # Dynamic prediction combination
                    final_prediction, final_confidence, explanation = self._combine_predictions(
                        semantic_pred, semantic_conf, semantic_exp, 
                        model_prediction, model_confidence
                    )
                    
                    results.append({
                        "original_log_line": log,
                        "normalized_log": self.normalize_text(log),
                        "prediction": final_prediction,
                        "max_similarity_score": float(final_confidence),
                        "explanation": explanation,
                        "model": "BERT"
                    })
                    
            except Exception as e:
                results.append({
                    "original_log_line": log,
                    "normalized_log": self.normalize_text(log),
                    "prediction": "Error",
                    "max_similarity_score": 0.0,
                    "explanation": f"Processing error: {str(e)}",
                    "model": "BERT"
                })
        
        return pd.DataFrame(results)

    def normalize_text(self, text: str) -> str:
        """Dynamic text normalization using preprocessing pipeline"""
        return self._preprocess(text)

    def _free_memory(self):
        torch.cuda.empty_cache()
        gc.collect()

# Dynamic factory function
def get_secbert_analyzer(config_path=None, config_override=None):
    """
    Create analyzer instance with configuration loaded from JSON file
    
    Args:
        config_path: Path to JSON config file (default: "config.json")
        config_override: Dict to override specific config values at runtime
    
    Returns:
        DynamicBERTAnalyzer instance
    """
    return DynamicBERTAnalyzer(config_path, config_override)