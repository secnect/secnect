# secbert_model.py
import torch
import numpy as np
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import gc
from tqdm import tqdm

class SecBERTAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load SecBERT model
        model_name = "jackaduma/SecBERT"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        self.class_names = ["Non-login event", "Failed login", "Successful login"]
        self.explainer = LimeTextExplainer(class_names=self.class_names)
        
        # Login patterns
        self.success_patterns = [
            r"accepted\s+password", r"login\s+successful", 
            r"authentication\s+granted", r"sign[-_]in\s+successful",
            r"credentials\s+accepted", r"result\":\"SUCCESS\"",
            r"consolelogin\":\"success\"", r"status\":\"success\"",
            r"eventid=4624", r"access\s+granted",
            r"authentication\s+succeeded", r"user\s+logged\s+in"
        ]
        
        self.failure_patterns = [
            r"login\s+failed", r"authentication\s+failure",
            r"access\s+denied", r"sign[-_]in\s+denied",
            r"invalid\s+credentials", r"failed\s+password",
            r"result\":\"FAILURE\"", r"result\":\"DENIED\"",
            r"consolelogin\":\"failure\"", r"status\":\"failure\"",
            r"eventid=4625", r"authentication\s+unsuccessful",
            r"user\s+not\s+found", r"account\s+locked"
        ]
        
        self.success_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.success_patterns]
        self.failure_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.failure_patterns]
    
    def _free_memory(self):
        torch.cuda.empty_cache()
        gc.collect()
    
    def is_login_event(self, log_line):
        """Check if the log line contains any login-related keywords"""
        lower_log = log_line.lower()
        login_keywords = [
            'login', 'logon', 'authenticat', 'signin', 'sign-in',
            'session', 'access', 'credential', 'password', 'auth'
        ]
        return any(keyword in lower_log for keyword in login_keywords)
    
    def predict_proba(self, texts):
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**encodings)
                probabilities = torch.softmax(outputs.logits, dim=1)
            
            return probabilities.cpu().numpy()
        except RuntimeError as e:
            print(f"Prediction error: {str(e)}")
            return np.array([[1.0, 0.0, 0.0]])  # Default to non-login event
    
    def analyze_logs(self, log_lines, confidence_threshold=0.6):
        results = []
        
        for log in tqdm(log_lines, desc="Analyzing logs"):
            try:
                # First check for clear patterns
                for pattern in self.success_regex:
                    if pattern.search(log):
                        results.append({
                            "original_log_line": log,
                            "normalized_log": self.normalize_text(log),
                            "prediction": "Successful login",
                            "max_similarity_score": 1.0,
                            "explanation": f"Matched success pattern: '{pattern.pattern}'",
                            "model": "SecBERT"
                        })
                        break
                else:  # No break occurred, no success pattern matched
                    for pattern in self.failure_regex:
                        if pattern.search(log):
                            results.append({
                                "original_log_line": log,
                                "normalized_log": self.normalize_text(log),
                                "prediction": "Failed login",
                                "max_similarity_score": 1.0,
                                "explanation": f"Matched failure pattern: '{pattern.pattern}'",
                                "model": "SecBERT"
                            })
                            break
                    else:  # No pattern matched
                        if not self.is_login_event(log):
                            results.append({
                                "original_log_line": log,
                                "normalized_log": self.normalize_text(log),
                                "prediction": "Non-login event",
                                "max_similarity_score": 0.0,
                                "explanation": "No login-related keywords found",
                                "model": "SecBERT"
                            })
                        else:
                            # Use SecBERT for classification
                            probabilities = self.predict_proba(log)[0]
                            predicted_class = np.argmax(probabilities)
                            confidence = probabilities[predicted_class]
                            
                            if confidence < confidence_threshold:
                                prediction = "Ambiguous event"
                            else:
                                prediction = self.class_names[predicted_class]
                            
                            results.append({
                                "original_log_line": log,
                                "normalized_log": self.normalize_text(log),
                                "prediction": prediction,
                                "max_similarity_score": float(confidence),
                                "explanation": f"SecBERT classification with {confidence:.2f} confidence",
                                "model": "SecBERT"
                            })
            except Exception as e:
                results.append({
                    "original_log_line": log,
                    "normalized_log": self.normalize_text(log),
                    "prediction": "Error",
                    "max_similarity_score": 0.0,
                    "explanation": f"Processing error: {str(e)}",
                    "model": "SecBERT"
                })
        
        return pd.DataFrame(results)

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text similar to the original model_utils implementation"""
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

# Cache the model instance
def get_secbert_analyzer():
    return SecBERTAnalyzer()