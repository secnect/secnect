# 🔐 Login Event Detector

A Streamlit application that utilizes semantic similarity of pre-trained SecBERT for recognition of Login Event type Security logs

![Demo](assets/demo.gif)

project-root/
├── Data/
│   ├── created-logs/                # Output of giga_dataset_gen.py for SecBERT fine-tuning
│   ├── raw-logs/                    # Raw logs from LogHub (Linux, SSH)
│   ├── sample-logs/                 # 100 examples of login events (failed/success)
│   ├── train-logs/                  # Used during testing of previous BERT model iterations
│   └── log_preprocessing.ipynb
│
├── Feedback/
│   └──                              # (Empty or to be populated)
│
├── Model/
│   ├── annotated_logs.txt
│   ├── BERT_clayrity.ipynb
│   ├── SecBERT_puvodni.ipynb
│   ├── SecBERT_test_lepsi.ipynb
│   └── model_utils/
│       ├── giga_dataset_gen.py
│       ├── model_utils.py
│       └── secbert_model.py
│
├── enhanced_dataset_generator.py
├── feature_bert - experiment.py
├── pokus_NER_LSTM_funny_mvp.py
├── README.md
├── requirements.txt                # Added tf-keras due to recent package update
└── streamlit_app.py




- Data
    - created-logs (output file of giga_dataset_gen.py for SecBERT finetuning)
    - raw-logs (raw form of logs from LogHub repository, features Linux, SSH logs)
    - sample-logs (consists of 100 examples of login events (failed/success))
    - train-logs (folder that was used during testing the past iteration of BERT models)
    - log_preprocessing.ipynb (self explanatory)

- Feedback
    -
    -

- Model
    - annotated_logs.txt
    - BERT_clayrity.ipynb
    - SecBERT_puvodni.ipynb
    - SecBERT_test_lepsi.ipynb
    - model_utils
        - giga_dataset_gen.py
        - model_utils.py
        - secbert_model.py

- enhanced_dataset_generator.py 
- feature_bert - experiment.py
- pokus_NER_LSTM_funny_mvp.py
- README.md
- requirements.txt (musel jsem addnout tf-keras protože teď updatnuli package a rozmrdalo se to)
- streamlit_app.py




## 🌟 Features

- **Semantic Analysis**: Uses Sentence-BERT (all-MiniLM-L6-v2) model for semantic similarity
- **Text Normalization**: Automatically removes timestamps, IP addresses, and numbers
- **Interactive UI**: User-friendly interface with real-time analysis
- **Configurable Thresholds**: Adjust confidence levels for detection
- **Visual Analytics**: Distribution plots and threshold analysis
- **Export Options**: Download full results or high-confidence matches only

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/failed-login-detector.git
cd failed-login-detector
