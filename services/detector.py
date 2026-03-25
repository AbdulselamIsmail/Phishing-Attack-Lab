import joblib
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from urllib.parse import urlparse
from spellchecker import SpellChecker

current_dir = os.path.dirname(os.path.realpath(__file__))

# Path to models
model_path = os.path.join(current_dir, "..", "aimodels")

pre_attack_model = joblib.load(os.path.join(model_path, "pre_attack_model.joblib"))
post_attack_model = joblib.load(os.path.join(model_path, "post_attack_model.joblib"))
scaler = joblib.load(os.path.join(model_path, "scaler.joblib"))

# NLTK setup
data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(data_path):
    os.makedirs(data_path)

nltk.data.path.append(data_path)
nltk.download('punkt_tab', download_dir=data_path, quiet=True)
nltk.download('stopwords', download_dir=data_path, quiet=True)

# Regex Patterns
url_pattern = r'(https?://[^\s<>"]+|www\.[^\s<>"]+\.[^\s<>"]+)'
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}'

URGENT_KEYWORDS = {
    'urgent', 'suspended', 'locked', 'action', 
    'required', 'invoice', 'billing',"immediately",
    "deadline", "important", "warning", "expired",
    "unauthorized", "activity", "confirm",
    "payment", "refund", "transaction", "bank", "statement", "overdue",
    "password", "reset", "login", "access", "credentials", "support"
}

spell = SpellChecker()

def checkEMail(emailText: str, model_type: str):
    data = extractData(emailText)
    
    # Updated feature list (15 features)
    feature_cols = [
        'num_words', 'num_unique_words', 'num_stopwords', 'num_links', 
        'num_unique_domains', 'num_email_addresses', 'num_spelling_errors', 
        'num_urgent_keywords', 'link_density', 'unique_word_density', 
        'stopwords_ratio', 'unique_domain_ratio', 'has_email_addresses', 
        'spelling_error_ratio'
    ]
    
    new_data_raw = pd.DataFrame([data], columns=feature_cols)
    new_data_scaled = scaler.transform(new_data_raw) 
    
    if model_type == "pre-attack":
        return pre_attack_model.predict_proba(new_data_scaled)[0][1]
    elif model_type == "post-attack":
        return post_attack_model.predict_proba(new_data_scaled)[0][1]
    return "False Input"

def extractData(text_input: str):
    inputLower = text_input.lower()
    
    # Base Counts
    words_list = inputLower.split()
    n_words = len(words_list)
    n_unique = len(set(words_list))
    n_stop = get_num_stopwords(inputLower)
    n_links = get_num_links(inputLower)
    n_domains = get_unique_domains(inputLower)
    n_emails = get_num_emails(inputLower)
    n_errors = get_num_spelling_errors(inputLower)
    n_urgent = get_num_urgent(inputLower)
    
    # Safety denominator
    denom = n_words if n_words > 0 else 1
    
    return {
        'num_words': n_words,
        'num_unique_words': n_unique,
        'num_stopwords': n_stop,
        'num_links': n_links,
        'num_unique_domains': n_domains,
        'num_email_addresses': n_emails,
        'num_spelling_errors': n_errors,
        'num_urgent_keywords': n_urgent,
        
        # New Engineered Features
        'link_density': n_links / denom,
        'unique_word_density': n_unique / denom,
        'stopwords_ratio': n_stop / denom,
        'unique_domain_ratio': n_domains / (n_links if n_links > 0 else 1),
        'has_email_addresses': 1 if n_emails > 0 else 0,
        'spelling_error_ratio': n_errors / denom
    }

def get_num_stopwords(text: str):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)  
    # Fixed w.lower() call
    return len([w for w in words if w.lower() in stop_words])

def get_num_links(text: str):
    return len(re.findall(url_pattern, text))

def get_unique_domains(text: str):
    links = re.findall(url_pattern, text, re.IGNORECASE)
    unique_domains = set()
    for link in links:
        if not link.startswith(('http://', 'https://')):
            link = 'http://' + link
        domain = urlparse(link).netloc
        if domain:
            unique_domains.add(domain.lower())
    return len(unique_domains)

def get_num_emails(text: str):
    return len(re.findall(email_pattern, text))

def get_num_urgent(text: str):
    words = text.lower().split()
    count = 0
    for word in words:
        if any(keyword in word for keyword in URGENT_KEYWORDS):
            count += 1
    return count

def get_num_spelling_errors(text: str):
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = clean_text.split()
    return len(spell.unknown(words))