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

pre_attack_model = joblib.load(os.path.join(current_dir,"..","aimodels","pre_attack_model.joblib"))
post_attack_model = joblib.load(os.path.join(current_dir,"..","aimodels","post_attack_model.joblib"))
scaler = joblib.load(os.path.join(current_dir,"..","aimodels","scaler.joblib"))


# NLTK stop words list
# Define a path inside your project folder to keep things portable
data_path = os.path.join(os.getcwd(), "nltk_data")

if not os.path.exists(data_path):
    os.makedirs(data_path)

# Tell NLTK to look in this new folder
nltk.data.path.append(data_path)

nltk.download('punkt_tab', download_dir=data_path)
nltk.download('stopwords', download_dir=data_path)


# This pattern looks for:
# 1. http or https (optional)
# 2. :// (optional if www is present)
# 3. Domain name and TLD (like .com, .net, .tr)
# 4. Optional path/query parameters
url_pattern = r'(https?://[^\s<>"]+|www\.[^\s<>"]+\.[^\s<>"]+)'

# This pattern looks for:
# 1. Alphanumeric characters, dots, underscores, or hyphens
# 2. The '@' symbol
# 3. A domain name (alphanumeric/hyphen)
# 4. A dot and a TLD (2 to 6 characters like .com, .edu, .tr)
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}'

# Urgent keyword list for detection
URGENT_KEYWORDS = {
    'urgent', 'suspended', 'locked', 'action', 
    'required', 'invoice', 'billing',"immediately",
    "deadline", "important", "warning", "expired",
    "unauthorized", "activity", "confirm", "suspended", "locked",
    "payment", "refund", "transaction", "bank", "statement", "overdue",
    "password", "reset", "login", "access", "credentials", "support"
}

# model that checks for spelling errors
spell = SpellChecker()

# Checks if email is Spam or Ham
def checkEMail(emailText : str, model: str):
    data = extractData(emailText)
    feature_cols = ['num_words', 'num_unique_words', 'num_stopwords', 'num_links', 
                'num_unique_domains', 'num_email_addresses', 'num_spelling_errors', 'num_urgent_keywords']
    

    new_data_raw = pd.DataFrame([data],columns=feature_cols)
    
    # Scale the new data using the loaded scaler
    new_data_scaled = scaler.transform(new_data_raw) 
    
    if model == "pre-attack":
        pre_model_prediction = pre_attack_model.predict_proba(new_data_scaled)[0][1]
        return pre_model_prediction
    elif model == "post-attack":
        post_model_prediction = post_attack_model.predict_proba(new_data_scaled)[0][1]
        return post_model_prediction
    else:
        return "False Input"
    
    
    
    
def extractData(input: str):
    inputLower = input.lower()
    
    # Extracting data information from string
    
    num_words = len(inputLower.split())
    
    num_unique_words = len(set(inputLower.split()))
    
    num_stopwords = get_num_stopwords(inputLower)
    
    num_links = get_num_links(inputLower)
    
    num_unique_domains = get_unique_domains(inputLower)
    
    num_email_addresses = get_num_emails(inputLower)
    
    num_spelling_errors = get_num_spelling_errors(inputLower)
    
    num_urgent_keywords = get_num_urgent(inputLower)
    
    return {
        "num_words" : num_words, 
        "num_unique_words" : num_unique_words, 
        "num_stopwords" : num_stopwords, 
        "num_links" : num_links, 
        "num_unique_domains" : num_unique_domains, 
        "num_email_addresses" : num_email_addresses, 
        "num_spelling_errors" : num_spelling_errors,
        "num_urgent_keywords" : num_urgent_keywords
    }
        
    

    
    
    
# Stop words extraction using nltk library    
def get_num_stopwords(text : str):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)  
    return len([w for w in words if w.lower not in stop_words])
        
# Link count extraction using regex
def get_num_links(text : str):
    links = re.findall(url_pattern, text)
    return len(links)


# Unique domain name extraction using regex and set
def get_unique_domains(text : str):
    # 1. Regex to find all links
    links = re.findall(url_pattern, text, re.IGNORECASE)
    
    # 2. Extract the domain (netloc) from each link
    unique_domains = set()
    for link in links:
        # Add 'http://' prefix if it's missing (e.g., for 'www.test.com') 
        # so urlparse can read it correctly
        if not link.startswith(('http://', 'https://')):
            link = 'http://' + link
            
        domain = urlparse(link).netloc
        if domain:
            unique_domains.add(domain.lower())
            
    return len(unique_domains)
 
 
# Email adress extraction using regex
def get_num_emails(text : str):
    emails = re.findall(email_pattern, text)
    return len(emails)


# Urgent word count extraction using dictionary
def get_num_urgent(text : str):
    text = text.lower()
    count = 0
    words = text.split()
    for word in words:
        # Check if any red flag is a substring of the current word
        if any(keyword in word for keyword in URGENT_KEYWORDS):
            count += 1
    return count


def get_num_spelling_errors(text : str):
    # 1. Clean the text: Remove numbers and special characters 
    # so they aren't counted as "misspellings"
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = clean_text.split()
    
    # 2. Find which words are misspelled
    # .unknown() returns a set of words not found in its dictionary
    misspelled = spell.unknown(words)
    
    return len(misspelled)