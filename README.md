# Phishing Attack Lab: Adversarial AI Robustness Evaluation

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Bootstrap](https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)

An interactive cybersecurity research platform built to demonstrate and defend against **Adversarial Machine Learning** attacks in email phishing detection.

## Overview

This project simulates a "Cat-and-Mouse" game between a sophisticated attacker and a machine learning defender. It highlights how standard AI models can be "tricked" by professional language and how **Adversarial Training** can harden these models.

### The "Two-Model" Side-by-Side Analysis:
1. **Victim Model (Pre-Attack):** A baseline Random Forest classifier. High accuracy on "obvious" phishing but vulnerable to **Professional Padding** (long, clean text).
2. **Defender Model (Post-Attack):** A hardened model "vaccinated" against stealthy attacks. It prioritizes **Security-Relevant Features** over linguistic style.

---

## Technical Stack

- **Backend:** FastAPI (Python 3.10+)
- **Frontend:** Jinja2 Templates & Bootstrap 5
- **Machine Learning:** Scikit-Learn (Random Forest)
- **NLP & Heuristics:** - `NLTK`: Tokenization and Stop-word analysis.
  - `pyspellchecker`: Lexical correctness scoring.
  - `Regex`: URL and Domain extraction.

---

## The "Chameleon" Attack Concept

The core of this lab is demonstrating the **Bypass**. 
By crafting a "Chameleon Email"—one that is long, has zero spelling errors, and uses a formal academic tone—we can lower the **Victim Model's** risk score while the **Defender Model** identifies the underlying malicious intent based on structural patterns and domain anomalies.



---

## Feature Extraction Logic

Dataset from Kaggle, ethancratchley/email-phishing-dataset, was used.
Each email is processed into a numerical vector of 8 key features:
* `num_words`: Total word count (detects "Padding").
* `num_unique_words`: Lexical diversity.
* `num_stopwords`: Measures natural language "flow."
* `num_links`: Count of embedded URLs.
* `num_unique_domains`: Identifies cross-site redirection.
* `num_email_addresses`: Presence of contact leads.
* `num_spelling_errors`: Uses frequency-based dictionaries.
* `num_urgent_keywords`: Tracks panic-inducing triggers (e.g., "Required", "Suspended").

---
