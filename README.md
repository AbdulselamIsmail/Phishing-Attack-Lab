# Phishing Attack Lab: Adversarial AI Robustness Evaluation

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Bootstrap](https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)

An interactive cybersecurity research platform designed to evaluate and enhance the robustness of **Machine Learning (ML)** models against sophisticated "Chameleon" phishing attacks.

## Project Overview

This project simulates a "Cat-and-Mouse" game between a sophisticated attacker and an AI defender. It demonstrates how standard filters can be bypassed using **Adversarial Machine Learning** techniques and how proactive "Defender" models can be hardened to recognize hidden threats.

### The "Two-Model" Architecture:
1. **Victim Model (Baseline):** A standard Random Forest classifier. Highly effective at catching "obvious" spam but vulnerable to **Professional Padding** and lexical masking.
2. **Defender Model (Hardened):** A robust model developed through **Adversarial Retraining**. It prioritizes structural patterns and infrastructure over linguistic "politeness."

---

## Feature Engineering & Extraction

The system transforms raw email strings into a **14-dimensional numerical vector**. Moving beyond simple keyword counting, the project utilizes **Linguistic Ratios** to identify adversarial tactics.

### 1. Basic Heuristic Counts
* **`num_words`**: Total token count to identify "Wall of Text" padding.
* **`num_links` / `num_unique_domains`**: Identifies infrastructure complexity and redirection depth.
* **`num_email_addresses`**: Detects "Contact us" social engineering hooks.
* **`num_spelling_errors`**: Measured via `pyspellchecker` to identify low-effort spam.
* **`num_urgent_keywords`**: Tracks 25+ panic-inducing triggers (e.g., *Required, Suspended, Unauthorized*).

### 2. Advanced Behavioral Ratios (The "Robustness" Layer)
* **Unique Word Density ($TTR$):** Measures Lexical Richness. Low density ($<0.4$) flags repetitive institutional templates used to "drown out" malicious signals.
* **Link Density:** Normalizes URLs against word count ($N_{links} / N_{words}$) to prevent "Link Dilution" in long-form text.
* **Stopwords Ratio:** Measures natural language "flow." High ratios often correlate with human-written, non-urgent academic correspondence.
* **Unique Domain Ratio:** Detects if multiple links point to the same host or diverse external entities.
* **Spelling Error Ratio:** Normalizes errors against length to distinguish between typos and systemic low-quality generation.


---

## Adversarial Attack Strategy

The core research demonstrates the **Bypass Attack**. By crafting a "Chameleon Email"—one that is long, repetitive, and uses a formal academic tone—attackers can lower the **Victim Model's** risk score ($<0.50$). 

The **Defender Model** counters this by maintaining a high weight on normalized features ($Link Density$), ensuring the attack is caught regardless of the "padding" used.

---

##  Dataset Information

* **Primary Source:** [Kaggle Email Phishing Dataset](https://www.kaggle.com/datasets/ethancratchley/email-phishing-dataset) by Ethan Cratchley.
* **Adversarial Samples:** Custom-curated "Chameleon" samples designed to test sensitivity to high-vocabulary and low-urgency attacks.
