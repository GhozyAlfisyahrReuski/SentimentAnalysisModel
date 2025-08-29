---

# 💬 Mental Health Statement Classification (NLP)

![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-orange.svg)
![Model](https://img.shields.io/badge/Model-ANN-lightblue.svg)
![Status](https://img.shields.io/badge/Status-Research--Prototype-yellow.svg)

---

## Overview

Mental health issues such as **depression, anxiety, stress, and suicidal tendencies** often remain hidden due to stigma and lack of awareness. Language — the words people use — can reveal subtle signals of emotional state.

This project explores how **Natural Language Processing (NLP)** can classify online statements into different mental health categories.

⚠**Disclaimer:**
This is a **research prototype**. The current model is **not reliable for real-world sentiment or mental health prediction**. It cannot replace professional assessment and should not be used for clinical purposes.

---

## Repository Structure

```
MentalHealthNLP/
│
├── notebooks/
│   ├── preprocessing.ipynb      # Text preprocessing & cleaning
│   ├── modeling.ipynb           # ANN modeling & evaluation
│   ├── inference.ipynb          # Example inference pipeline
│
├── CombinedData.csv             # Dataset (statements + labels)
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## Dataset

* **Source:** [Kaggle Dataset](#) (collected from online statements, e.g., Twitter, Reddit)
* **Size:** Thousands of rows
* **Columns:**

  * `Statement` → user’s text statement
  * `Status` → mental health label (`normal, depressed, stressed, suicidal, anxiety, bipolar, personality disorder`)
* **Target Grouping:** Optionally grouped into **3 risk levels** → Normal, Mild Risk, High Risk

---

## Methodology

1. **Text Preprocessing**

   * Cleaning (punctuation, stopwords, lowercasing)
   * Tokenization & padding
   * Embedding layer preparation

2. **Modeling**

   * Baseline: Artificial Neural Network (ANN)
   * Planned: Transformer-based models (BERT, RoBERTa)

3. **Evaluation**

   * Accuracy, F1-score, confusion matrix
   * Word frequency analysis per label
   * Top 10 words for each status group

4. **Interpretation**

   * Analysis of model misclassifications
   * Identification of dataset imbalance (e.g., “normal” over-represented)

---

## Tech Stack

* **Language:** Python
* **Frameworks:** TensorFlow / Keras, scikit-learn
* **Libraries:** pandas, numpy, nltk, matplotlib, seaborn, re (regex)

---

## Results & Insights

* The ANN baseline **struggled with critical categories** (e.g., suicidal ideation often misclassified as “normal”).
* **Class imbalance** and **limited vocabulary diversity** reduced performance.
* **Contextual nuances** like sarcasm or misspellings were not captured.

 Future Work:

* Improve preprocessing (spelling correction, contextual embeddings)
* Apply **transformer-based models (BERT, RoBERTa)** for better contextual understanding
* Use **data augmentation** to reduce imbalance
* Explore **multi-level risk classification** (Normal, Mild, High Risk)

---

## Limitations

* Dataset may not generalize beyond online text sources
* No consideration of **tone, history, or external context**
* Cannot be used for **diagnosis or real-time mental health monitoring**
* Current model **does not perform reliable sentiment analysis yet**

---

##  How to Run

Clone the repository:

```bash
git clone https://github.com/yourusername/MentalHealthNLP.git
cd MentalHealthNLP
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run notebooks for preprocessing & training:

```bash
jupyter notebook notebooks/preprocessing.ipynb
```

---

## References

* [NLP for Mental Health — arXiv](https://arxiv.org/abs/2309.13933)
* [Text Classification with Deep Learning — arXiv](https://arxiv.org/abs/2003.11591)
* Kaggle Dataset: Mental Health Statements

---

 **Author**: Ghozy Reuski

* GitHub: [@GhozyAlfisyahrReuski](https://github.com/GhozyAlfisyahrReuski)
* LinkedIn: [Ghozy Alfisyahr Reuski](https://www.linkedin.com/in/ghozy-alfisyahr-reuski-1133481ba/)

---

Do you want me to also **draft a one-page portfolio summary** (PDF-style, combining all projects with quick links) that you can attach to job/research applications?
