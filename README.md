---

# Mental Health Text Classifier

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Status](https://img.shields.io/badge/status-experimental-yellow.svg)

## Overview

This project explores the use of Natural Language Processing (NLP) and Artificial Neural Networks (ANN) to classify mental health–related text. The system attempts to identify signals of mental health conditions (e.g., stressed, depressed, suicidal) from raw online statements.

⚠**Disclosure:** At its current stage, the model is **not yet capable of performing full sentiment analysis**. Predictions may be incomplete or unreliable, especially for nuanced emotional states. This repository is primarily for **research and learning purposes**.

---

## Project Details

### Tech Stack

* **Language:** Python
* **Libraries:** TensorFlow / Keras, scikit-learn, pandas, numpy
* **Deployment:** [Hugging Face Spaces](https://huggingface.co/spaces/ghozyreuski/SentimentAnalysis)

### Repository Structure

```
mental-health-text-classifier/
│
├── ann_text_classifier.h5   # Trained ANN model (HDF5 format)  
├── final_ann_model.keras    # Saved ANN model (Keras format)  
├── inference.ipynb          # Inference notebook for testing model predictions  
├── label_encoder.pkl        # Label encoder for target classes  
├── tokenizer.json           # Tokenizer for text preprocessing  
└── README.md                # Project documentation  
```

---

## Methodology

1. **Data Preparation** – Text cleaning, tokenization, padding sequences
2. **Exploratory Analysis** – Word frequency, class distribution
3. **Modeling** – Artificial Neural Network (ANN)

   * Embedding Layer
   * Dense layers with ReLU activation
   * Output layer with Softmax
4. **Evaluation** – Accuracy, confusion matrix, qualitative checks
5. **Deployment** – Hosted as an interactive app on Hugging Face

---

## Dataset

* **Source:** Online mental health text dataset (anonymized for privacy)
* **Target Labels:** Categories such as *stressed*, *depressed*, *suicidal*, *normal*

---

## Results & Insights

* The ANN shows **some ability** to distinguish between categories.
* Struggles with subtle sentiment differences (e.g., *stressed vs. depressed*).
* Needs improvement via larger dataset and model fine-tuning.

---

## Limitations

* **Not yet a true sentiment analysis model.**
* Limited dataset size → reduced generalization.
* Not reliable for critical use cases (e.g., real suicide prevention systems).

---

## References

* TensorFlow Documentation
* Research papers on mental health NLP classification

---

## Author

👤 **Ghozy Reuski**

* GitHub: [@ghozyreuski](https://github.com/ghozyreuski)
* Hugging Face: [Sentiment Analysis App](https://huggingface.co/spaces/ghozyreuski/SentimentAnalysis)
* LinkedIn: [Ghozy Alfisyahr Reuski](https://www.linkedin.com/in/ghozy-alfisyahr-reuski-1133481ba/)

---
