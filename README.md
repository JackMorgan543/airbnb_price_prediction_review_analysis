# Airbnb Price Prediction & Review Analysis

Analysis of Las Vegas Airbnb listings using real-world data from 
[Inside Airbnb](https://insideairbnb.com/get-the-data/). This project has 
two components: predicting listing prices using a neural network, and 
extracting themes from guest reviews using NLP topic modeling.

Developed as a group project for Applications of AI Models at UMass Amherst (Spring 2026).

---

## Project Overview

### Price Prediction
Built a neural network in TensorFlow/Keras to predict listing prices based on 
features including room type, number of bedrooms/bathrooms, host attributes, and 
neighborhood. Includes data cleaning, outlier removal, and one-hot encoding of 
categorical variables.

### Review Topic Modeling
Applied four topic modeling approaches to guest reviews to identify recurring themes:
- **LDA** (Latent Dirichlet Allocation) — using Bag-of-Words
- **NMF** (Non-negative Matrix Factorization)
- **LSA** (Latent Semantic Analysis)
- **BERTopic** — transformer-based semantic modeling

Consistent themes across models included location relative to the Strip, 
cleanliness, host responsiveness, and overall comfort.

---

## Tools & Libraries
- **Languages:** Python
- **Data:** pandas, NumPy
- **Modeling:** scikit-learn, TensorFlow/Keras, BERTopic
- **NLP:** NLTK
- **Visualization:** Matplotlib, Seaborn, Plotly

---

## Files
- `airbnb_price_prediction_review_analysis.ipynb` — full analysis notebook

---

## Data Source
[Inside Airbnb — Clark County, NV](https://insideairbnb.com/get-the-data/)
