# Airbnb Price Prediction & Review Analysis

Analysis of Las Vegas Airbnb listings using real-world data from [Inside Airbnb](https://insideairbnb.com/get-the-data/). This project has two components: predicting listing prices using a neural network, and extracting themes from guest reviews using NLP topic modeling.

*Developed as a group project for Applications of AI Models at UMass Amherst (Spring 2026).*

## Project Overview

### Price Prediction

Built a neural network in TensorFlow/Keras to predict listing prices from listing attributes: bedrooms, bathrooms, room type, host attributes, and neighborhood. Prices were capped at $1,000/night to scope the model to standard rentals (excluding luxury party homes, a distinct market segment).

**Test set performance:**
- **R²: 0.504** — explains roughly half of price variance from listing features alone
- **MAE: $58.50** — average prediction within ~$60 of actual price
- **RMSE: $95.00**

**Key finding:** Listing capacity (bathrooms + bedrooms) drives ~50% of predictions, while location features contribute almost nothing once room type is known. Room type effectively acts as a proxy for location where hotel rooms cluster on the Strip, entire homes are predominantly residential, and so on.

### Review Topic Modeling

Filtered to ~10,000 English-language reviews from 2025, then applied four topic modeling approaches at independently-tuned topic counts:

- **LDA** — perplexity minimized at 3 topics, recovering a clean host / property / location decomposition
- **NMF** — 6 more granular sub-themes within the LDA categories
- **LSA** — 4 components, but underperformed with three of four topics dominated by generic praise patterns rather than distinct themes 
- **BERTopic** — transformer-based semantic clustering as a complementary view

Consistent themes across the stronger methods: Strip proximity, cleanliness, host hospitality, and overall comfort.

## Tools & Libraries

- **Languages:** Python
- **Data:** pandas, NumPy
- **Modeling:** scikit-learn, TensorFlow/Keras, BERTopic
- **NLP:** NLTK, langdetect
- **Visualization:** Matplotlib, Seaborn, Plotly

## Files

- `airbnb_price_prediction_review_analysis.ipynb` — full analysis notebook

## Data Source

[Inside Airbnb — Clark County, NV](https://insideairbnb.com/get-the-data/)
