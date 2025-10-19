# TECHTONES - DECODING PUBLIC SENTIMENT ON GOOGLE AND APPLE FROM TWITTER

## OVERVIEW
**TechTones** is a Natural Language Processing (NLP)-driven sentiment analysis project designed to decode public perception of **Apple** and **Google**, two of the world’s most influential technology companies.  
By analyzing thousands of real tweets, the project classifies sentiment as **positive**, **negative**, or **neutral**, uncovering emotional and linguistic patterns around both brands.

The project demonstrates how NLP and machine learning can transform unstructured social media data into actionable insights for **brand monitoring**, **marketing strategy**, and **reputation management**.

## AUTHORS
- Kiprono Ben  
- Norman Mwapea  
- Pauline Kariuki  
- Wesley Owino  
- Judith Otieno  
- Alvin Kipleting  

## OBJECTIVES
The project’s core objectives are:

1. Develop an NLP model to classify Apple and Google tweets as positive, negative, or neutral.  
2. Clean and preprocess text data through tokenization, normalization, stopword removal, and lemmatization.  
3. Transform textual data into numerical features using **TF-IDF vectorization**.  
4. Evaluate multiple machine learning algorithms to identify the best-performing classifier.  
5. Provide interpretable, data-driven insights about public sentiment and brand reputation.  
6. Create a modular and scalable pipeline for potential real-time brand intelligence systems.

## DATA DESCRIOTION
**Source:** [CrowdFlower - Brands and Product Emotions](https://data.world/crowdflower/brands-and-product-emotions)  
**Records:** ~9,000 tweets collected on August 30, 2013.

| Feature | Description |
|----------|-------------|
| tweet_text | Full tweet text referencing Apple or Google products. |
| emotion_in_tweet_is_directed_at | Product or brand mentioned (e.g. iPhone, Android, Google). |
| is_there_an_emotion_directed_at_a_brand_or_product | Annotated sentiment label — Positive, Negative, or Neutral. |


## DATA CLEANING AND PREPROCESSING
The dataset was refined using the following pipeline:

1. **Remove duplicates** and **null tweet entries**.  
2. Drop rows with missing product references (≈60% of total data).  
3. **Text Normalization:**
   - Lowercasing  
   - Stopword removal (customized to retain key negations)  
   - Punctuation, mentions, and URL stripping  
   - Tokenization  
   - Lemmatization  
   - Word correction via **SymSpell**  
4. **Product Mapping:** Standardized all mentions to two categories -> Apple and Google.


## EXPLORATORY DATA ANALYSIS (EDA)
A comprehensive EDA was performed to visualize data distributions and extract linguistic insights.

### KEY FINDINGS
- **Apple dominates** conversations with ~73% of mentions.  
- **Sentiment Distribution:**  
  - Positive -> 81%  
  - Negative -> 16%  
  - Neutral -> 3%  
- **Top Products:**
  - Apple -> iPad, iPhone, iTunes  
  - Google -> Android, Chrome, Pixel  
- **WordClouds:** Show distinct emotional landscapes for each brand and sentiment type.  
- **Tweet Length:** Google tweets are slightly longer, suggesting more detailed discussions.  
- **N-grams Analysis:**  
  - Apple: “apple store”, “ipad launch”, “temporary store”  
  - Google: “social network”, “google map”, “launch major new”


## FEATURE ENGINEERING
- **Text Vectorization:** TF-IDF used to convert text into numerical features.  
- **Dimensionality Reduction:** Truncated SVD (Latent Semantic Analysis).  
- **Label Encoding:** Converted categorical sentiment classes to numeric form for model input.


## MODELING
Multiple supervised learning algorithms were evaluated:

| Model | Type | Description |
|--------|------|-------------|
| Logistic Regression | Linear | Baseline model with strong interpretability |
| Decision Tree | Tree-based | Simple interpretable model |
| Random Forest | Ensemble | Robust classifier using bagging |
| XGBoost | Gradient Boosting | High-performing, optimized model |

### TRAINING STRATEGY
- **Train-Test Split:** 80/20  
- **Cross-Validation:** Stratified K-Fold  
- **Hyperparameter Tuning:** GridSearchCV  
- **Class Imbalance Handling:** SMOTE (Synthetic Minority Oversampling Technique)


## MODEL EVALUATION
Performance metrics used include:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  

**Target Metric:** ≥80% F1-score across sentiment classes.

✅ **Best Performers:**  
- **Logistic Regression** (best interpretability)  
- **XGBoost** (best overall performance)


## INSIGHTS
- **Apple** tweets express stronger positivity and excitement, especially around product launches.  
- **Google** tweets are more analytical, focusing on software and innovation.  
- **Negative sentiments** often involve technical frustrations (battery, bugs, or design).  
- **Positive sentiments** emphasize innovation, design, and product experience.  
- **Neutral sentiments** mostly reflect news-style or factual commentary.


## MODEL INTERPRETABILITY
Using **LIME (Local Interpretable Model-Agnostic Explanations)**, the most influential words behind sentiment predictions were visualized — improving explainability and trust in model outcomes.


## SCALABILITY AND FUTURE WORK
Future enhancements will include:

- **Real-time data ingestion** via Twitter API v2.  
- Expansion to **multi-brand sentiment analysis**.  
- **Deployment** through a Streamlit/Flask dashboard for interactive exploration.  
- Integration of **transformer-based models** (e.g., BERT, RoBERTa) for deeper semantic understanding.  


## TECH STACK
**Languages:** Python 3.9+  
**Libraries:**  
pandas, numpy, matplotlib, seaborn, nltk, symspellpy, wordcloud,  
scikit-learn, imblearn, xgboost, lime

**Environment:** Jupyter Notebook  
**Version Control:** Git + GitHub  


## Installation & Usage
```bash
# Clone repository
git clone https://github.com/Vain-Guy/TechTones.git
cd TechTones

# Run the notebook
jupyter notebook techtones.ipynb

