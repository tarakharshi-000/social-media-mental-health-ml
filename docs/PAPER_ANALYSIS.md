# Paper Analysis: Detecting the Impact of Social Media on Users' Mental Health Using Machine Learning and XAI

## Paper Information
- **Title:** Detecting the Impact of Social Media on Users' Mental Health Using Machine Learning and XAI
- **Author:** Ara Bela Zulfa Laila
- **Journal:** Jurnal Buana Informatika
- **Volume:** Vol. 17 No. 1
- **Date:** April 2026
- **DOI:** https://doi.org/10.24002/jbi.v17i1.13409
- **License:** Creative Commons Attribution-ShareAlike 4.0 International

## 1. Research Problem
The paper addresses the problem of detecting potential depression caused by social media usage using machine learning techniques. The goal is to develop a predictive system that can identify users at risk of depression based on their social media behavior patterns.

## 2. Methodology
The research follows a supervised machine learning approach:
- Data collection via survey
- Data preprocessing and feature engineering
- Model training with multiple algorithms
- Model comparison using standard evaluation metrics
- Explainable AI (XAI) for model interpretation

## 3. Algorithms/Models Used
The paper compares three machine learning algorithms:
1. **Random Forest** - Ensemble tree-based method
2. **XGBoost** (eXtreme Gradient Boosting) - Gradient boosting framework
3. **Naïve Bayes** - Probabilistic classifier based on Bayes' theorem

**Best performing model:** XGBoost

## 4. Dataset Used
The paper uses survey-based data with the following features:
- **Age** (numerical)
- **Gender** (categorical)
- **Relationship status** (categorical)
- **Daily usage duration** (numerical/categorical)
- **Social media platform** (categorical)

**Target variable:** Depression status (binary classification)

**Sample size:** Not explicitly stated in the abstract, but based on typical survey studies

## 5. Input Features
- Age
- Gender
- Relationship status
- Daily social media usage duration
- Social media platform used

## 6. Target Variable
- Depression status (binary: depressed/not depressed)

## 7. Preprocessing Steps
The paper mentions:
- Data preprocessing techniques (specifics not detailed in abstract)
- Feature scaling and encoding (standard practices)
- Likely handling of missing values
- Categorical encoding for gender, relationship status, platform

## 8. Train/Test/Validation Methodology
The paper does not explicitly specify the train/test split methodology in the abstract. Standard practices would include:
- Train/test split (likely 70/30 or 80/20)
- Cross-validation (possible but not specified)
- Stratified sampling to maintain class balance

## 9. Evaluation Metrics
The paper evaluates models using:
- **Accuracy** - Overall correctness
- **Precision** - True positive rate
- **Recall** - Sensitivity
- **F1-score** - Harmonic mean of precision and recall

## 10. Reported Experimental Results
- **XGBoost:** 90% accuracy with high F1-score (best performance)
- **Random Forest:** Not explicitly stated in abstract
- **Naïve Bayes:** Not explicitly stated in abstract

**Key findings:**
- XGBoost showed the best performance
- Main features affecting depression prediction:
  1. Duration of social media use
  2. Age
  3. Platforms

## 11. Explainable AI (XAI)
The paper incorporates:
- **LIME (Local Interpretable Model-agnostic Explanations)** for model interpretability
- Increases transparency of model predictions
- Provides relevant explanations for individuals
- Strengthens confidence in predictions

## 12. Real-world Application
The paper emphasizes:
- Importance of transparency in mental health model implementation
- Flexible solution for digital applications
- Potential for chatbots or real-time mental health monitoring dashboards

## 13. Comparison with Our Dataset

### Similarities
✅ **Shared features:**
- Age (both datasets have age)
- Daily usage duration (both have time spent)
- Social media platform (both have platform information)

✅ **Similar research question:**
- Both examine social media impact on mental health
- Both use survey data
- Both use classification approach

### Differences
❌ **Different target variables:**
- Paper: Depression (binary)
- Our dataset: Mental health impact perception (Yes/No/Maybe)

❌ **Missing features in our dataset:**
- Gender (not available)
- Relationship status (not available)

❌ **Additional features in our dataset:**
- Time of day usage
- Primary purpose
- Break attempts
- Content type engaged
- Online drama experience
- Trust in influencers
- Would delete social media permanently

❌ **Platform data quality:**
- Paper: Likely clean platform data
- Our dataset: Platform column has data quality issues (inconsistent naming, multiple platforms per cell)

## 14. Reproducible Experiment Design

### Target Variable Adaptation
**Paper target:** Depression (binary)
**Our target:** "Do you feel social media affects your mental health?" (Yes/No/Maybe)

**Adaptation strategy:**
1. **Option A:** Binary classification - Yes vs (No + Maybe)
2. **Option B:** Binary classification - (Yes + Maybe) vs No
3. **Option C:** Multi-class classification - Yes/No/Maybe

**Selected approach:** Option A (Yes vs No+Maybe) as it aligns with detecting "at-risk" users similar to depression detection

### Feature Mapping
| Paper Feature | Our Dataset Feature | Notes |
|---------------|---------------------|-------|
| Age | Enter your age: | Direct match |
| Daily usage duration | On average, how much time do you spend on social media per day? | Needs ordinal encoding |
| Social media platform | Which platform do you use the most daily? | Requires significant cleaning |
| Gender | Not available | Cannot use |
| Relationship status | Not available | Cannot use |

### Additional Features to Include
Since we have additional relevant features, we will include them to potentially improve model performance:
- Time of day usage (When do you use social media the most?)
- Primary purpose (What do you primarily use social media for?)
- Break attempts (Have you ever tried to take breaks from social media?)
- Content type engaged (What type of content do you engage with the most?)
- Online drama experience (Have you ever experienced online drama or conflict because of social media?)
- Trust in influencers (Do you trust influencers' product recommendations?)

### Expected Differences in Results
Our results may differ from the paper's 90% accuracy due to:
1. **Different target variable:** Mental health perception vs clinical depression
2. **Missing demographic features:** No gender or relationship status
3. **Dataset size:** 1,195 responses vs unknown paper sample size
4. **Platform data quality:** Our platform column requires cleaning
5. **Cultural/geographic differences:** Unknown in paper vs our dataset
6. **Survey design differences:** Different question wording and scales
7. **Time period:** Paper from 2026, our dataset timing unknown

## 15. Ethical Considerations
The paper emphasizes:
- Transparency in mental health model implementation
- Appropriate use of XAI for explanations
- Not replacing clinical diagnosis
- Supporting, not replacing, professional mental health services

## 16. Limitations Acknowledged in Paper
- Survey-based data (self-reported bias)
- Cross-sectional design (no temporal causality)
- Model generalizability concerns
- Need for clinical validation

## 17. Experiment We Will Reproduce
We will reproduce the core methodology:
1. **Task:** Binary classification of mental health impact
2. **Algorithms:** XGBoost, Random Forest, Naïve Bayes (same as paper)
3. **Evaluation metrics:** Accuracy, Precision, Recall, F1-score (same as paper)
4. **Comparison:** Compare our best model performance with paper's 90% accuracy baseline
5. **Explainability:** Include feature importance analysis (similar to XAI approach)

## 18. What We Cannot Reproduce
- ❌ Exact dataset (different survey, different questions)
- ❌ Gender and relationship status features (not in our dataset)
- ❌ Exact preprocessing steps (not fully detailed in paper)
- ❌ Exact train/test split methodology (not specified)
- ❌ Exact hyperparameters (not provided in abstract)
- ❌ LIME explanations (unless we implement separately)

## 19. Our Adaptation Strategy
1. **Use same algorithms:** XGBoost, Random Forest, Naïve Bayes
2. **Use same evaluation metrics:** Accuracy, Precision, Recall, F1-score
3. **Adapt target variable:** Mental health impact perception
4. **Use available features:** Age, usage duration, platform (cleaned), plus additional behavioral features
5. **Document all differences:** Clearly explain why results may differ
6. **Add value:** Include additional features and analyses beyond the paper

## 20. Academic Integrity Statement
- We will NOT fabricate paper results
- We will clearly state which results come from the paper (90% accuracy)
- We will clearly state which results come from our implementation
- We will explain all differences between our setup and the paper
- We will not claim to have "reproduced" the paper exactly due to dataset differences
- We will present our work as an adaptation and extension of the paper's methodology
