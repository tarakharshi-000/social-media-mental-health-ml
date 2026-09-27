# Experiment Design: Mental Health Impact Classification

## Research Question
**Can we predict whether users perceive social media as affecting their mental health based on their usage patterns and behavior?**

## Target Variable
**Variable:** "Do you feel social media affects your mental health?"
**Original values:** Yes (422), No (393), Maybe (380)
**Binary transformation:**
- Positive class (1): Yes = 422 samples (35.3%)
- Negative class (0): No + Maybe = 773 samples (64.7%)

**Rationale:** This aligns with the paper's depression detection task by identifying users who perceive negative mental health impact from social media use.

## Input Features

### Numerical Features
1. **Age** - "Enter your age:"
   - Range: 13-59
   - Preprocessing: StandardScaler (mean=0, std=1)

### Categorical Features (One-Hot Encoded)
2. **Usage Duration** - "On average, how much time do you spend on social media per day?"
   - Categories: Less than 1 hour, 1-2 hours, 3-4 hours, 5+ hours
   - Encoding: One-hot (4 features)

3. **Platform** - "Which platform do you use the most daily?" (after cleaning)
   - Categories: Instagram, Facebook, TikTok, Snapchat, YouTube, Streaming, Gaming, Other
   - Encoding: One-hot (8 features)

4. **Time of Day** - "When do you use social media the most?"
   - Categories: Morning, Afternoon, Evening, Late Night
   - Encoding: One-hot (4 features)

5. **Primary Purpose** - "What do you primarily use social media for?"
   - Categories: Entertainment, Chatting, Studying/Work, Shopping
   - Encoding: One-hot (4 features)

6. **Break Attempts** - "Have you ever tried to take breaks from social media?"
   - Categories: Yes, Maybe, No, Thinking about it
   - Encoding: One-hot (4 features)

7. **Content Type** - "What type of content do you engage with the most?"
   - Categories: Reels, Stories, Posts, Tweets
   - Encoding: One-hot (4 features)

8. **Online Drama** - "Have you ever experienced online drama or conflict because of social media?"
   - Categories: Yes, No, Maybe
   - Encoding: One-hot (3 features)

9. **Trust in Influencers** - "Do you trust influencers' product recommendations?"
   - Categories: Yes, No, Sometimes
   - Encoding: One-hot (3 features)

**Total features after encoding:** 1 (age) + 4 + 8 + 4 + 4 + 4 + 4 + 3 + 3 = 35 features

## Preprocessing Steps

### Step 1: Data Loading
- Load Excel file: `Fai Datasets Rishu.xlsx`
- Sheet: Sheet1
- Shape: (1195, 11)

### Step 2: Platform Cleaning
**Cleaning rules:**
1. Convert to lowercase
2. Standardize common misspellings:
   - instagram, instagarm, insta, instgram → Instagram
   - facebook, fb → Facebook
   - youtube, yt, you tube → YouTube
   - whatsapp, wtsapp, watsapp → WhatsApp
3. Extract primary platform from multi-platform entries (first mentioned)
4. Categorize platforms:
   - Instagram, Facebook, TikTok, Snapchat, YouTube, Twitter/X, LinkedIn, WhatsApp → Keep as individual
   - Netflix, Disney+, Amazon Prime Video, Spotify → Streaming
   - BGMI, Free fire, etc. → Gaming
   - All others → Other
5. Handle missing values: Impute with mode (most common category)

### Step 3: Missing Value Imputation
- For all categorical features: Impute with mode (most frequent value)
- Age: No missing values
- Overall missing rate < 0.5%, simple imputation appropriate

### Step 4: Target Variable Creation
- Create binary target: `mental_health_impact`
  - 1 if "Do you feel social media affects your mental health?" == "Yes"
  - 0 otherwise ("No" or "Maybe")

### Step 5: Feature Selection
- Exclude: "Would you ever delete social media permanently?" (potential target leakage)
- Include: All other features as specified above

### Step 6: Train/Test Split
- Ratio: 80% train (956 samples), 20% test (239 samples)
- Stratification: Yes (maintain class balance)
- Random seed: 42
- Shuffle: Yes

### Step 7: Categorical Encoding
- One-hot encoding for all categorical features
- Fit on training data only
- Transform both train and test data

### Step 8: Feature Scaling
- StandardScaler for numerical features (age)
- Fit on training data only
- Transform both train and test data

## Models and Hyperparameters

### Model 1: XGBoost (Primary model - matches paper)
**Library:** xgboost
**Parameters:**
- n_estimators: 100
- max_depth: 6
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 0.8
- random_state: 42
- use_label_encoder: False
- eval_metric: 'logloss'
- scale_pos_weight: ratio of negative/positive classes (773/422 ≈ 1.83) to handle class imbalance

**Rationale:** Matches paper's best-performing model, standard hyperparameters for tabular data

### Model 2: Random Forest (Comparison - matches paper)
**Library:** sklearn.ensemble
**Parameters:**
- n_estimators: 100
- max_depth: None
- min_samples_split: 2
- min_samples_leaf: 1
- random_state: 42
- class_weight: 'balanced' (to handle class imbalance)

**Rationale:** Matches paper's comparison model, standard hyperparameters

### Model 3: Naïve Bayes (Comparison - matches paper)
**Library:** sklearn.naive_bayes
**Type:** GaussianNB (appropriate for continuous features after encoding)
**Parameters:**
- Default parameters (no hyperparameters to tune)

**Rationale:** Matches paper's comparison model, simple baseline

### Baseline Model: Majority Class Classifier
**Library:** sklearn.dummy
**Type:** DummyClassifier(strategy='most_frequent')
**Purpose:** Establish lower bound (64.7% accuracy)

## Evaluation Metrics

### Primary Metrics (matches paper)
1. **Accuracy** - Overall correctness
2. **Precision** - True positive rate (how many predicted positive are actually positive)
3. **Recall** - Sensitivity (how many actual positives are correctly predicted)
4. **F1-score** - Harmonic mean of precision and recall

### Additional Metrics
5. **ROC-AUC** - Area under ROC curve (threshold-independent performance)
6. **Confusion Matrix** - Detailed classification results
7. **Classification Report** - Per-class metrics

### Why These Metrics?
- **Accuracy:** Overall model performance (matches paper)
- **Precision/Recall:** Important for imbalanced classification
- **F1-score:** Balanced measure (matches paper)
- **ROC-AUC:** Threshold-independent performance assessment
- **Confusion Matrix:** Detailed error analysis

## Cross-Validation Strategy
- **Method:** Stratified 5-Fold Cross-Validation
- **Purpose:** Model selection and performance estimation
- **Random seed:** 42
- **Stratification:** Yes (maintain class balance in each fold)

## Experimental Procedure

### Step 1: Data Preparation
1. Load dataset
2. Clean platform column
3. Handle missing values
4. Create binary target
5. Split train/test (80/20, stratified, seed=42)

### Step 2: Preprocessing Pipeline
1. Fit OneHotEncoder on training categorical features
2. Fit StandardScaler on training numerical features
3. Transform training data
4. Transform test data using fitted transformers

### Step 3: Model Training
1. Train XGBoost on training data
2. Train Random Forest on training data
3. Train Naïve Bayes on training data
4. Train baseline (majority class) on training data

### Step 4: Model Evaluation
1. Predict on test set using all models
2. Calculate metrics: accuracy, precision, recall, F1-score, ROC-AUC
3. Generate confusion matrices
4. Perform cross-validation on training set

### Step 5: Feature Importance Analysis
1. Extract feature importance from XGBoost
2. Extract feature importance from Random Forest
3. Compare with paper's findings (duration, age, platform most important)

### Step 6: Results Comparison
1. Compare our best model accuracy with paper's 90% baseline
2. Analyze differences due to dataset variations
3. Document all limitations and differences

## Reproducibility Measures
- **Random seed:** 42 for all random operations
- **Library versions:** Fixed in requirements.txt
- **Data split:** Stratified 80/20 with seed=42
- **Preprocessing:** Identical pipeline for all models
- **Code organization:** Modular, documented functions

## Design Decision Rationale

### Why Binary Classification?
- Paper uses binary classification (depression vs no depression)
- Our target has 3 classes (Yes/No/Maybe)
- Binary transformation (Yes vs No+Maybe) aligns with paper's "at-risk" detection
- Simpler interpretation and comparison

### Why Include Additional Features?
- Paper used: age, gender, relationship status, duration, platform
- We have: age, duration, platform + additional behavioral features
- Additional features may improve performance
- Provides richer analysis beyond paper's scope

### Why One-Hot Encoding?
- Categorical features have no inherent order
- Avoids imposing artificial ordinal relationships
- Standard approach for nominal categorical data
- Works well with tree-based models

### Why Class Weighting?
- Moderate class imbalance (35.3% vs 64.7%)
- Class weighting helps models learn minority class patterns
- Better than oversampling (no synthetic data)
- Matches best practices for imbalanced classification

### Why 80/20 Split?
- Standard practice in ML
- Sufficient training data (956 samples)
- Adequate test data (239 samples) for reliable evaluation
- Matches common practice in survey-based studies

### Why Stratified Split?
- Maintains class balance in train and test
- Ensures representative test set
- Important for imbalanced classification
- Standard practice for classification tasks

### Why These Hyperparameters?
- Standard default values for each algorithm
- No extensive hyperparameter tuning (avoid overfitting)
- Comparable to paper's approach (not specified in paper)
- Focus on methodology comparison rather than optimization

### Why Multiple Evaluation Metrics?
- Paper uses accuracy, precision, recall, F1-score
- We match these metrics for direct comparison
- Additional metrics (ROC-AUC) provide deeper insight
- Confusion matrix for detailed error analysis

## Expected Outcomes
- **Baseline accuracy:** 64.7% (majority class)
- **Target performance:** Approach paper's 90% accuracy
- **Best model:** XGBoost (matches paper)
- **Key features:** Duration, age, platform (matches paper findings)
- **Differences:** Lower accuracy due to different target and dataset

## Limitations of This Design
1. **Different target:** Mental health perception vs clinical depression
2. **Missing features:** No gender or relationship status
3. **Platform quality:** Requires significant cleaning
4. **Survey bias:** Self-reported data
5. **Cross-sectional:** No causal inference
6. **Cultural context:** Unknown geographic/cultural factors
7. **Sample size:** 1,195 may limit complex modeling

## Success Criteria
- Model accuracy > baseline (64.7%)
- XGBoost performs best (matches paper)
- Feature importance aligns with paper (duration, age, platform)
- Clear documentation of all differences
- Reproducible results with fixed random seed
- Complete comparison with paper's 90% baseline
