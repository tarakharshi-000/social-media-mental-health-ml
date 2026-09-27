# ML Dataset Analysis: Mental Health Impact Classification

## Target Variable Selection
**Selected target:** "Do you feel social media affects your mental health?"
- Values: Yes (422, 35.3%), No (393, 32.9%), Maybe (380, 31.8%)
- **Adaptation strategy:** Binary classification - Yes vs (No + Maybe)
  - Yes (positive class): 422 (35.3%)
  - No + Maybe (negative class): 773 (64.7%)
- **Rationale:** Similar to paper's depression detection (identifying "at-risk" users)

## Feature Selection

### Primary Features (from paper)
1. **Age** - "Enter your age:"
   - Type: Numerical (int64)
   - Range: 13-59
   - Mean: 37.7
   - Missing: 0
   - Status: ✅ Ready to use

2. **Usage Duration** - "On average, how much time do you spend on social media per day?"
   - Type: Categorical/Ordinal
   - Values: Less than 1 hour (273), 1-2 hours (302), 3-4 hours (326), 5+ hours (293)
   - Missing: 1 (0.08%)
   - Encoding: Ordinal (0, 1, 2, 3)
   - Status: ✅ Ready to use

3. **Platform** - "Which platform do you use the most daily?"
   - Type: Categorical
   - Unique values: 50
   - Missing: 5 (0.42%)
   - Status: ⚠️ Requires significant cleaning

### Additional Features (from our dataset)
4. **Time of Day** - "When do you use social media the most?"
   - Type: Categorical
   - Values: Morning (259), Afternoon (284), Evening (331), Late Night (319)
   - Missing: 2 (0.17%)
   - Encoding: One-hot or ordinal
   - Status: ✅ Ready to use

5. **Primary Purpose** - "What do you primarily use social media for?"
   - Type: Categorical
   - Values: Entertainment (376), Chatting (291), Studying/Work (278), Shopping (248)
   - Missing: 2 (0.17%)
   - Encoding: One-hot
   - Status: ✅ Ready to use

6. **Break Attempts** - "Have you ever tried to take breaks from social media?"
   - Type: Categorical
   - Values: Yes (348), Maybe (298), No (277), Thinking about it (272)
   - Missing: 0
   - Encoding: One-hot or ordinal
   - Status: ✅ Ready to use

7. **Content Type** - "What type of content do you engage with the most?"
   - Type: Categorical
   - Values: Reels (396), Stories (279), Posts (267), Tweets (250)
   - Missing: 3 (0.25%)
   - Encoding: One-hot
   - Status: ✅ Ready to use

8. **Online Drama** - "Have you ever experienced online drama or conflict because of social media?"
   - Type: Categorical
   - Values: No (420), Yes (397), Maybe (377)
   - Missing: 1 (0.08%)
   - Encoding: One-hot or ordinal
   - Status: ✅ Ready to use

9. **Trust in Influencers** - "Do you trust influencers' product recommendations?"
   - Type: Categorical
   - Values: No (442), Sometimes (393), Yes (359)
   - Missing: 1 (0.08%)
   - Encoding: One-hot or ordinal
   - Status: ✅ Ready to use

### Features Not Used
- "Would you ever delete social media permanently?" - Excluded to avoid target leakage (related to mental health impact)

## Platform Cleaning Strategy

### Issues Identified
1. Inconsistent capitalization (Instagram, instagram, INSTAGRAM, instagarm, Insta)
2. Multiple platforms in single cells (e.g., "Instagram and Spotify")
3. Non-social media entries (Wps office, Stocks, BGMI, platinum(Pt), Ground)
4. Streaming services mixed with social media (Netflix, Disney+, Amazon Prime Video)

### Cleaning Approach
1. **Case normalization:** Convert all to lowercase
2. **Spelling correction:** Map common misspellings to standard names
3. **Platform categorization:** Group into major categories:
   - Social Media: Instagram, Facebook, Twitter/X, TikTok, Snapchat, LinkedIn, WhatsApp, Telegram
   - Streaming: Netflix, Disney+, Amazon Prime Video, Spotify
   - Other: Gaming, productivity, etc.
4. **Multi-platform handling:** Extract primary platform (first mentioned) or create "Multi-platform" category
5. **Non-social media:** Create "Other" category for non-social media platforms

### Platform Categories (Final)
- Instagram (including variants)
- Facebook (including variants)
- TikTok
- Snapchat
- YouTube
- Twitter/X
- LinkedIn
- WhatsApp
- Streaming Services (Netflix, Disney+, Amazon Prime Video, Spotify)
- Gaming (BGMI, Free fire, etc.)
- Other (all others)

## Missing Value Strategy
- **Age:** 0 missing - no action needed
- **Usage Duration:** 1 missing - impute with mode (most common value)
- **Time of Day:** 2 missing - impute with mode
- **Primary Purpose:** 2 missing - impute with mode
- **Platform:** 5 missing - impute with mode after cleaning
- **Break Attempts:** 0 missing - no action needed
- **Content Type:** 3 missing - impute with mode
- **Online Drama:** 1 missing - impute with mode
- **Trust in Influencers:** 1 missing - impute with mode

**Overall missing rate:** < 0.5% - very low, simple imputation appropriate

## Feature Encoding Strategy

### Numerical Features
- Age: Standard scaling (StandardScaler)

### Ordinal Features
- Usage Duration: Ordinal encoding (0, 1, 2, 3)
- Time of Day: Could be ordinal (Morning=0, Afternoon=1, Evening=2, Late Night=3) or one-hot
- Break Attempts: Could be ordinal (No=0, Thinking about it=1, Maybe=2, Yes=3) or one-hot
- Online Drama: Could be ordinal (No=0, Maybe=1, Yes=2) or one-hot
- Trust in Influencers: Could be ordinal (No=0, Sometimes=1, Yes=2) or one-hot

### Nominal Features
- Platform: One-hot encoding (after cleaning)
- Primary Purpose: One-hot encoding
- Content Type: One-hot encoding

**Decision:** Use one-hot encoding for all categorical features to avoid imposing artificial ordinal relationships

## Data Leakage Prevention
✅ **Target variable:** "Do you feel social media affects your mental health?"
- Excluded from features
- Used only as target

✅ **Related variables excluded:**
- "Would you ever delete social media permanently?" - strongly correlated with target, excluded

✅ **Temporal considerations:**
- Cross-sectional data, no temporal leakage concerns

✅ **Preprocessing:**
- Fit preprocessing on training data only
- Apply same transformation to test data

## Class Balance Analysis
- Positive class (Yes): 422 (35.3%)
- Negative class (No + Maybe): 773 (64.7%)
- Ratio: 1:1.83 (moderate imbalance)

**Handling strategy:**
- Option 1: Use class weights in models
- Option 2: Use SMOTE for oversampling
- Option 3: No special handling (moderate imbalance)

**Decision:** Start without special handling, use class weights if performance is poor on minority class

## Train/Test Split Strategy
- **Split ratio:** 80/20 (standard practice)
- **Stratification:** Yes (maintain class balance)
- **Random seed:** 42 (reproducibility)
- **Shuffle:** Yes

## Cross-Validation Strategy
- **Method:** Stratified K-Fold (k=5)
- **Purpose:** Model selection and hyperparameter tuning
- **Reproducibility:** Random seed = 42

## Feature Correlation Analysis
Will perform correlation analysis to identify:
- Highly correlated features (potential for removal)
- Feature-target correlations (feature importance insights)
- Multicollinearity issues

## Baseline Model
- **Strategy:** Simple majority class classifier
- **Expected accuracy:** 64.7% (predict negative class always)
- **Purpose:** Establish lower bound for model performance

## Dataset Suitability for Task
✅ **Suitable:**
- Target variable clearly defined
- Relevant features available
- Sample size adequate (1,195 samples)
- Missing values minimal
- Class imbalance moderate

⚠️ **Limitations:**
- Platform data quality issues
- Missing demographic features (gender, relationship status)
- Self-reported survey data (potential bias)
- Cross-sectional (no causal inference)
- Cultural/geographic context unknown

## Preprocessing Pipeline Summary
1. Load data
2. Clean platform column
3. Handle missing values (mode imputation)
4. Create binary target (Yes vs No+Maybe)
5. Encode categorical features (one-hot)
6. Scale numerical features (standard scaling)
7. Split train/test (80/20, stratified)
8. Apply same preprocessing to train and test

## Feature Importance Analysis Plan
- Use model-based feature importance (XGBoost, Random Forest)
- Use permutation importance
- Use SHAP values for interpretability (if time permits)
- Compare with paper's findings (duration, age, platform most important)

## Expected Challenges
1. Platform cleaning complexity
2. Class imbalance may affect minority class performance
3. Self-reported target may not reflect clinical depression
4. Different feature set from paper may affect comparability
5. Cultural/geographic differences may limit generalizability

## Success Criteria
- Model accuracy > baseline (64.7%)
- Model precision/recall balanced
- Feature importance interpretable
- Results comparable to paper (90% accuracy)
- Clear documentation of all differences
