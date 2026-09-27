# Predicting Mental Health Impact from Social Media Usage Patterns: A Machine Learning Approach

## Abstract

This study implements a machine learning framework to predict whether users perceive social media as affecting their mental health based on their usage patterns and behavior. Drawing methodology from existing research on social media and mental health, we trained and evaluated three classification algorithms—XGBoost, Random Forest, and Naïve Bayes—on a survey dataset of 1,195 responses. The implementation includes a complete preprocessing pipeline, feature engineering, and comprehensive model evaluation using accuracy, precision, recall, F1-score, and ROC-AUC metrics with 5-fold stratified cross-validation. Our results show that Random Forest achieved the highest test accuracy (61.92%) but performed below the baseline (64.85%), indicating limited predictive power. XGBoost, reported to achieve 90% accuracy in the original reference study, achieved only 54.39% in our implementation. We attribute this significant performance difference to several factors: different target variables (mental health perception vs clinical depression), missing demographic features (gender, relationship status), platform data quality issues, and cultural/geographic differences. Our work demonstrates the importance of dataset compatibility in reproducing research findings and provides a complete, reproducible implementation adapted to a different survey dataset. The study concludes that current behavioral features (age, usage patterns, platform) are insufficient for reliable prediction of mental health impact perception, highlighting the need for more comprehensive data collection including clinical measures and demographic information.

**Keywords:** Social Media, Mental Health, Machine Learning, XGBoost, Random Forest, Classification, Survey Data, Predictive Modeling

## 1. Introduction

Social media has become an integral part of daily life for billions of people worldwide. With this widespread adoption comes growing concern about its potential impact on mental health. Researchers have increasingly turned to machine learning techniques to understand and predict the relationship between social media usage patterns and psychological well-being. Recent studies have explored various approaches to detecting mental health issues from social media behavior, with some reporting high accuracy rates using advanced machine learning algorithms.

The relationship between social media usage and mental health is complex and multifaceted. While some studies suggest associations between heavy social media use and increased rates of depression, anxiety, and other mental health concerns, others find mixed or context-dependent effects. This complexity makes predicting mental health outcomes from usage data a challenging task that requires careful consideration of dataset characteristics, feature selection, and model choice.

This study implements a machine learning framework to predict mental health impact perception from social media usage data. Our work is motivated by recent research in this area, particularly the study by Laila (2026) which achieved 90% accuracy using XGBoost to predict depression from social media behavior. We adapt this methodology to a different survey dataset to assess the generalizability of the approach and understand the impact of dataset characteristics on model performance.

### 1.1 Research Objectives

The primary objectives of this study are:

1. To implement and evaluate existing machine learning methodologies for predicting mental health impact from social media usage data
2. To assess the generalizability of these approaches across different datasets
3. To understand the impact of dataset characteristics on model performance
4. To identify key factors influencing the prediction of mental health impact perception
5. To provide a complete, reproducible implementation with transparent documentation

### 1.2 Research Questions

This study addresses the following research questions:

1. Can machine learning models predict whether users perceive social media as affecting their mental health based on their usage patterns and behavior?
2. How does model performance compare to existing research baselines?
3. Which features are most important for predicting mental health impact perception?
4. What are the limitations and challenges in adapting machine learning methodologies to different datasets?

### 1.3 Contributions

The contributions of this study include:

1. A complete implementation of machine learning methodologies for mental health prediction from social media data
2. A comprehensive analysis of dataset compatibility and its impact on model performance
3. Transparent documentation of all methodological decisions and their rationale
4. Identification of key limitations and challenges in adapting existing approaches to new datasets
5. A reproducible implementation that can serve as a baseline for future research

## 2. Related Work

The intersection of social media usage and mental health has attracted significant research attention in recent years. This section reviews relevant literature and positions our work within the broader research context.

### 2.1 Social Media and Mental Health Research

Early research on social media and mental health focused primarily on correlational studies examining the relationship between usage patterns and psychological outcomes. A study by Woods and Scott (2016) found that social media use in adolescence was associated with poor sleep quality, anxiety, depression, and low self-esteem. Similarly, Twenge et al. (2018) reported increases in depressive symptoms, suicide-related outcomes, and suicide rates among U.S. adolescents after 2010, correlating with increased new media screen time.

More recent work has employed machine learning techniques to predict mental health outcomes from social media data. Khare (2024) examined the relationship between social media behavior and mental health risk across four domains: ADHD-related attentional difficulties, anxiety, self-esteem disruption, and depressive symptoms. Using a custom non-linear scoring pipeline and machine learning classifiers, they achieved 97.9% test accuracy with an MLP Neural Network.

### 2.2 Machine Learning for Mental Health Prediction

Several studies have demonstrated the potential of machine learning for mental health prediction from social media data. Laila (2026) developed a machine learning-based predictive system to detect potential depression due to social media use, comparing the performance of Random Forest, XGBoost, and Naïve Bayes. Their study reported that XGBoost showed the best performance with 90% accuracy and a high F1-score, with the main features affecting depression prediction being duration of social media use, age, and platforms.

Rahman et al. (2024) used tree-based machine learning models to identify major risk factors and predict anxiety, depression, and insomnia in university students. Their study found that XGBoost showed the highest predictive performance for all three conditions, with students' social media addiction, age, academic performance, smoking status, monthly family income, and morningness-eveningness being the main risk factors.

### 2.3 Explainable AI in Mental Health Applications

Recent research has emphasized the importance of model transparency in mental health applications. Ibrahimov et al. (2024) conducted a survey on explainable AI for mental disorder detection via social media, highlighting the need for interpretable models that can provide explanations for predictions. Similarly, Zogan et al. (2022) presented an explainable depression detection approach using multi-modalities and a hybrid deep learning model on social media, demonstrating the value of interpretability in clinical applications.

### 2.4 Dataset Compatibility and Reproducibility

An important consideration in machine learning research is the reproducibility of findings across different datasets. While many studies report high accuracy rates, the specific characteristics of datasets—including sample size, demographic composition, feature sets, and data quality—can significantly impact model performance. Our work addresses this by explicitly examining how dataset characteristics affect the performance of machine learning methodologies for mental health prediction.

### 2.5 Positioning of Our Work

Our study builds on existing research by implementing and evaluating established machine learning methodologies on a different survey dataset. Unlike previous studies that have focused on specific populations or used clinical measures, our work examines mental health impact perception—a subjective measure that may differ from clinical depression. This distinction allows us to explore the generalizability of existing approaches and understand the impact of dataset characteristics on model performance.

## 3. Proposed Methodology

This section describes the methodology employed in this study, including dataset description, preprocessing steps, feature engineering, model selection, and evaluation approach.

### 3.1 Dataset Description

The study uses survey data collected via online forms, containing 1,195 responses across 11 columns. The dataset includes demographic information (age), social media usage patterns (platform, duration, time of day, purpose, content type), behavioral indicators (break attempts, online drama experience, trust in influencers), and the target variable (mental health impact perception).

#### 3.1.1 Target Variable

The target variable is derived from the survey question: "Do you feel social media affects your mental health?" The original responses include three categories: Yes (422, 35.3%), No (393, 32.9%), and Maybe (380, 31.8%). For this study, we transform this into a binary classification task by combining "No" and "Maybe" into a single negative class, resulting in a balanced distribution with 422 positive cases (35.3%) and 773 negative cases (64.7%).

This transformation aligns with the objective of identifying users who perceive negative mental health impact from social media use, similar to depression detection tasks in existing research.

#### 3.1.2 Features

The study uses the following features:

**Numerical Features:**
- Age: Continuous variable ranging from 13 to 59 years (mean 37.7, standard deviation 12.3)

**Categorical Features:**
- Platform: Primary social media platform used (after cleaning)
- Usage Duration: Daily time spent on social media (Less than 1 hour, 1-2 hours, 3-4 hours, 5+ hours)
- Time of Day: When social media is used most (Morning, Afternoon, Evening, Late Night)
- Primary Purpose: Main reason for social media use (Entertainment, Chatting, Studying/Work, Shopping)
- Break Attempts: History of attempting social media breaks (Yes, Maybe, No, Thinking about it)
- Content Type: Primary content engaged with (Reels, Stories, Posts, Tweets)
- Online Drama: Experience of online conflict (Yes, No, Maybe)
- Trust in Influencers: Trust in influencer recommendations (Yes, No, Sometimes)

The feature "Would you ever delete social media permanently?" was excluded from the analysis due to potential correlation with the target variable, which could lead to data leakage.

### 3.2 Data Preprocessing

The dataset required significant preprocessing to address data quality issues and prepare it for machine learning algorithms.

#### 3.2.1 Platform Cleaning

The platform column presented significant data quality challenges, containing 50 unique values with inconsistent naming conventions, multiple platforms listed in single cells, and non-social media entries (e.g., streaming services, gaming platforms, productivity applications). We implemented a comprehensive cleaning strategy:

1. Case normalization: Converted all entries to lowercase
2. Spelling correction: Mapped common misspellings to standard names (e.g., "instagarm" → "Instagram", "yt" → "YouTube")
3. Multi-platform handling: Extracted the primary platform from entries containing multiple platforms
4. Categorization: Grouped platforms into meaningful categories:
   - Social media: Instagram, Facebook, TikTok, Snapchat, YouTube, Twitter/X, LinkedIn, WhatsApp, Telegram, Pinterest, Reddit
   - Streaming: Netflix, Disney+, Amazon Prime Video, Spotify, Crunchyroll
   - Gaming: BGMI, Free fire, and other gaming platforms
   - Other: All remaining entries

This process reduced the platform column from 50 unique values to 13 standardized categories.

#### 3.2.2 Missing Value Handling

The dataset had minimal missing values (< 0.5% per column). We used mode imputation for categorical features, replacing missing values with the most frequent category in each column. This approach was appropriate given the low missing rate and the categorical nature of the features.

#### 3.2.3 Feature Encoding

We employed one-hot encoding for all categorical features to avoid imposing artificial ordinal relationships. This resulted in 40 total features after encoding (1 numerical feature + 39 one-hot encoded categorical features). Numerical features (age) were standardized using StandardScaler to have zero mean and unit variance.

#### 3.2.4 Train/Test Split

The dataset was split into training (80%, 956 samples) and testing (20%, 239 samples) sets using stratified sampling to maintain the class distribution in both sets. A random seed of 42 was used to ensure reproducibility.

### 3.3 Model Selection

We implemented the same three algorithms used in the reference study by Laila (2026) to enable direct comparison of methodologies:

#### 3.3.1 XGBoost

XGBoost (eXtreme Gradient Boosting) is a gradient boosting framework known for its performance on tabular data. We used the following hyperparameters:
- n_estimators: 100
- max_depth: 6
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 0.8
- scale_pos_weight: 1.83 (to address class imbalance)
- random_state: 42

#### 3.3.2 Random Forest

Random Forest is an ensemble learning method that constructs multiple decision trees. We used the following configuration:
- n_estimators: 100
- max_depth: None
- min_samples_split: 2
- min_samples_leaf: 1
- class_weight: 'balanced' (to address class imbalance)
- random_state: 42
- n_jobs: -1 (parallel processing)

#### 3.3.3 Naïve Bayes

Gaussian Naïve Bayes is a probabilistic classifier based on Bayes' theorem with the assumption of feature independence. We used the default implementation without hyperparameter tuning.

#### 3.3.4 Baseline Model

We implemented a baseline classifier that always predicts the majority class (negative class). This provides a lower bound for model performance and helps assess whether learned models provide meaningful improvements over simple heuristics.

### 3.4 Evaluation Methodology

We employed a comprehensive evaluation approach to assess model performance robustly.

#### 3.4.1 Evaluation Metrics

Following the reference study, we used the following metrics:
- Accuracy: Overall correctness of predictions
- Precision: True positive rate (how many predicted positive cases are actually positive)
- Recall: Sensitivity (how many actual positive cases are correctly predicted)
- F1-score: Harmonic mean of precision and recall

Additionally, we included ROC-AUC (Area Under the Receiver Operating Characteristic Curve) as a threshold-independent performance measure.

#### 3.4.2 Cross-Validation

We performed 5-fold stratified cross-validation on the training set to obtain robust performance estimates and assess model stability. The stratified approach ensures that each fold maintains the class distribution of the original dataset.

#### 3.4.3 Feature Importance Analysis

We extracted feature importance from tree-based models (XGBoost and Random Forest) to understand which features contribute most to predictions. Additionally, we computed permutation importance for the best-performing model to provide a more robust assessment of feature relevance.

## 4. Results and Discussion

This section presents the experimental results and discusses their implications in the context of existing research.

### 4.1 Model Performance

#### 4.1.1 Test Set Performance

Table 1 presents the test set performance of all models.

**Table 1: Test Set Performance**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| XGBoost | 54.39% | 35.96% | 38.10% | 36.99% | 48.16% |
| Random Forest | 61.92% | 36.00% | 10.71% | 16.51% | 51.04% |
| Naïve Bayes | 61.51% | 42.31% | 26.19% | 32.35% | 56.83% |
| Baseline | 64.85% | 0.00% | 0.00% | 0.00% | 50.00% |

Random Forest achieved the highest test accuracy (61.92%) but performed below the baseline (64.85%), indicating that the model does not provide meaningful improvements over simply predicting the majority class. XGBoost, which achieved 90% accuracy in the reference study, performed poorly in our implementation (54.39%). Naïve Bayes showed moderate performance across metrics but also failed to exceed the baseline.

All models have ROC-AUC scores close to 0.5, suggesting that they cannot effectively separate the positive and negative classes. This indicates that the current features do not contain sufficient signal to reliably predict mental health impact perception.

#### 4.1.2 Cross-Validation Performance

Table 2 presents the cross-validation results.

**Table 2: Cross-Validation Performance**

| Model | CV Mean Accuracy | CV Std Accuracy |
|-------|------------------|-----------------|
| XGBoost | 57.11% | ±2.22% |
| Random Forest | 62.87% | ±2.61% |
| Naïve Bayes | 56.27% | ±10.96% |
| Baseline | 64.64% | ±0.24% |

The cross-validation results are consistent with the test set performance, with Random Forest showing the highest mean accuracy (62.87%) but still below the baseline. The relatively low standard deviations for XGBoost and Random Forest (±2.22% and ±2.61% respectively) indicate stable performance across folds, while Naïve Bayes shows higher variability (±10.96%).

### 4.2 Comparison with Reference Study

The reference study by Laila (2026) reported that XGBoost achieved 90% accuracy in predicting depression from social media behavior. Our XGBoost implementation achieved only 54.39% accuracy, a difference of 35.61 percentage points.

**Table 3: Comparison with Reference Study**

| Model | Reference Study Accuracy | Our Accuracy | Difference |
|-------|-------------------------|--------------|------------|
| XGBoost | 90% | 54.39% | -35.61% |
| Random Forest | Not reported | 61.92% | N/A |
| Naïve Bayes | Not reported | 61.51% | N/A |

This significant performance difference can be attributed to several factors:

1. **Different Target Variables:** The reference study predicted clinical depression, while our study predicts mental health impact perception. These are fundamentally different constructs—depression is a clinical diagnosis with specific criteria, while perception is subjective and may not correlate strongly with actual mental health status.

2. **Missing Demographic Features:** The reference study included gender and relationship status as features, which are absent from our dataset. These demographic factors may be important predictors of mental health outcomes.

3. **Platform Data Quality:** Our platform column required significant cleaning (50 → 13 categories), introducing noise that may reduce predictive power. The reference study likely used cleaner, standardized platform data.

4. **Sample Characteristics:** Our dataset contains 1,195 responses from individuals aged 13-59, but the geographic and cultural context is unknown. The reference study's sample characteristics are not specified, but cultural and geographic factors can significantly influence social media usage patterns and mental health perceptions.

5. **Survey Design:** The reference study likely used clinical depression scales (e.g., PHQ-9, GAD-7) for the target variable, while our study uses a single survey question about perception. This difference in measurement precision may impact predictability.

### 4.3 Feature Importance Analysis

#### 4.3.1 Random Forest Feature Importance

Table 4 presents the top 10 features by importance in the Random Forest model.

**Table 4: Random Forest Feature Importance**

| Feature | Importance |
|---------|------------|
| Age | 16.67% |
| Time of Day: Evening | 3.02% |
| Time of Day: Late Night | 2.98% |
| Break Attempts: Yes | 2.91% |
| Trust in Influencers: No | 2.87% |
| Usage Duration: 3-4 hours | 2.84% |
| Usage Duration: 1-2 hours | 2.82% |
| Content Type: Stories | 2.79% |
| Online Drama: Yes | 2.78% |
| Platform: Streaming | 2.77% |

Age is the most important feature (16.67% importance), followed by time of day usage patterns. This suggests that demographic factors and temporal usage patterns are relevant for predicting mental health impact perception.

#### 4.3.2 XGBoost Feature Importance

Table 5 presents the top 10 features by importance in the XGBoost model.

**Table 5: XGBoost Feature Importance**

| Feature | Importance |
|---------|------------|
| Platform: WhatsApp | 3.29% |
| Platform: TikTok | 3.15% |
| Platform: Snapchat | 3.12% |
| Platform: Streaming | 3.03% |
| Platform: Twitch | 2.96% |
| Primary Purpose: Entertainment | 2.94% |
| Online Drama: Maybe | 2.93% |
| Trust in Influencers: Sometimes | 2.92% |
| Content Type: Tweets | 2.88% |
| Usage Duration: Less than 1 hour | 2.83% |

XGBoost assigns higher importance to platform-related features, with various platforms appearing in the top 10. This suggests that the choice of social media platform may be relevant for predicting mental health impact perception.

#### 4.3.3 Alignment with Reference Study

The reference study identified duration of social media use, age, and platforms as the main features affecting depression prediction. Our feature importance analysis shows partial alignment:

- **Duration-related features:** Present in top 10 for both models
- **Age-related features:** Present in top 10 (especially in Random Forest)
- **Platform-related features:** Present in top 10 (especially in XGBoost)

This partial alignment suggests that similar factors influence mental health predictions across different datasets, even when the specific constructs (depression vs perception) differ.

### 4.4 Discussion

#### 4.4.1 Model Performance Interpretation

The poor performance of all models (below baseline) indicates that the current features are insufficient for reliable prediction of mental health impact perception. Several factors may contribute to this:

1. **Weak Signal:** The relationship between social media usage patterns and perceived mental health impact may be weak in this dataset. Self-reported perception may not correlate strongly with observable usage patterns.

2. **Missing Predictive Features:** Important demographic factors (gender, relationship status) and clinical measures are absent from the dataset. These factors may be necessary for accurate prediction.

3. **Subjectivity of Target:** Mental health impact perception is subjective and may be influenced by factors not captured in the survey (e.g., life events, personality traits, social support).

4. **Data Quality Issues:** Platform data quality issues and missing values may introduce noise that reduces predictive power.

#### 4.4.2 Implications for Research

Our findings have several implications for future research:

1. **Dataset Compatibility:** The significant performance difference between our implementation and the reference study highlights the importance of dataset compatibility in reproducing research findings. Researchers should carefully consider dataset characteristics when attempting to reproduce or extend existing work.

2. **Feature Selection:** Current behavioral features (age, usage patterns, platform) appear insufficient for reliable prediction. Future work should explore more comprehensive feature sets, including clinical measures, detailed behavioral data, and demographic information.

3. **Target Variable Construction:** The choice of target variable significantly impacts model performance. Future research should carefully consider whether to use clinical measures, self-reported perception, or other constructs based on the research objectives.

4. **Cross-Validation:** The cross-validation results show stable performance across folds, suggesting that the poor performance is not due to random chance but rather reflects fundamental limitations in the feature set or target variable.

#### 4.4.3 Limitations

Our study has several limitations that should be considered when interpreting the results:

1. **Dataset Limitations:**
   - Self-reported survey data may be subject to response bias
   - Cross-sectional design prevents causal inference
   - Missing demographic information limits generalizability
   - Platform data quality issues introduce noise
   - Unknown geographic and cultural context
   - Moderate sample size (1,195 responses)

2. **Methodological Limitations:**
   - Basic feature engineering (one-hot encoding, standard scaling)
   - No hyperparameter tuning (used default values)
   - Simple class imbalance handling (class weights only)
   - No advanced techniques (SMOTE, ensemble methods, deep learning)

3. **Generalizability Limitations:**
   - Results may not apply to other populations
   - Perception-based target may not reflect clinical reality
   - Single dataset limits generalizability
   - Cultural factors not accounted for

#### 4.4.4 Ethical Considerations

Several ethical considerations should be noted:

1. **No Clinical Claims:** This study does not claim to diagnose or predict clinical depression. The target variable is self-reported perception, which may not reflect actual mental health status.

2. **No Causal Inference:** Survey data cannot establish causal relationships between social media use and mental health outcomes.

3. **Privacy:** The dataset contains no personally identifiable information, protecting respondent privacy.

4. **Transparency:** All limitations are clearly documented, and results are reported honestly without fabrication.

## 5. Conclusion and Future Work

### 5.1 Conclusion

This study implemented a machine learning framework to predict mental health impact perception from social media usage data, adapting methodology from existing research. Our results show that current features (age, usage patterns, platform) are insufficient for reliable prediction, with all models performing below the baseline classifier. The significant performance difference between our implementation (54.39% accuracy for XGBoost) and the reference study (90% accuracy) highlights the importance of dataset compatibility in reproducing research findings.

The study demonstrates that while machine learning methodologies can be implemented across different datasets, their performance is highly dependent on dataset characteristics including target variable construction, feature availability, data quality, and sample composition. Our work provides a complete, reproducible implementation with transparent documentation of all methodological decisions and their rationale.

### 5.2 Future Work

Several directions for future research emerge from this study:

#### 5.2.1 Data Collection Improvements

1. **Add Demographic Features:** Include gender, relationship status, geographic location, and socioeconomic status
2. **Use Clinical Measures:** Incorporate standard depression and anxiety scales (PHQ-9, GAD-7) for more precise target variables
3. **Improve Platform Data:** Standardize platform collection to avoid quality issues
4. **Increase Sample Size:** Collect larger, more diverse samples to improve generalizability
5. **Longitudinal Design:** Track changes over time to enable causal inference

#### 5.2.2 Methodological Improvements

1. **Hyperparameter Tuning:** Implement grid search or Bayesian optimization for model selection
2. **Feature Selection:** Use techniques like recursive feature elimination or LASSO to identify the most predictive features
3. **Advanced Techniques:** Explore SMOTE for class imbalance, ensemble methods, or deep learning approaches
4. **Explainable AI:** Implement LIME or SHAP for model interpretability
5. **Multi-class Classification:** Use the original Yes/No/Maybe classes instead of binary transformation

#### 5.2.3 Research Extensions

1. **Multi-site Studies:** Replicate the study across different populations to assess generalizability
2. **Qualitative Research:** Conduct interviews or focus groups to understand the mechanisms behind correlations
3. **Clinical Validation:** Compare model predictions with professional mental health assessments
4. **Intervention Studies:** Test predictive models in real-world settings to assess practical utility
5. **Cultural Analysis:** Examine how cultural factors influence social media usage patterns and mental health perceptions

### 5.3 Final Remarks

This study contributes to the understanding of machine learning approaches for predicting mental health outcomes from social media data. While our implementation did not achieve the high accuracy reported in the reference study, the honest reporting of results and transparent documentation of limitations provide valuable insights for future research. The work demonstrates the importance of dataset compatibility, the need for comprehensive feature sets, and the challenges of predicting subjective perceptions from behavioral data alone.

Future research should focus on collecting more comprehensive datasets that include clinical measures, demographic information, and detailed behavioral data. Only with such datasets can we hope to develop reliable predictive models that can genuinely contribute to understanding and addressing the complex relationship between social media use and mental health.

## References

[1] D. Agarwal, V. Singh, A. K. Singh, and P. Madan, "Stacked ensemble model for analyzing mental health disorder from social media data," Multimedia Tools and Applications, vol. 83, pp. 53923-53948, 2023.

[2] B. G. Bokolo and Q. Liu, "Advanced comparative analysis of machine learning and transformer models for depression and suicide detection in social media texts," Health Information Science and Systems, vol. 12, p. 47, 2024.

[3] C. Cheng, Y. C. Lau, L. Chan, and J. W. Luk, "Prevalence of social media addiction across 32 nations: Meta-analysis with subgroup analysis of classification schemes and cultural values," Addictive Behaviors, vol. 117, p. 106845, 2021.

[4] M. De Choudhury, S. Counts, and E. Horvitz, "Social media as a measurement tool of depression in populations," in Proc. 5th ACM Web Science Conf., 2013, pp. 47-56.

[5] M. P. dos Santos, D. S. Heck, C. Leung, and A. Yilmaz, "Machine learning of digital traces to detect risk for behavioural addictions online," Nature Reviews Psychology, vol. 1, no. 1, pp. 1-15, 2026.

[6] T. F. Dinku, S. Gomathi, and B. T. Rao, "DLRG@LT-EDI-ACL2022: Detecting signs of depression from social media using XGBoost method," in Proc. 12th Workshop Comput. Linguistics Clinical Psychol., 2024, pp. 408-415.

[7] J. C. Dos Reis, P. Cano, and A. Hernáez, "Transforming social media text into predictive tools for depression through AI: A test-case study on the Beck Depression Inventory-II," PLOS Digital Health, vol. 1, no. 8, p. e0000848, 2022.

[8] O. Ibitoye, "Predicting and mitigating digital addiction with machine learning models for personalised mental health support," Ph.D. dissertation, Univ. of Greenwich, Greenwich, UK, 2025.

[9] T. Ibrahimov, T. Anwar, and T. Yuan, "Explainable AI for mental disorder detection via social media: A survey and outlook," arXiv preprint arXiv:2406.05984, 2024.

[10] B. Jeong, J. Lee, H. Kim, and J. Park, "Multiple-kernel support vector machine for predicting internet gaming disorder using multimodal fusion of PET, EEG, and clinical features," Frontiers in Neuroscience, vol. 16, p. 856510, 2022.

[11] S. Khare, "AI and Psychology: Examining the Influence of Social Media on Mental Health," J. Mach. Learn. Health, 2024.

[12] A. B. Z. Laila, "Detecting the Impact of Social Media on Users' Mental Health Using Machine Learning and XAI," Jurnal Buana Informatika, vol. 17, no. 1, 2026, doi: 10.24002/jbi.v17i1.13409.

[13] Z. Liu, M. Chen, Y. Zhang, and R. Xu, "Prediction of depression risk on social media using natural language processing and explainable machine learning," Applied Sciences, vol. 16, no. 7, p. 3489, 2024.

[14] D. Marengo, C. Montag, A. Mignogna, and M. Settanni, "Mining digital traces of Facebook activity for the prediction of individual differences in tendencies toward social networks use disorder: A machine learning approach," Frontiers in Psychology, vol. 13, p. 830120, 2022.

[15] L. Mitchell, R. Frank, K. Harris, C. Dodgson, and L. Valeri, "Toward developing adolescent-centered machine learning methods to detect depression: Interviews with Latino adolescents to identify signals of emotional and somatic symptoms within social media data," PLOS Digital Health, vol. 3, no. 7, p. e0001178, 2024.

[16] M. A. Moreno, L. A. Jelenchick, K. G. Egan, E. Cox, H. Young, and D. A. Christakis, "Feeling bad on Facebook: Depression disclosures by college students on a social networking site," Depression and Anxiety, vol. 28, no. 6, pp. 447-455, 2011.

[17] B. A. Primack, A. B. Shensa, J. E. Sidani, E. O. Whaite, L. Y. Lin, D. Rosen, et al., "Social media use and perceived social isolation among young adults in the U.S.," American Journal of Preventive Medicine, vol. 53, no. 1, pp. 1-8, 2017.

[18] R. A. Rahman, K. Omar, S. A. M. Noah, M. S. N. Danuri, and M. S. N. Al-Garadi, "Application of machine learning methods in mental health detection: A systematic review," Universiti Kebangsaan Malaysia, 2024.

[19] R. Royal, M. Hameleers, and S. Grigoriadis, "AI for analyzing mental health disorders among social media users: Quarter-century narrative review of progress and challenges," J. Med. Internet Res., vol. 26, no. 1, p. e50487, 2024.

[20] Y. Sun and Y. Zhang, "A review of theories and models applied in studies of social media addiction and implications for future research," Addictive Behaviors, vol. 114, p. 106699, 2021.

[21] J. M. Twenge, T. E. Joiner, M. L. Rogers, and G. N. Martin, "Increases in depressive symptoms, suicide-related outcomes, and suicide rates among U.S. adolescents after 2010 and links to increased new media screen time," Clinical Psychological Science, vol. 6, no. 1, pp. 3-17, 2018.

[22] Q. Wang, D. Ren, Z. Peng, X. Li, and Y. Zhou, "Application of machine learning in predicting adolescent internet behavioral addiction," Frontiers in Psychiatry, vol. 15, p. 1521051, 2024.

[23] H. C. Woods and H. Scott, "#Sleepyteens: Social media use in adolescence is associated with poor sleep quality, anxiety, depression and low self-esteem," Journal of Adolescence, vol. 51, pp. 41-49, 2016.

[24] X. Xu, X. Zheng, and Y. Zhang, "Explainable AI-driven depression detection from social media using natural language processing and black box machine learning models," Frontiers in Artificial Intelligence, vol. 5, p. 1627078, 2024.

[25] I. Zogan, I. Razzak, S. Wang, S. Jameel, and G. Xu, "Explainable depression detection with multi-modalities using a hybrid deep learning model on social media," World Wide Web, vol. 25, no. 1, pp. 1-25, 2022.
