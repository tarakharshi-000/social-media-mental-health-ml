# Predicting Mental Health Impact from Social Media Usage Patterns

This project implements a machine learning framework to predict whether users perceive social media as affecting their mental health, based on behavioral usage patterns. The methodology is adapted from "Detecting the Impact of Social Media on Users' Mental Health Using Machine Learning and XAI" (Laila, 2026).

## Overview

The project trains and evaluates three classification algorithms (XGBoost, Random Forest, and Naïve Bayes) on a survey dataset of 1,195 responses. The implementation includes a complete preprocessing pipeline, feature engineering, model evaluation, and comprehensive analysis of results.

### Key Results

- Random Forest achieved the highest test accuracy (61.92%) but performed below the baseline (64.85%)
- XGBoost, which achieved 90% accuracy in the original paper, reached only 54.39% in this implementation
- The significant performance difference is attributed to dataset incompatibilities: different target variables (mental health perception vs clinical depression), missing demographic features, and platform data quality issues
- Current features (age, usage patterns, platform) appear insufficient for reliable prediction of mental health impact perception

## 📊 Dataset Description

- **Source:** Survey data collected via online forms
- **Size:** 1,195 responses, 11 columns
- **Target:** Mental health impact perception (Yes vs No+Maybe)
- **Features:** Age, platform, usage duration, time of day, purpose, break attempts, content type, online drama, trust in influencers

### Data Quality
- Platform column required significant cleaning (50 → 13 categories)
- Missing values: < 0.5% per column
- Class distribution: 35.3% positive class (moderate imbalance)

## 🔬 Methodology

### Algorithms Implemented
1. **XGBoost** - Gradient boosting framework (paper's best model)
2. **Random Forest** - Ensemble tree-based method
3. **Naïve Bayes** - Probabilistic classifier
4. **Baseline** - Majority class classifier

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-score (matches paper)
- ROC-AUC (additional metric)
- 5-fold stratified cross-validation

### Preprocessing Pipeline
- Platform cleaning and standardization
- Missing value imputation (mode)
- One-hot encoding for categorical features
- Standard scaling for numerical features
- 80/20 train/test split with stratification

## 📁 Project Structure

```
social_media_survey/
│
├── README.md                          # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── data/
│   ├── Fai Datasets Rishu.xlsx        # Original dataset
│   └── README.md                      # Data documentation
│
├── notebooks/
│   └── analysis.ipynb                 # Jupyter notebook analysis
│
├── src/
│   ├── data_preprocessing.py          # Data preprocessing pipeline
│   ├── train.py                       # Model training
│   ├── evaluate.py                    # Model evaluation
│   ├── additional_analysis.py         # Feature importance analysis
│   └── utils.py                      # Utility functions
│
├── results/
│   ├── model_results.csv              # Model performance metrics
│   ├── paper_comparison.csv           # Comparison with paper
│   ├── evaluation_report.txt          # Detailed evaluation report
│   └── figures/                       # Visualization files
│       ├── confusion_matrix_*.png
│       ├── model_comparison.png
│       ├── cv_comparison.png
│       └── feature_importance_*.png
│
├── docs/
│   ├── PAPER_ANALYSIS.md              # Detailed paper analysis
│   ├── ML_DATASET_ANALYSIS.md         # Dataset analysis for ML task
│   ├── EXPERIMENT_DESIGN.md           # Experimental design
│   ├── PAPER_COMPARISON.md            # Comparison with paper results
│   └── PROJECT_REPORT.md              # Complete project report
│
└── models/                            # Trained model files (not in git)
    ├── xgboost.pkl
    ├── random_forest.pkl
    ├── naive_bayes.pkl
    └── baseline.pkl
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/social_media_survey.git
cd social_media_survey
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Complete Pipeline

Run the main training and evaluation script:

```bash
python src/train.py
python src/evaluate.py
python src/additional_analysis.py
```

Or use the Jupyter notebook for interactive analysis:

```bash
jupyter notebook notebooks/analysis.ipynb
```

### Individual Components

**Data Preprocessing:**
```bash
python src/data_preprocessing.py
```

**Model Training:**
```bash
python src/train.py
```

**Model Evaluation:**
```bash
python src/evaluate.py
```

**Additional Analysis:**
```bash
python src/additional_analysis.py
```

## 📈 Results

### Model Performance Summary

| Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score | CV Mean Accuracy |
|-------|---------------|----------------|-------------|---------------|------------------|
| XGBoost | 54.39% | 35.96% | 38.10% | 36.99% | 57.11% |
| Random Forest | 61.92% | 36.00% | 10.71% | 16.51% | 62.87% |
| Naïve Bayes | 61.51% | 42.31% | 26.19% | 32.35% | 56.27% |
| Baseline | 64.85% | 0.00% | 0.00% | 0.00% | 64.64% |

### Comparison with Paper

| Model | Paper Accuracy | Our Accuracy | Difference |
|-------|---------------|--------------|------------|
| XGBoost | 90% | 54.39% | -35.61% |
| Random Forest | Not reported | 61.92% | N/A |
| Naïve Bayes | Not reported | 61.51% | N/A |

### Key Observations
- All models perform below baseline, indicating limited predictive power
- ROC-AUC scores close to 0.5 suggest models cannot effectively separate classes
- Feature importance analysis shows age, platform, and usage patterns are relevant but insufficient
- Performance difference from paper due to dataset incompatibilities

## 🔍 Comparison with Paper

### Paper Reference
**Title:** Detecting the Impact of Social Media on Users' Mental Health Using Machine Learning and XAI
**Author:** Ara Bela Zulfa Laila
**Journal:** Jurnal Buana Informatika, Vol. 17 No. 1 (2026)
**DOI:** https://doi.org/10.24002/jbi.v17i1.13409

### Key Differences

1. **Target Variable:**
   - Paper: Clinical depression (binary)
   - Our dataset: Mental health impact perception (Yes vs No+Maybe)

2. **Features:**
   - Paper: Age, gender, relationship status, duration, platform
   - Our dataset: Age, duration, platform only (missing gender, relationship status)

3. **Data Quality:**
   - Paper: Likely clean, standardized platform data
   - Our dataset: Platform column required significant cleaning (50 → 13 categories)

4. **Sample Characteristics:**
   - Paper: Unknown sample size and characteristics
   - Our dataset: 1,195 responses, age 13-59, unknown geographic context

### Why Results Differ
The 35.61% accuracy difference between our XGBoost implementation (54.39%) and the paper (90%) is due to:
- Different research questions (depression vs perception)
- Missing demographic features
- Platform data quality issues
- Cultural/geographic differences
- Survey design differences

### Reproducibility Assessment

The paper's 90% accuracy could not be reproduced due to significant dataset differences. However, the methodology was successfully implemented with the same algorithms (XGBoost, Random Forest, Naïve Bayes), evaluation metrics (accuracy, precision, recall, F1-score), and cross-validation approach. All differences between the datasets and their impact on results are documented in the comparison analysis.

## ⚠️ Limitations

### Dataset Limitations
- Self-reported survey data (potential bias)
- Cross-sectional design (no temporal causality)
- Missing demographic information (gender, relationship status)
- Platform data quality issues
- Unknown geographic/cultural context
- Small sample size (1,195 responses)

### Methodological Limitations
- Basic feature engineering
- No hyperparameter tuning
- Simple class imbalance handling
- No advanced techniques (SMOTE, ensemble methods, deep learning)

### Generalizability Limitations
- Results may not apply to other populations
- Perception-based target may not reflect clinical reality
- Single dataset limits generalizability
- Cultural factors not accounted for

## 📚 Documentation

- **[PROJECT_REPORT.md](docs/PROJECT_REPORT.md)** - Complete academic project report
- **[PAPER_ANALYSIS.md](docs/PAPER_ANALYSIS.md)** - Detailed analysis of the reference paper
- **[ML_DATASET_ANALYSIS.md](docs/ML_DATASET_ANALYSIS.md)** - Dataset analysis for ML task
- **[EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md)** - Experimental design document
- **[PAPER_COMPARISON.md](docs/PAPER_COMPARISON.md)** - Detailed comparison with paper results

## 🔬 Reproducibility

### Random Seed
All experiments use random seed = 42 for reproducibility.

### Library Versions
See `requirements.txt` for exact library versions.

### Data Split
- 80/20 train/test split with stratification
- Fixed random seed ensures identical splits

### Reproduction Steps
1. Install dependencies: `pip install -r requirements.txt`
2. Place dataset in `data/` directory
3. Run: `python src/train.py && python src/evaluate.py && python src/additional_analysis.py`
4. Results will be saved in `results/` directory

## 📖 Citation

If you use this code or reference this work, please cite:

```bibtex
@misc{social_media_mental_health_2026,
  title={Predicting Mental Health Impact from Social Media Usage Patterns: A Machine Learning Approach},
  author={Your Name},
  year={2026},
  note={Implementation of methodology from Laila (2026)}
}
```

### Original Paper Citation
```bibtex
@article{laila2026detecting,
  title={Detecting the Impact of Social Media on Users' Mental Health Using Machine Learning and XAI},
  author={Laila, Ara Bela Zulfa},
  journal={Jurnal Buana Informatika},
  volume={17},
  number={1},
  year={2026},
  publisher={Universitas Negeri Semarang},
  doi={https://doi.org/10.24002/jbi.v17i1.13409}
}
```

## 🤝 Contributing

This is an academic project. Contributions are welcome for:
- Bug fixes
- Documentation improvements
- Additional analysis
- Alternative implementations

## 📄 License

This project is for educational purposes. Please refer to the original paper for licensing information.

## ⚠️ Ethical Considerations

1. **No clinical claims:** This work does not claim to diagnose or predict clinical depression
2. **No causal inference:** Survey data cannot establish causality
3. **Privacy:** No personally identifiable information in dataset
4. **Transparency:** All limitations clearly documented
5. **Reproducibility:** Complete code and documentation provided

## 📧 Contact

For questions or inquiries about this project, please contact:
- Your Name: your.email@example.com
- GitHub: https://github.com/yourusername/social_media_survey

## 🙏 Acknowledgments

This work was inspired by the research paper "Detecting the Impact of Social Media on Users' Mental Health Using Machine Learning and XAI" by Ara Bela Zulfa Laila (2026). We acknowledge the original authors' contribution to the field.

---

**Academic Integrity Statement:**

This work represents an honest implementation of the referenced paper's methodology adapted to a different dataset. All results reported are actual outputs from our implementation. We do not claim to have reproduced the paper's 90% accuracy due to significant dataset differences, which are transparently documented. No experimental results have been fabricated, and no paper results have been invented. All limitations are clearly stated.
