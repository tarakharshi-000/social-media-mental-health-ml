# Data Directory

This directory contains the dataset used for the social media mental health impact analysis.

## Dataset Information

- **Filename:** Fai Datasets Rishu.xlsx
- **Format:** Excel file (.xlsx)
- **Sheet:** Sheet1
- **Size:** 1,195 rows × 11 columns
- **Source:** Survey data collected via online forms

## Dataset Description

### Columns

1. **Enter your age:** (Numerical) Age of respondent (13-59 years)
2. **Which platform do you use the most daily?** (Categorical) Primary social media platform
3. **On average, how much time do you spend on social media per day?** (Categorical) Daily usage duration
4. **When do you use social media the most?** (Categorical) Time of day of peak usage
5. **What do you primarily use social media for?** (Categorical) Primary purpose of usage
6. **Have you ever tried to take breaks from social media?** (Categorical) Break attempt history
7. **Do you feel social media affects your mental health?** (Categorical) Mental health impact perception
8. **What type of content do you engage with the most?** (Categorical) Primary content type
9. **Have you ever experienced online drama or conflict because of social media?** (Categorical) Online conflict experience
10. **Do you trust influencers' product recommendations?** (Categorical) Trust in influencers
11. **Would you ever delete social media permanently?** (Categorical) Intention to delete social media

### Target Variable

**Original question:** "Do you feel social media affects your mental health?"
**Original values:** Yes (422, 35.3%), No (393, 32.9%), Maybe (380, 31.8%)
**Binary transformation:** Yes (positive class) vs No+Maybe (negative class)

### Data Quality

- **Missing values:** < 0.5% per column (very low)
- **Duplicate rows:** 0
- **Platform column:** 50 unique values with inconsistent naming, requires cleaning
- **Class distribution:** Moderately imbalanced (35.3% positive class)

## Preprocessing Applied

1. **Platform cleaning:** Reduced from 50 unique values to 13 standardized categories
2. **Missing value imputation:** Mode imputation for categorical features
3. **Feature encoding:** One-hot encoding for categorical features
4. **Feature scaling:** Standard scaling for numerical features
5. **Train/test split:** 80/20 split with stratification

## Usage

The dataset is automatically loaded by the preprocessing pipeline:

```python
from src.data_preprocessing import full_preprocessing_pipeline

processed_data = full_preprocessing_pipeline('data/Fai Datasets Rishu.xlsx')
```

## Data Privacy

- No personally identifiable information (PII) in the dataset
- All responses are anonymous
- No location or contact information included
- Suitable for public research use

## License

The dataset is provided for research and educational purposes. Please ensure compliance with any applicable data usage policies.

## Citation

If you use this dataset in your research, please cite:

```
Social Media Mental Health Survey Dataset (2026). Anonymous survey responses
collected for research on social media usage and mental health impact.
```
