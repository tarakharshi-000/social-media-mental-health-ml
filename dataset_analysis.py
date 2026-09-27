import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load dataset
df = pd.read_excel(r'D:\social_media_survey\Fai Datasets Rishu.xlsx', sheet_name='Sheet1')

print("=" * 80)
print("DATASET ANALYSIS REPORT")
print("=" * 80)

print("\n1. BASIC INFORMATION")
print("-" * 80)
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print(f"\nColumn names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print("\n2. DATA TYPES")
print("-" * 80)
print(df.dtypes)

print("\n3. MISSING VALUES")
print("-" * 80)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0])

print("\n4. DUPLICATE ROWS")
print("-" * 80)
print(f"Number of duplicate rows: {df.duplicated().sum()}")

print("\n5. NUMERICAL STATISTICS")
print("-" * 80)
print(df.describe())

print("\n6. AGE DISTRIBUTION")
print("-" * 80)
print(f"Age range: {df['Enter your age:'].min()} - {df['Enter your age:'].max()}")
print(f"Mean age: {df['Enter your age:'].mean():.2f}")
print(f"Median age: {df['Enter your age:'].median():.2f}")
print(f"Standard deviation: {df['Enter your age:'].std():.2f}")

# Age bins
age_bins = [10, 20, 30, 40, 50, 60]
age_labels = ['10-19', '20-29', '30-39', '40-49', '50-59']
df['age_group'] = pd.cut(df['Enter your age:'], bins=age_bins, labels=age_labels, right=False)
print("\nAge group distribution:")
print(df['age_group'].value_counts().sort_index())

print("\n7. CATEGORICAL VARIABLE DISTRIBUTIONS")
print("-" * 80)
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    print(f"\n{col}:")
    print(f"  Unique values: {df[col].nunique()}")
    print(f"  Distribution:")
    value_counts = df[col].value_counts()
    for val, count in value_counts.items():
        print(f"    {val}: {count} ({count/len(df)*100:.1f}%)")

print("\n8. DATA QUALITY ISSUES")
print("-" * 80)
print("Issues identified:")
print("  - 'Which platform do you use the most daily?' has inconsistent naming (Instagram, instagram, INSTAGRAM, etc.)")
print("  - Some entries contain multiple platforms in one cell")
print("  - Some entries are not actually social media platforms (e.g., Wps office, Stocks, BGMI)")
print("  - Missing values present in several columns (5, 1, 2, 2, 0, 0, 3, 1, 1, 1 respectively)")

print("\n9. POTENTIAL TARGET VARIABLES")
print("-" * 80)
print("Possible target variables for ML tasks:")
print("  - Classification: 'Would you ever delete social media permanently?' (Yes/No/Maybe)")
print("  - Classification: 'Do you feel social media affects your mental health?' (Yes/No/Maybe)")
print("  - Classification: 'Have you ever experienced online drama or conflict because of social media?' (Yes/No/Maybe)")
print("  - Classification: 'Do you trust influencers' product recommendations?' (Yes/Sometimes/No)")
print("  - Regression: 'Enter your age:' (predict age from behavior patterns)")

print("\n10. POTENTIAL PREDICTOR VARIABLES")
print("-" * 80)
print("Potential features:")
print("  - Age (numerical)")
print("  - Platform used (categorical - needs cleaning)")
print("  - Time spent on social media (categorical/ordinal)")
print("  - Time of day usage (categorical)")
print("  - Primary purpose (categorical)")
print("  - Break attempts (categorical)")
print("  - Content type engaged (categorical)")
print("  - Online drama experience (categorical)")
print("  - Trust in influencers (categorical)")

print("\n11. DATASET SUITABILITY FOR ML TASKS")
print("-" * 80)
print("This dataset is suitable for:")
print("  - Classification tasks (predicting behavioral outcomes)")
print("  - Exploratory data analysis")
print("  - Pattern recognition in social media usage")
print("\nLimitations:")
print("  - Survey data - self-reported, potential bias")
print("  - No ground truth - cannot establish causality")
print("  - Small sample size (1195 responses)")
print("  - Inconsistent data entry in platform column")
print("  - Cross-sectional data (single time point)")
print("  - Geographic and demographic information not provided")

print("\n12. BIASES AND LIMITATIONS")
print("-" * 80)
print("Potential biases:")
print("  - Selection bias: people who take surveys may not represent general population")
print("  - Response bias: social desirability may affect answers")
print("  - Age distribution skewed (13-59, mean 37.7)")
print("  - Platform column contains non-social media entries")
print("  - No information about geographic location, gender, or socioeconomic status")

# Create visualizations
print("\n13. GENERATING VISUALIZATIONS")
print("-" * 80)

# Age distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Age histogram
axes[0, 0].hist(df['Enter your age:'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].grid(True, alpha=0.3)

# Time spent on social media
time_order = ['Less than 1 hour', '1-2 hours', '3-4 hours', '5+ hours']
time_counts = df['On average, how much time do you spend on social media per day?'].value_counts()
time_counts = time_counts.reindex(time_order)
axes[0, 1].bar(time_counts.index, time_counts.values, color='steelblue', edgecolor='black')
axes[0, 1].set_xlabel('Time spent per day')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Time Spent on Social Media')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3)

# Mental health impact
mental_health = df['Do you feel social media affects your mental health?'].value_counts()
axes[1, 0].pie(mental_health.values, labels=mental_health.index, autopct='%1.1f%%', startangle=90)
axes[1, 0].set_title('Does Social Media Affect Mental Health?')

# Delete social media permanently
delete_social = df['Would you ever delete social media permanently?'].value_counts()
axes[1, 1].pie(delete_social.values, labels=delete_social.index, autopct='%1.1f%%', startangle=90)
axes[1, 1].set_title('Would Delete Social Media Permanently?')

plt.tight_layout()
plt.savefig('D:\\social_media_survey\\figures\\basic_distributions.png', dpi=300, bbox_inches='tight')
print("Saved: basic_distributions.png")

# Platform distribution (top 10)
fig, ax = plt.subplots(figsize=(12, 6))
platform_counts = df['Which platform do you use the most daily?'].value_counts().head(10)
platform_counts.plot(kind='bar', color='coral', edgecolor='black', ax=ax)
ax.set_xlabel('Platform')
ax.set_ylabel('Frequency')
ax.set_title('Top 10 Platforms Used')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('D:\\social_media_survey\\figures\\platform_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: platform_distribution.png")

# Purpose distribution
fig, ax = plt.subplots(figsize=(10, 6))
purpose_counts = df['What do you primarily use social media for?'].value_counts()
purpose_counts.plot(kind='bar', color='lightgreen', edgecolor='black', ax=ax)
ax.set_xlabel('Primary Purpose')
ax.set_ylabel('Frequency')
ax.set_title('Primary Purpose of Social Media Usage')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('D:\\social_media_survey\\figures\\purpose_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: purpose_distribution.png")

# Cross-tabulation: Mental health vs Time spent
fig, ax = plt.subplots(figsize=(10, 6))
crosstab = pd.crosstab(df['On average, how much time do you spend on social media per day?'],
                      df['Do you feel social media affects your mental health?'],
                      normalize='index') * 100
crosstab = crosstab.reindex(time_order)
crosstab.plot(kind='bar', stacked=True, ax=ax)
ax.set_xlabel('Time spent per day')
ax.set_ylabel('Percentage')
ax.set_title('Mental Health Impact by Time Spent on Social Media')
ax.tick_params(axis='x', rotation=45)
ax.legend(title='Mental Health Impact', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('D:\\social_media_survey\\figures\\mental_health_vs_time.png', dpi=300, bbox_inches='tight')
print("Saved: mental_health_vs_time.png")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nFigures saved to: D:\\social_media_survey\\figures\\")
