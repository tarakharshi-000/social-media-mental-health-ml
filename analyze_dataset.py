import pandas as pd

df = pd.read_excel(r'D:\social_media_survey\Fai Datasets Rishu.xlsx', sheet_name='Sheet1')

categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    print(f'\n{col}:')
    print(df[col].value_counts())
    print(f'Unique values: {df[col].nunique()}')
