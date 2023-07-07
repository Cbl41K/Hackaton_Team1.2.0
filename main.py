import pandas as pd

data = pd.read_csv('data.csv')
bacterial_descriptors = pd.read_csv('bacterial_descriptors.csv')
drug_descriptors = pd.read_csv('drug_descriptors.csv')

merged_data = pd.merge(data, bacterial_descriptors, on='Bacteria', how='left')
merged_data = pd.merge(merged_data, drug_descriptors, on='drug', how='left')

merged_data.to_csv('merged_data.csv', index=False)
print(merged_data.head())