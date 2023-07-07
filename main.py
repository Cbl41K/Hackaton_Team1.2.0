import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

"""Объединение датасетов и загрузка из других баз"""

data = pd.read_csv('data.csv')
bacterial_descriptors = pd.read_csv('bacterial_descriptors.csv')
drug_descriptors = pd.read_csv('drug_descriptors.csv')

merged_data = pd.merge(data, bacterial_descriptors, on='Bacteria', how='left')
merged_data = pd.merge(merged_data, drug_descriptors, on='drug', how='left')

def calc_rdkit_descriptors(s, mol_objects):
    """Получение дескрипторов из rdkit по smiles"""
    res = []
    for mol in mol_objects:
        if mol is not None:
            res += [eval(s)(mol)]
        else:
            res += [np.NaN]
    return res

smiles = merged_data['smiles'].tolist()
mol_objects = []

for smi in smiles:
    if isinstance(smi, str):
        mol_objects += [Chem.MolFromSmiles(smi)]
    else:
        mol_objects += [None]

mw = calc_rdkit_descriptors('Descriptors.MolWt', mol_objects)
logp = calc_rdkit_descriptors('Descriptors.MolLogP', mol_objects)
hba = calc_rdkit_descriptors('Descriptors.NumHAcceptors', mol_objects)
hbd = calc_rdkit_descriptors('Descriptors.NumHDonors', mol_objects)
tpsa = calc_rdkit_descriptors('Descriptors.TPSA', mol_objects)

merged_data['Molecular_Weight'] = mw
merged_data['LogP'] = logp
merged_data['HBA'] = hba
merged_data['HBD'] = hbd
merged_data['TPSA'] = tpsa


"""Очистка базы"""

merged_data['gram'] = merged_data['gram'].replace({'n': 0, 'p': 1})
ZOI = ['ZOI_drug', 'ZOI_NP', 'ZOI_drug_NP']
for zoi in ZOI:
    merged_data[zoi] = merged_data[zoi].str.replace(r'\+.*', '', regex=True)
merged_data = merged_data[merged_data['kingdom'] != 'Fungi']
merged_data = merged_data[~merged_data['NP_concentration'].str.contains('/', na=False)]
columns = ['Unnamed: 0.1', 'Unnamed: 0_x', 'Unnamed: 0_y',
           'NP size_min', 'NP size_max',
           'min_Incub_period, h', 'avg_Incub_period, h', 'max_Incub_period, h',
           'growth_temp, C', 'prefered_name', 'kingdom',
           'subkingdom', 'clade', 'phylum', 'class',
           'order', 'family', 'genus', 'species']
merged_data = merged_data.drop(columns, axis=1)


"""Анализ данных, построение графиков"""

numeric_data = merged_data.select_dtypes(include='number')
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


merged_data.to_csv('merged_data.csv', index=False)
print(merged_data.info())
print(merged_data.isnull().mean()*100)