import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

data = pd.read_csv('data.csv')
bacterial_descriptors = pd.read_csv('bacterial_descriptors.csv')
drug_descriptors = pd.read_csv('drug_descriptors.csv')

merged_data = pd.merge(data, bacterial_descriptors, on='Bacteria', how='left')
merged_data = pd.merge(merged_data, drug_descriptors, on='drug', how='left')

merged_data.to_csv('merged_data.csv', index=False)

print(merged_data.info())
print(merged_data.isnull().mean()*100)

smiles = merged_data['smiles'].tolist()
mol_objects = []
for smi in smiles:
    if isinstance(smi, str):
        mol_objects += [Chem.MolFromSmiles(smi)]
    else:
        mol_objects += [None]


def calc_rdkit_descriptors(s, mol_objects):
    res = []
    for mol in mol_objects:
        if mol is not None:
            res += [eval(s)(mol)]
        else:
            res += [np.NaN]
    return res


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

merged_data.to_csv('update_rdkit_merged_data.csv', index=False)

print('-----------------------------')
print(merged_data.info())
print(merged_data.isnull().mean()*100)