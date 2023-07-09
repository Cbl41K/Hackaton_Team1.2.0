import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.pipeline import make_pipeline
import graphs
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import pickle


def merge_and_load_datasets(name_1, name_2, name_3):
    """
    Объединяет датасеты и загружает данные из других баз.
    Args:
        name_1, name_2, name_3 (str): Имена CSV-файлов (датасетов).
    Returns:
        merged_data (pandas.DataFrame): Объединенный датасет.
    """
    data = pd.read_csv(name_1)
    bacterial_descriptors = pd.read_csv(name_2)
    drug_descriptors = pd.read_csv(name_3)

    merged_data = pd.merge(data, bacterial_descriptors, on='Bacteria', how='left')
    merged_data = pd.merge(merged_data, drug_descriptors, on='drug', how='left')

    return merged_data


def calculate_rdkit_descriptors(s, mol_objects):
    """
    Вычисляет дескрипторы RDKit для заданных молекул.
    Args:
        s (str): Имя дескриптора для вычисления.
        mol_objects (list): Список объектов молекул.
    Returns:
        descriptors (list): Список значений дескрипторов.
    """
    descriptors = []
    for mol in mol_objects:
        if mol is not None:
            # Вычисление указанного дескриптора для молекулы
            descriptors += [eval(s)(mol)]
        else:
            descriptors += [np.NaN]
    return descriptors


def process_data(data):
    """
    Обрабатывает и очищает базу данных.
    Args:
        data (pandas.DataFrame): Объединенный датасет.
    Returns:
        data (pandas.DataFrame): Обработанный датасет.
    """
    # Преобразование значений в столбце 'gram' на числовые (0 и 1)
    data['gram'] = data['gram'].replace({'n': 0, 'p': 1})

    # Приведение в стандартный вид
    data['drug'] = data['drug'].str.capitalize().str.rstrip()

    data = data.drop(data[data['drug'] == 'Co-trimoxazole'].index)
    data = data.drop(data[data['drug'] == 'Neomycin'].index)

    ZOI = ['ZOI_drug', 'ZOI_NP', 'ZOI_drug_NP']
    for zoi in ZOI:
        # Удаление символа "+" и всего, что следует за ним, в столбцах 'ZOI_drug', 'ZOI_NP' и 'ZOI_drug_NP'
        data[zoi] = data[zoi].str.replace(r'\+.*', '', regex=True)

    # Исключение строк, где значение в столбце 'kingdom' равно 'Fungi'
    data = data[data['kingdom'] != 'Fungi']

    # Исключение строк, где значение в столбце 'NP_concentration' содержит символ '/'
    data = data[~data['NP_concentration'].str.contains('/', na=False)]

    # Приcвоение Nan для Drug_dose, если drug равно Nan
    data.loc[data['drug'].isna(), 'Drug_dose'] = np.nan

    columns = ['Unnamed: 0.1', 'Unnamed: 0_x', 'Unnamed: 0_y',
               'NP size_min', 'NP size_max',
               'min_Incub_period, h', 'avg_Incub_period, h', 'max_Incub_period, h',
               'growth_temp, C', 'prefered_name', 'kingdom',
               'subkingdom', 'clade', 'phylum', 'class',
               'order', 'family', 'genus', 'species']

    # Удаление указанных столбцов
    data = data.drop(columns, axis=1)

    mask = (data['fold_increase_in_antibacterial_activity (%)'] < 0) | (data['fold_increase_in_antibacterial_activity (%)'] > 10)
    data.loc[mask, 'fold_increase_in_antibacterial_activity (%)'] = np.nan

    mask = (data['MolWt'] > 700)
    data.loc[mask, 'MolWt'] = np.nan

    mask = (data['NP size_avg'] > 50)
    data.loc[mask, 'NP size_avg'] = np.nan

    mask = (data['MolLogP'] < -10)
    data.loc[mask, 'MolLogP'] = np.nan

    data['ZOI_drug'] = data['ZOI_drug'].astype(float)
    data['ZOI_NP'] = data['ZOI_NP'].astype(float)
    data['ZOI_drug_NP'] = data['ZOI_drug_NP'].astype(float)

    mask = (data['ZOI_drug'] > 60)
    data.loc[mask, 'ZOI_drug'] = np.nan

    mask = (data['ZOI_NP'] > 60)
    data.loc[mask, 'ZOI_NP'] = np.nan

    mask = (data['ZOI_drug_NP'] > 60)
    data.loc[mask, 'ZOI_drug_NP'] = np.nan

    return data


def load_and_process_drugbank(name, data):
    """
    Загружает данные из одного датасета и объединяет со вторым.
    Args:
        name (str): Имя CSV-файла (датасета).
        data (pandas.DataFrame): Другой датасет, содержащий столбец "drug".
    Returns:
        data (pandas.DataFrame): Объединенный и обработанный датасет.
    """
    drugbank = pd.read_csv(name)

    # Замена значений в столбце "name"
    exceptions = {
        'Phenoxymethylpenicillin': 'Penicillin',
        'Polymyxin B': 'Polymyxin',
        'Amphotericin B': 'Amphotericin b'
    }
    drugbank['name'] = drugbank['name'].replace(exceptions)

    # Фильтрация по существующим значениям в столбце "drug"
    drugsfilt = drugbank.name.isin(data.drug.unique())
    drugbank = drugbank[drugsfilt]

    # Удаление ненужных столбцов
    columns = ['Unnamed: 0', 'type', 'state', 'cal_logp', 'cal_molecular_weight',
               'cal_polar_surface_area', 'cal_water_solubility', 'cal_refractivity',
               'cal_polarizability', 'cal_rotatable_bond_count', 'cal_h_bond_acceptor_count',
               'cal_h_bond_donor_count', 'cal_bioavailability', 'exp_molecular_weight',
               'cal_physiological_charge', 'exp_pka', 'half_life', 'drugbank_id', 'exp_logp']
    drugbank = drugbank.drop(columns, axis=1)

    # Объединение датасетов
    data = pd.merge(data, drugbank, left_on='drug', right_on='name', how='left')
    data = data.drop('name', axis=1)

    return data


def analyze_data(data):
    """
    Анализирует данные и строит графики.
    Args:
        data (pandas.DataFrame): Обработанный датасет.
    """
    # Выбор числовых столбцов
    numeric_data = data.select_dtypes(include='number')

    # Построение тепловой карты корреляции
    graphs.matrix_correlation(numeric_data)
    graphs.boxplot(numeric_data)


def save_data(data, name = 'data_new.csv'):
    """
    Сохраняет обработанные данные в CSV-файл.
    Args:
        data (pandas.DataFrame): Обработанный датасет.
        name (str): Имя CSV-файла для сохранения.
    """
    data.to_csv(name, index=False)


def data_for_model(data):
    """
    Обрабатывает данные для обучения модели
    Args:
        data (pandas.DataFrame): Датасет.
    Returns:
        data (pandas.DataFrame): Обработанный датасет.
    """
    columns = ['Drug_dose', 'NP_concentration', 'NumHAcceptors', 'NumHDonors',
               'TPSA', 'cal_pka_basic', 'cal_number_of_rings', 'Bacteria', 'NP_Synthesis',
               'drug', 'Drug_class_drug_bank', 'shape', 'method', 'isolated_from',
               'Tax_id', 'chemID', 'smiles']
    data = data.drop(columns, axis=1)
    x = data.values
    cols = data.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data = pd.DataFrame(x_scaled, columns=cols)

    return data


def training_model(data):
    """
    Обучает модель, выводит метрики (RMSE и R-squared (R^2) score) и строит график "feature importances".
    Args:
        data (pandas.DataFrame): Датасет.
    Returns:
        pipeline (sklearn.pipeline.Pipeline): Обученная модель.
    """
    X = data.drop('ZOI_drug_NP', axis=1)
    y = data['ZOI_drug_NP']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание пайплайна с заполнением пропущенных значений
    pipeline = make_pipeline(
        SimpleImputer(strategy='mean'),
        GradientBoostingRegressor()
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    # Вывод метрик
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("Root Mean Squared Error (RMSE):", rmse)
    r2 = r2_score(y_test, y_pred)
    print("R-squared (R^2) score:", r2)

    # Вывод кросс-валидации
    scores = cross_val_score(pipeline, X, y, cv=3)
    print('cross_val_score:', -scores.mean())

    feature_importance = pipeline.named_steps['gradientboostingregressor'].feature_importances_

    # Сортировка значимости признаков
    sorted_indices = np.argsort(feature_importance)
    sorted_feature_importance = feature_importance[sorted_indices]
    sorted_features = X_train.columns[sorted_indices]

    # График значимости признаков
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_feature_importance)), sorted_feature_importance, align='center')
    plt.yticks(range(len(sorted_feature_importance)), sorted_features)
    plt.xlabel('Значимость признака')
    plt.ylabel('Признак')
    plt.title('Значимость признаков в модели')
    plt.show()

    return pipeline


def save_model(model, filename = 'model_weights.pkl'):
    """
    Сохраняет обученную модель в файл для дальнейшего использования.
    Args:
         model (sklearn.pipeline.Pipeline): Обученная модель.
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def load_model(filename = 'model_weights.pkl'):
    """
    Загружает обученную модель.
    Args:
         filename (str): Путь к файлу.
    Returns:
        model (sklearn.pipeline.Pipeline): Обученная модель.
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model


def predict_ZOI_drug_NP(data, model):
    """
    Предсказывает ZOI_drug_NP в представленном датасете.
    Args:
        data (pandas.DataFrame): Датасет.
        model (sklearn.pipeline.Pipeline): Обученная модель.
    Returns:
        predicted_values (pandas.DataFrame): Предсказания модели.
    """
    features = data[['NP size_avg', 'ZOI_drug', 'ZOI_NP', 'fold_increase_in_antibacterial_activity (%)',
                     'MDR_check', 'gram', 'biosafety_level', 'MolWt', 'MolLogP',
                     'cal_logs', 'cal_pka_acidic', 'cal_rule_of_five']]
    predicted_values = model.predict(features)

    return predicted_values


def main():
    data = merge_and_load_datasets('data.csv', 'bacterial_descriptors.csv', 'drug_descriptors.csv')

    # Получение SMILES-строк и объектов молекул
    smiles = data['smiles'].tolist()
    mol_objects = [Chem.MolFromSmiles(smi) if isinstance(smi, str) else None for smi in smiles]

    # Вычисление дескрипторов RDKit
    descriptors = ['Descriptors.MolWt', 'Descriptors.MolLogP', 'Descriptors.NumHAcceptors',
                   'Descriptors.NumHDonors', 'Descriptors.TPSA']
    for descriptor in descriptors:
        descriptor_values = calculate_rdkit_descriptors(descriptor, mol_objects)
        data[descriptor.split('.')[-1]] = descriptor_values

    data = process_data(data)

    data = load_and_process_drugbank('drugbank.csv', data)
    merged = data
    save_data(data, 'merged_data.csv')

    #analyze_data(data)

    data = data_for_model(data)

    # Разделение датасета для обучения модели по значению таргета
    data, data_ZOI_drug_NP_Nan = data[data['ZOI_drug_NP'].notna()], data[data['ZOI_drug_NP'].isna()]

    model = training_model(data)
    save_model(model)
    model = load_model()

    # Пример использования модели
    print('Предсказания для данных:', predict_ZOI_drug_NP(merged, model))
    merged['ZOI_drug_NP'] = predict_ZOI_drug_NP(merged, model)
    save_data(merged, 'merged_data_2.0.csv')


if __name__ == '__main__':
    main()