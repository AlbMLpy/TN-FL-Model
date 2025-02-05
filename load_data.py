import os
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path('data')
ARTIFACTS_DIR = Path('artifacts')
DATA2ID = {'energy': 242, 'wine': 186, 'concrete': 165}

def load_data():
    load_banana()
    load_yacht()
    load_airfoil()
    load_spambase()
    load_census_income()
    for data_name in DATA2ID.keys():
        load_uci(data_name)

def load_uci(data_name: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    from ucimlrepo import fetch_ucirepo 
    print(f'{data_name} data download started') 
    dataset = fetch_ucirepo(id=DATA2ID[data_name]) 
    x, y = dataset.data.features, dataset.data.targets 
    yv = y.values[:, 0] if y.ndim > 1 else y.values
    x = x.assign(target=yv)
    x.to_csv(DATA_DIR / f'{data_name}.csv', index=None)
    print(f'{data_name} data download finished')

def _load_url(url: str, name_data: str) -> str:
    # Create data dir if not exists:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f'{name_data} data download started')
    temp_file = DATA_DIR / 'temp.zip'
    urllib.request.urlretrieve(url, filename=temp_file)
    print(f'{name_data} data download finished')
    return temp_file

def load_airline(path_raw: str):
    def get_airline_df(
        path, 
    ):
        return (
            pd.read_csv(path, header=0, index_col=0)
            .drop('Year', axis=1)
            .assign(ArrTime=lambda x: 60*np.floor(x.ArrTime/100) + np.mod(x.ArrTime, 100))
            .assign(DepTime=lambda x: 60*np.floor(x.DepTime/100) + np.mod(x.DepTime, 100))
            .rename({'ArrDelay': 'target'}, axis=1)
            .reset_index(drop=True)
            .astype(int)
        )
    get_airline_df(path_raw).to_csv(DATA_DIR / 'airline.csv', header=True, index=False)

def load_banana():
    url = (
        'https://raw.githubusercontent.com/' 
         + 'SaravananJaichandar/MachineLearning/master/' 
         + 'Standard%20Classification%20Dataset/banana/banana.csv'
    )
    data_path = DATA_DIR / 'banana.csv'
    temp_file = _load_url(url, 'Banana')
    os.rename(temp_file, data_path)
    pd.read_csv(data_path).rename({'Class': 'class'}, axis=1).to_csv(data_path, index=None)

def load_yacht():
    url = 'https://archive.ics.uci.edu/static/public/243/yacht+hydrodynamics.zip'
    temp_file = _load_url(url, 'Yacht')
    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    temp_file2 = DATA_DIR / 'yacht_hydrodynamics.data'
    with open(temp_file2, 'r') as f:
        lines = [[float(v) for v in line.rstrip('\n').split(' ') if v != ''] for line in f]
    pd.DataFrame(
        lines[:-1], 
        columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'target']
    ).to_csv(DATA_DIR / 'yacht.csv', index=None)
    # Remove temp file:
    temp_file.unlink(missing_ok=True)
    temp_file2.unlink(missing_ok=True)

def load_airfoil():
    url = 'https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip'
    temp_file = _load_url(url, 'Airfoil')
    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    temp_file2 = DATA_DIR / 'airfoil_self_noise.dat'
    with open(temp_file2) as f:
        lines = [[float(v) for v in line.rstrip('\n').split('\t') if v != ''] for line in f]
    pd.DataFrame(
        lines[:-1], 
        columns=['f1', 'f2', 'f3', 'f4', 'f5', 'target']
    ).to_csv(DATA_DIR / 'airfoil.csv', index=None)
    # Remove temp file:
    temp_file.unlink(missing_ok=True)
    temp_file2.unlink(missing_ok=True)

def load_spambase():
    url = 'https://archive.ics.uci.edu/static/public/94/spambase.zip'
    temp_file = _load_url(url, 'Spambase')
    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR, members=['spambase.data'])
    temp_file2 = DATA_DIR / 'spambase.data'
    df = pd.read_csv(temp_file2, header=None, names=[f"f{i}" for i in range(1, 58)] + ['class'])
    df.loc[df['class'] == 0, 'class'] = -1
    # Remove temp file:
    temp_file.unlink(missing_ok=True)
    temp_file2.unlink(missing_ok=True)
    df.to_csv(DATA_DIR / 'spambase.csv', index=None)

def load_census_income():
    url = 'https://archive.ics.uci.edu/static/public/2/adult.zip'
    temp_file = _load_url(url, 'Censis Income')
    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR, members=['adult.data', 'adult.test'])
    tf2, tf3 = DATA_DIR / 'adult.data', DATA_DIR / 'adult.test'
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'target'
    ]
    df = pd.read_csv(tf2, names=columns, na_values=['?',' ?'])
    df = pd.concat(
        [df, pd.read_csv(tf3, names=columns, na_values=['?',' ?'])], 
        ignore_index=True, 
        sort=False, 
        axis=0
    )
    df['target'] = df['target'].str.rstrip('.')
    df = df.dropna()
    continuous_labels = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_labels = list(set(set(columns)-set(continuous_labels)))
    categorical_labels.append(categorical_labels.pop(categorical_labels.index('target')))
    df = pd.get_dummies(df, prefix=categorical_labels, columns=categorical_labels, drop_first=True)
    df = df.rename({'target_ >50K': 'class'}, axis=1)
    mask = df.columns[df.dtypes == 'bool']
    df[mask] = df[mask].astype(int)
    df.loc[df['class'] == 0, 'class'] = -1
    df.to_csv(DATA_DIR / 'census_income.csv', header=True, index=False)
    # Remove temp file:
    temp_file.unlink(missing_ok=True)
    tf2.unlink(missing_ok=True)
    tf3.unlink(missing_ok=True) 

def prepare_dir():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    load_data()
    prepare_dir()
