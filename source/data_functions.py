from typing import Optional

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def print_df_stats(df: pd.DataFrame, save_path=None) -> None:
    n_samples, n_features = df.shape
    task = 'Classification' if 'class' in df.columns else 'Regression'
    print(
        f'# Samples = {n_samples}\n'
        f'# Features = {n_features - 1}\n'
        f'Task: {task}\n'
    )
    print(df.head(3))
    if save_path is not None:
        pd.Series(
            {
                'n_samples': n_samples,
                'n_features': n_features,
                'task': task,
            }
        ).to_csv(save_path)

def load_prepare_data(
    data_path: str, 
    test_size: float, 
    split_seed: int, 
    verbose: bool = False,
    n_sample: Optional[int] = None,
) -> tuple[pd.DataFrame]:
    df = pd.read_csv(data_path, header=0)
    if n_sample:
        df = df.sample(n_sample, random_state=split_seed)
    if verbose: 
        print_df_stats(df)
    if 'class' in df.columns:
        target_col = 'class'
    elif 'target' in df.columns:
        target_col = 'target'
    else:
        raise ValueError("Bad task! Check columns for: 'class' or 'target'")
    return train_test_split(
        df[[v for v in df.columns if v != target_col]], 
        df[target_col], 
        test_size=test_size,
        random_state=split_seed,
    )

def scale_data(x, x_test, y, y_test) -> tuple[pd.DataFrame]:
    mms = MinMaxScaler()
    mms.fit(x)
    x, x_test = mms.transform(x), mms.transform(x_test)
    if y.dtype == float:
        y_mean, y_std = y.mean(), y.std()
        y, y_test = (y - y_mean) / y_std, (y_test - y_mean) / y_std
    return x, x_test, y, y_test

def get_mse_for_model(model, x_train, x_test, y_train, y_test):
    y_train_mean, y_train_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_train_mean) / y_train_std
    y_test = (y_test - y_train_mean) / y_train_std
    model.fit(x_train, y_train) 
    return mean_squared_error(y_test, model.predict(x_test))

def load_transform_data(data_path, test_size, split_seed, transform_x: bool = True, transform_y: bool = True):
    x, x_test, y, y_test = load_prepare_data(data_path, test_size, split_seed)
    if transform_x: 
        mms = MinMaxScaler()
        mms.fit(x)
        x, x_test = mms.transform(x), mms.transform(x_test)  
    if transform_y:
        y_mean, y_std = y.mean(), y.std()
        y, y_test = (y - y_mean) / y_std, (y_test - y_mean) / y_std
    return x, x_test, y, y_test
