import json
from time import time
from typing import Callable
from collections import defaultdict
from typing import Optional, Callable, Iterable

from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, RepeatedKFold


import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from .models.QCPR import QCPR
from .features import FFeature
from .models.QCPRf import QCPRf
from .data_functions import load_transform_data, load_prepare_data
from .general_functions import update_results_dict, query_df

sns.set_theme()

PARAMS = {
    'figure.figsize': (10, 5),
    'figure.constrained_layout.use': True,
    'figure.facecolor': 'white',
    'font.size': 10,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.titlesize': 16,
    'figure.max_open_warning': 50,
}
REG_TASK = 'regression'
CLS_TASK = 'classification'

def model_factory(model_cls, params, task: str = REG_TASK):
    tasks = [REG_TASK, CLS_TASK]
    if task not in tasks:
        raise ValueError(f'Bad task: {task}. Need one of these: {tasks}')
    return Pipeline([('scaler', MinMaxScaler()), (task, model_cls(**params))])

def prepare_train_funct_fl(model_params):
    def train_model(x, y):
        model = model_factory(QCPRf, model_params, REG_TASK) 
        model.fit(x, y)  
        return model, 'FL_Model'
    return train_model

def prepare_train_funct_cv(model_params):
    def train_model(x, y):
        all_scores = []
        for value in model_params['features_list']:
            params = dict(feature_map=value) | model_params['fixed_params']
            model = model_factory(QCPR, params, REG_TASK)
            cv = RepeatedKFold(
                n_splits=model_params['n_splits'],
                n_repeats=model_params['n_repeats'], 
                random_state=model_params['kfold_rs']
            )
            scores = cross_val_score(
                model, x, y, scoring='neg_mean_squared_error', cv=cv, 
                n_jobs=model_params['n_jobs'])
            all_scores.append(np.mean(scores))

        best_feature = model_params['features_list'][np.argmax(all_scores)]
        params = dict(feature_map=best_feature) | model_params['fixed_params']
        model = model_factory(QCPR, params, REG_TASK)
        model.fit(x, y)  
        return model, 'CV_Model'
    return train_model

def data2train_f(data_name: str, model_name: str, p_scale_list: list) -> Callable:
    if model_name == 'FL_Model':
        return prepare_train_funct_fl(data2model_params(data_name, model_name, p_scale_list))
    elif model_name == 'CV_Model':
        return prepare_train_funct_cv(data2model_params(data_name, model_name, p_scale_list))


def train_model_time(train_model: Callable, x: np.ndarray, y: np.ndarray) -> dict:
    start_t = time()
    model, model_name = train_model(x, y)
    end_time = time() - start_t
    return dict(model=model, model_name=model_name, train_time=end_time)

def get_stats_several_trials(
    data_path: str,
    train_model: Callable,
    metric_f: Callable,
    n_trials: int = 10,
    test_size: float = 0.2,
    tqdm_disable: bool = False,
    transform_x: bool = False,
    transform_y: bool = True,
):
    res = defaultdict(list)
    for trial in tqdm(range(1, n_trials + 1), disable=tqdm_disable): # Do several trials to get mean and std of metrics
        x, x_test, y, y_test = load_transform_data(
            data_path, test_size, trial, transform_x, transform_y)
        model_dt = train_model_time(train_model, x, y)
        # Compute stats:
        final_score = metric_f(y_test, model_dt['model'].predict(x_test))
        update_results_dict(
            res, trial=trial, model=model_dt['model_name'], 
            metric=final_score, train_time=model_dt['train_time'],
        )
    return res

def data2model_params(
    data_name: str, 
    model_name: str,
    p_scale_list: list,
    reg_type: str = 'l1', 
    l_pos: bool = False, 
) -> dict:
    if data_name == 'airfoil': 
        data_p = dict(rank=51, m_order=4) 
    elif data_name == 'energy': 
        data_p = dict(rank=15, m_order=4)
    elif data_name == 'yacht': 
        data_p = dict(rank=6, m_order=2)
    elif data_name == 'wine': 
        data_p = dict(rank=25, m_order=16) 
    elif data_name == 'concrete': 
        data_p = dict(rank=10, m_order=8)
    elif data_name == 'airline':
        data_p = dict(rank=20, m_order=64) 

    fixed_p = dict(n_epoch=10, alpha=1e-2, random_state=13)
    if model_name == 'FL_Model':
        model_p = dict(
            fmaps_list=list(map(FFeature, p_scale_list)),
            beta=1e-1, 
            lambda_reg_type=reg_type, 
            positive_lambda=l_pos,
            n_steps_l1=500, 
            update_order_t='lw'
        )
        return fixed_p | data_p | model_p
    elif model_name == 'CV_Model':
        model_params = dict(
            features_list=list(map(FFeature, p_scale_list)),
            n_jobs=1, 
            n_splits=3, 
            n_repeats=2, 
            kfold_rs=1,
            fixed_params=fixed_p | data_p,
        )
        return model_params
    
def plot_mean_conf(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    std_col: str,
    sep_col: str,
    x_label: str = '',
    y_label: str = '',
    title: str = '',
    save_path: str = '',
    show_plot: bool = True,
    y_scale: str = 'linear',
    x_ticks: Optional[Iterable] = None,
):
    with mpl.rc_context(PARAMS):
        for value in df[sep_col].unique():
            subset = df[df[sep_col] == value]
            upper = subset[y_col] + subset[std_col]
            lower = subset[y_col] - subset[std_col]
            if y_scale == 'log':
                lower[lower < 0] = np.median(lower[lower >= 0])
            plt.fill_between(subset[x_col], lower, upper, alpha=0.2)
        sns.lineplot(x=x_col, y=y_col, data=df, hue=sep_col, markers=True, style=sep_col, dashes=False)
        plt.xticks(x_ticks)
        plt.yscale(y_scale)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        if save_path: 
            plt.savefig(save_path, dpi=500)
        if show_plot:
            plt.show()
        else:
            plt.close()

def get_res_df(data_name: str, dir_name: str = 'FLvsCV') -> pd.DataFrame:
    with open(f'./artifacts/{dir_name}/{data_name}/results.json', 'r') as f:
        res = json.load(f)
    return pd.DataFrame(res)

def get_res_table(data_names, n_features, data_dir):
    res_dict = defaultdict(list)
    for data_name in data_names:
        # Model stats:
        if data_name == 'airline':
            dd = query_df(get_res_df(data_name), dict(n_features=6))
        else: 
            dd = query_df(get_res_df(data_name), dict(n_features=n_features))
        ms = defaultdict(dict)
        for model in ['FL_Model', 'CV_Model']:
            dd_model = query_df(dd, dict(model=model))
            Vmc_model = np.round(np.std(dd_model['metric']), 3)
            ms[model]['metric'] = f"{np.round(np.mean(dd_model['metric']), 4)} $\\pm$ {Vmc_model}"
            _time = np.mean(dd_model['train_time'])
            if _time < 1:
                _time = np.round(_time, 3)
            elif _time < 10:
                _time = np.round(_time, 1)
            else:
                _time = np.int64(_time)
            ms[model]['time'] = f"{_time}"
            if model == 'FL_Model':
                model_params = data2model_params(data_name, model, [])
                m_order, rank = model_params['m_order'], model_params['rank']
        # Data stats:
        _x1, _x2, _, _ = load_prepare_data(data_dir / f'{data_name}.csv', 0.5, split_seed=None) #test size does not matter here!
        n_samples, d_dim = _x1.shape[0] + _x2.shape[0], _x1.shape[-1]
        # Combine all the stats:
        update_results_dict(res_dict, Data=data_name, N=n_samples, D=d_dim, M=m_order, R=rank,
            MSE_FL=ms['FL_Model']['metric'], MSE_CV=ms['CV_Model']['metric'],
            TT_FL=ms['FL_Model']['time'], TT_CV=ms['CV_Model']['time'],
        )
    return res_dict
