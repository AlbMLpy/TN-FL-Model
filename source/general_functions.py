import os
from pathlib import Path
from itertools import product
from typing import Any, Iterator

import numpy as np
import pandas as pd

def create_dir_if_not_exists(directory: str) -> os.PathLike:
    path = Path(directory)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path

def query_df(df_res, query: dict) -> pd.DataFrame:
    mask = 1
    for col, val in query.items():
       mask &= (df_res[col] == val)
    return df_res[mask]

def extend_results_dict(res: dict, **kwargs) -> None:
    for key, value in kwargs.items():
        res[key].extend(value.copy())

def update_results_dict(res: dict, **kwargs) -> None:
    for key, value in kwargs.items():
        res[key].append(value)

def dirprod_dict(dt: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    keys = dt.keys()
    for vals in product(*dt.values()):
        yield dict(zip(keys, vals))

def prepare_for_dump(dt):
    for k in dt.keys():
        if isinstance(dt[k][0], np.ndarray):
            dt[k] = [list(v) for v in dt[k]]
    return dt
