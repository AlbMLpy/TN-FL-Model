Tensor Network Based Feature Learning Model
=====

## Project Description:
Many approximations were suggested to circumvent the cubic complexity of kernel-based algorithms, allowing their application to large-scale datasets. One strategy is to consider the primal formulation of the learning problem by mapping the data to a higher-dimensional space using tensor-product structured polynomial and Fourier features. The curse of dimensionality due to these tensor-product features was effectively solved by a tensor network reparameterization of the model parameters. However, another important aspect of model training - identifying optimal feature hyperparameters - has not been addressed and is typically solved with the standard cross-validation approach. In this paper, we introduce the Feature Learning (FL) model that resolves this issue by representing tensor-product features as a learnable Canonical Polyadic Decomposition (CPD). By exploiting this CPD structure we can efficiently learn the hyperparameters associated with different features alongside the model parameters using an Alternating Least Squares (ALS) optimization method. We prove the effectiveness of the FL model through experiments on real data of various dimensionality and scale. The results show that the FL model can be consistently trained 3-5 times faster than and have the prediction quality on par with a standard cross-validated model.

## Datasets:
In this work, we use 5 publicly available UCI regression datasets (Dua and Graff, 2017): Airfoil, Energy, Yacht, Concrete, Wine. In order to show and explore the behavior of the FL model on large scale data, we consider the Airline dataset (Hensman et al., 2013), contatining recordings of commercial airplane flight delays that occurred in 2008 in the USA.

Datasets statistics:
| Dataset  | N | D |
| ------------- | ------------- | ------------- | 
| Airfoil | 1502 | 5 |
| Energy | 768 | 8 | 
| Yacht | 308 | 6 | 
| Concrete | 1030 | 8 |
| Wine | 6497 | 11 | 
| Airline | 5929413 | 8 |

N - sample size; D - data dimensionality;

## Environment
We use `conda` package manager to install required python packages. In order to improve speed and reliability of package version resolution it is advised to use `mamba-forge` ([installation](https://github.com/conda-forge/miniforge#mambaforge)) that works over `conda`. Once `mamba is installed`, run the following command (while in the root of the repository):
```
mamba env create -f environment/environment.yaml
```
This will create new environment named `general_env` with all required packages already installed. You can install additional packages by running:
```
mamba install <package name>
```
To activate the virtual environment:
```
mamba activate general_env
```

In order to read and run `Jupyter Notebooks` you may follow either of two options:
1. [*recommended*] using notebook-compatibility features of modern IDEs, e.g. via `python` and `jupyter` extensions of [VS Code](https://code.visualstudio.com/).
2. install jupyter notebook packages:
  either with `mamba install jupyterlab` or with `mamba install jupyter notebook`

*Note*: If you prefer to use `conda`, just replace `mamba` commands with `conda`, e.g. instead of `mamba install` use `conda install`.

## Reproduction of the Numerical Experiments:

0. Create and activate the virtual environment (see Environment section).

1. Run:
   ```shell
   python load_data.py
   ```
   to load all the datasets locally and configure internal directories. 

2. Run the following notebook and follow the instructions therein: 
   ```shell
   jupyter experiments.ipynb
   ```
   to get the numerical experimental results. (Alternatively one can run the notebook in VS Code with appropriate environment)
