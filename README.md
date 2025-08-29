Tensor Network Based Feature Learning Model
=====

| ![Training time and test MSE plots](images/Dynamics.png) |
|:--:|
| **Figure 1:** Plots of the training time (first row) and test MSE (second row) of FL and CV models (🟠 orange and 🔵 blue curves respectively) as a function of the number of features **P** for different real-life datasets (column-wise). Solid lines represent mean metric calculations and shaded regions depict **+- 1** standard deviation around the mean across 10 restarts. *The proposed FL model requires consistently less time to train compared to the conventional cross-validation.* Likewise, the prediction error of the FL model is either similar to CV (shaded regions intersect) or significantly lower (Yacht data), demonstrating the superiority of the FL model. |

## Project Description
Many approximations were suggested to circumvent the cubic complexity of kernel-based algorithms, allowing their application to large-scale datasets. One strategy is to consider the primal formulation of the learning problem by mapping the data to a higher-dimensional space using tensor-product structured polynomial and Fourier features. The curse of dimensionality due to these tensor-product features was effectively solved by a tensor network reparameterization of the model parameters. However, another important aspect of model training — **identifying optimal feature hyperparameters** — has not been addressed and is typically handled using the standard cross-validation approach.

In this paper, we introduce **the Feature Learning (FL) model**, which addresses this issue by **representing tensor-product features as a learnable Canonical Polyadic Decomposition (CPD)**. By leveraging this CPD structure, we efficiently learn the hyperparameters associated with different features alongside the model parameters using an Alternating Least Squares (ALS) optimization method. We prove the effectiveness of the FL model through experiments on real data of various dimensionality and scale. **The results show that the FL model can be consistently trained 3-5 times faster than and have the prediction quality on par with a standard cross-validated model.**

## Datasets
In this work, we use 5 publicly available UCI regression datasets (Dua and Graff, 2017): *Airfoil, Energy, Yacht, Concrete, Wine.* In order to show and explore the behavior of the FL model on large scale data, we consider the *Airline dataset* (Hensman et al., 2013), contatining recordings of commercial airplane flight delays that occurred in 2008 in the USA.

**Datasets Statistics:**

|  | Airfoil | Energy | Yacht | Concrete | Wine| Airline |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | 
| **Sample Size** | 1502 | 768 | 308 | 1030 | 6497 | 5929413 | 
| **Data Dimensionality** | 5 | 8 | 6 | 8 | 11 | 8 |

## Environment
We use `conda` package manager to install required python packages. Once `conda is installed`, run the following command (while in the root of the repository):
```
conda env create -f environment/environment.yaml
```
This will create a new environment named `general_env` with all required packages already installed. You can install additional packages by running:
```
conda install <package name>
```

In order to read and run `Jupyter Notebooks` you may follow either of two options:
1. [*recommended*] using notebook-compatibility features of IDEs, e.g. via `python` and `jupyter` extensions of [VS Code](https://code.visualstudio.com/).
2. install jupyter notebook packages:
  either with `conda install jupyterlab` or with `conda install jupyter notebook`

## How to Reproduce the Numerical Experiments

0. Create and activate the virtual environment (Environment section):
   ```
   conda activate general_env
   ```

1. Run:
   ```shell
   python load_data.py
   ```
   to load all the datasets locally and configure internal directories. 

2. Run `experiments.ipynb` in VS Code using the `general_env` environment or use `jupyter lab`.

## 📜 Citation

If you find our work helpful, please consider citing the paper:

```bibtex
@inproceedings{pmlr-v258-saiapin25a,
  title={Tensor Network Based Feature Learning Model},
  author={Saiapin, Albert and Batselier, Kim},
  booktitle={Proceedings of The 28th International Conference on Artificial Intelligence and Statistics},
  pages={3277-3285},
  year={2025}
}
```
