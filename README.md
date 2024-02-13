# FIRST: Factor Importance Ranking and Selection for Total Indices

A ``Python3`` module of FIRST, a model-independent factor importance ranking and selection procedure that is based on total Sobol' indices ([Huang and Joseph, 2024][1]). This research is supported by U.S. National Science Foundation grants *DMS-2310637* and *DMREF-1921873*. The ``R`` implementation is also available on [CRAN][2]. 

## Installation

```bash
pip install pyfirst
```

or from source

```bash
pip install git+https://github.com/BillHuang01/pyfirst.git
```

## Usage

### Factor Importance Ranking and Selection

``FIRST`` is the main function of this module. It provides factor importance ranking and selection directly from scattered data without any model fitting, where the importance is computed based on total Sobol' indices ([Sobol', 2001][5]). ``FIRST`` requires the following two arguments:
- a numpy ndarray or a pandas dataframe for the factors/predictors ``X`` 
- a numpy ndarray or a pandas series for the response ``y`` 

``FIRST`` returns a numpy ndarray for the factor importance, with value of *zero* indicating that the factor is not important to the prediction of the response.   

```python
from pyfirst import FIRST
from sklearn.datasets import make_friedman1

X, y = make_friedman1(n_samples=10000, n_features=10, noise=1.0, random_state=43)

FIRST(X, y)
```
For more advanced usages of ``FIRST``, e.g., speeding up for big data, please see [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][7] or [API documentation][10].

To support an easy integration with [sklearn.pipeline.Pipeline][3] for a streamline model training process, we also provide ``SelectByFIRST``, a class that is built from [sklearn.feature_selection][4].

```python
import numpy as np
from pyfirst import SelectByFIRST
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = housing.data
y = np.log(housing.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

pipe = Pipeline([
    ('selector', SelectByFIRST(regression=True,random_state=43)),
    ('estimator', RandomForestRegressor(random_state=43))
]).fit(X_train, y_train)

pipe.predict(X_test)
```
For more details, please see [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][8] or [API documentation][11]. 

### Total Sobol' Indices Estimation

This module also provides the function ``TotalSobolKNN`` for a consistent estimation of total Sobol' indices ([Sobol', 2001][5]) directly from scattered data. When the response is noiseless, ``TotalSobolKNN`` implements the Nearest-Neighbor estimator from [Broto et al. (2020)][6]. For noisy response, ``TotalSobolKNN`` implements the Noise-Adjusted Nearest-Neighbor estimator from [Huang and Joseph (2024)][1]. ``TotalSobolKNN`` returns a numpy ndarray for the total Sobol' indices estimation.

```python
from pyfirst import TotalSobolKNN
from sklearn.datasets import make_friedman1

X, y = make_friedman1(n_samples=10000, n_features=5, noise=1.0, random_state=43)

TotalSobolKNN(X, y, noise=True)
```
For more details and applications, please see [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][9] or [API documentation][12]. 

A documentation of this module is also available on [Read the Docs][13].

## References

Huang, C., & Joseph, V. R. (2024). Factor Importance Ranking and Selection using Total Indices. arXiv preprint arXiv:2401.00800.

Sobol', I. M. (2001). Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates. Mathematics and computers in simulation, 55(1-3), 271-280.

Broto, B., Bachoc, F., & Depecker, M. (2020). Variance reduction for estimation of Shapley effects and adaptation to unknown input distribution. SIAM/ASA Journal on Uncertainty Quantification, 8(2), 693-716.

## Citation

If you find this module useful, please consider citing 

```
@article{huang2024factor,
  title={Factor Importance Ranking and Selection using Total Indices},
  author={Huang, Chaofan and Joseph, V Roshan},
  journal={arXiv preprint arXiv:2401.00800},
  year={2024}
}
```


[1]:https://arxiv.org/abs/2401.00800
[2]:https://cran.r-project.org/web/packages/first/index.html
[3]:https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
[4]: https://scikit-learn.org/stable/modules/feature_selection.html
[5]: https://www.sciencedirect.com/science/article/pii/S0378475400002706
[6]: https://epubs.siam.org/doi/10.1137/18M1234631
[7]: https://colab.research.google.com/github/BillHuang01/pyfirst/blob/main/docs/FIRST.ipynb
[8]: https://colab.research.google.com/github/BillHuang01/pyfirst/blob/main/docs/SelectByFIRST.ipynb
[9]: https://colab.research.google.com/github/BillHuang01/pyfirst/blob/main/docs/TotalSobolKNN.ipynb
[10]: https://pyfirst.readthedocs.io/en/latest/autoapi/pyfirst/index.html#pyfirst.FIRST
[11]: https://pyfirst.readthedocs.io/en/latest/autoapi/pyfirst/index.html#pyfirst.SelectByFIRST
[12]: https://pyfirst.readthedocs.io/en/latest/autoapi/pyfirst/index.html#pyfirst.TotalSobolKNN
[13]: https://pyfirst.readthedocs.io/