import faiss
import numpy as np
import pandas as pd
from twinning import twin
from math import comb
from joblib import Parallel, delayed
from pandas.api.types import is_numeric_dtype, is_bool_dtype
from typing import Dict, List, Optional, Tuple, Union
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

__all__ = ['TotalSobolKNN', 'ShapleySobolKNN', 'FIRSTRank', 'FIRST', 'SelectByFIRST']

def _check_X_y_and_preprocess(
        X:Union[pd.DataFrame, np.ndarray],
        y:Union[pd.Series, np.ndarray], 
        rescale:bool,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
    # function to check for the input X and output y

    # make a copy to avoid mutation of input
    X = X.copy()
    y = y.copy()

    # check for input X 
    assert isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray), f"X must be pd.DataFrame/np.ndarry type, but {type(X)} is provided."
    assert X.ndim == 2, f"Input X must be 2D."
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    assert X.apply(lambda x: not hasattr(x, "sparse")).all(), f"X cannot be sparse. Please convert them to dense."
    assert X.apply(lambda x: np.isfinite(x).all() if is_numeric_dtype(x) else x.notna().all()).all(), f"X cannot contain any missing/infinite value."

    # check for output y
    assert isinstance(y, pd.Series) or isinstance(y, np.ndarray), f"y must be pd.Series/np.ndarray type, but {type(y)} is provided."
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    y = y.squeeze()
    assert y.ndim == 1, f"Output y must be 1D."
    assert not hasattr(y, "sparse"), f"y cannot be sparse. Please convert it to dense."
    assert np.isfinite(y).all(), f"y cannot contain any missing/infinite value."
    if np.issubdtype(y.dtype, np.number):
        assert (y != y[0]).any(), f"y must have more than one unique value."
    else:
        uniq_y, y = np.unique(y, return_inverse=True)
        assert uniq_y.size == 2, f"y can only have two values for classification (only binary classification is supported)."

    # check consistent length between X and y
    assert X.shape[0] == y.size, f"Size of X ({X.shape[0]}) and y ({y.size}) does not match."

    # preprocess for input X if rescale
    if rescale:
        for col in X.columns:
            if is_numeric_dtype(X[col]) and X[col].nunique() > 1:
                if is_bool_dtype(X[col]) or X[col].isin([0,1]).all():
                    # make binary input -1/1, see Gelman 
                    # http://www.stat.columbia.edu/~gelman/research/published/standardizing7.pdf
                    X[col] = 2.0 * (X[col].astype(np.float32) - 0.5)
                else:
                    X[col] = (X[col] - np.mean(X[col])) / np.std(X[col])
    
    return X, y

def _check_mc_setting(
        n_mc: int,
        twin_mc: bool,
        y: np.ndarray,
    ) -> Tuple[int, bool]: 
    # function to check for using monte carlo subsamples

    # check for n_mc: positive integer and cannot be greater than n
    n = y.size
    n_mc = n if n_mc is None else min(n_mc, n)
    assert isinstance(n_mc, int) and n_mc > 0, f"n_mc ({n_mc}) must be a positive integer."

    # check for twin_mc: twinning subsample is available only when reduction ratio >= 2
    twin_mc = False if n//n_mc < 2 else twin_mc

    return n_mc, twin_mc

def _check_knn_setting(
        n_knn: int,
        approx_knn: bool,
        y: np.ndarray,
    ) -> Tuple[int, bool]:
    # function to check for nearest-neighbor search setting

    # check for n_knn 
    n_knn = (2 if np.unique(y).size > 2 else 3) if n_knn is None else n_knn
    assert isinstance(n_knn, int) and n_knn > 0, f"n_knn {(n_knn)} must be a postive integer."

    # approximate nearest-neighbor search is only supported for data size at least 10,000
    approx_knn = False if y.size < 1e4 else approx_knn

    return n_knn, approx_knn

def _preprocess_input(
        X:pd.DataFrame,
    ) -> np.ndarray:
    # preprocess input by transforming categorical features via one hot encoding 
    # numeric columns include bool, int, float
    p = X.shape[1]
    num_col_ind = [i for i in range(p) if is_numeric_dtype(X.iloc[:,i])]
    cat_col_ind = list(set(range(p)).difference(set(num_col_ind)))
    X = np.hstack([X.iloc[:,num_col_ind].values.astype(np.float32)] + [pd.get_dummies(X.iloc[:,i]).values.astype(np.float32) for i in cat_col_ind])
    return X

def _get_knn(
        data:np.ndarray, 
        query:np.ndarray, 
        k:int = 5, 
        approximate:bool = False,
    ) -> np.ndarray:
    # nearest-neighbor search using faiss
    assert k <= data.shape[0], f"k ({k}) cannot be greater than size of data ({data.shape[0]})."
    data = data.astype(np.float32)
    if approximate:
        nn_quantizer = faiss.IndexFlatL2(data.shape[1])
        nn_engine = faiss.IndexIVFFlat(nn_quantizer, data.shape[1], 100)
        nn_engine.train(data)
        nn_engine.nprobe = 10
    else:
        nn_engine = faiss.IndexFlatL2(data.shape[1])
    nn_engine.add(data)
    _, nn_index = nn_engine.search(query, k)
    return nn_index

def _exp_var_knn(
        X:pd.DataFrame,
        y:np.ndarray,
        subset:List[int],
        factor_nunique:Optional[np.ndarray] = None,
        n_knn:int = 2,
        approx_knn:bool = False,
        n_mc:int = None,
        twin_mc:bool = False,
        random_state:Union[int,np.random.RandomState] = None,
    ) -> float:

    # argument check for random state
    rng = check_random_state(random_state)

    # argument check for subset 
    assert isinstance(subset, List), f"subset must be a list of integer."
    if len(subset) > 0:
        assert isinstance(subset[0], int), f"subset must be a list of integer."
    else:
        return np.var(y, ddof=1)

    # argument check for month carlo subsample setting
    n_mc, twin_mc = _check_mc_setting(n_mc, twin_mc, y)

    # argument check for knn search setting
    n_knn, approx_knn = _check_knn_setting(n_knn, approx_knn, y)

    # check if any row of X duplicate and preprocess categorical features
    n, p = X.shape
    # get the subset 
    X = X.iloc[:,subset].copy()
    # check if any duplicated rows
    row_duplicated = False 
    if factor_nunique is None:
        if X.duplicated().any():
            row_duplicated = True 
    else:
        assert factor_nunique.size == p, f"factor_nunique ({factor_nunique.size}) must be the same size of X ({p})."
        factor_nunique = factor_nunique[subset]
        if (factor_nunique < n).all():
            if factor_nunique.size > 1:
                if X.duplicated().any():
                    row_duplicated = True
            else:
                row_duplicated = True
    # preprocess categorical features
    X = _preprocess_input(X)
    # handle duplicated row for nearest-neighbor search
    if row_duplicated:
        # random jittering to have unqiue rows
        X = np.hstack((X, 1e-3*rng.uniform(low=-1,high=1,size=(n,1))))
        faiss.cvar.distance_compute_blas_threshold = n + 1 # exact computation
    else:
        faiss.cvar.distance_compute_blas_threshold = 20 # default setting

    # nearest-neighbor search and compute expected conditional variance
    if n_mc < n:
        if twin_mc:
            r = n//n_mc
            keep_ind = rng.permutation(np.arange(n))[:(n_mc*r)]
            twin_ind = twin(data=X[keep_ind,:].astype(np.float64), r=r, u1=0)
            query_ind = keep_ind[twin_ind]
        else:
            query_ind = rng.permutation(np.arange(n))[:n_mc]
    else:
        query_ind = np.arange(n)
    nn_index = _get_knn(X, X[query_ind,:], k=n_knn, approximate=approx_knn)
    ev = np.mean(np.var(y[nn_index],ddof=1,axis=1))
        
    return ev

def TotalSobolKNN(
        X:Union[pd.DataFrame, np.ndarray], 
        y:Union[pd.Series, np.ndarray], 
        noise:bool, 
        n_knn:int = None, 
        approx_knn:bool = False, 
        n_mc:int = None, 
        twin_mc:bool = False, 
        rescale:bool = True,
        n_jobs:int = 1,
        random_state:Union[int,np.random.RandomState] = None,
    ) -> np.ndarray:

    """Estimating Total Sobol' Indices from Data 

    `TotalSobolKNN` provides consistent estimation of total Sobol' indices (Sobol', 2001) directly from scattered data. When the responses are noiseless (`noise=False`), it implements the Nearest-Neighbor estimator in Broto et al. (2020). When the responses are noisy (`noise=True`), it implements the Noise-Adjusted Nearest-Neighbor estimator in Huang and Joseph (2024).

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        A pd.DataFrame or np.ndarray for the factors / predictors.
    
    y : pd.Series or np.ndarray
        A pd.Series or np.ndarray for the responses. 
    
    noise : bool
        A logical indicating whether the responses are noisy.
    
    n_knn : int, default=None
        The number of nearest-neighbor for the inner loop conditional variance estimation. `n_knn=2` is recommended for regression, and `n_knn=3` for binary classification.
    
    approx_knn : bool, default=False
        A logical indicating whether to use approximate nearest-neighbor search, otherwise exact search is used. It is supported when there are at least 10,000 data instances.
    
    n_mc : int, default=None 
        The number of Monte Carlo samples for the outer loop expectation estimation.
    
    twin_mc : bool, default=False
        A logical indicating whether to use twinning subsamples, otherwise random subsamples are used. It is supported when the reduction ratio is at least 2. 
    
    rescale : bool, default=True
        A logical indicating whether to standardize the factors / predictors.
    
    n_jobs : int, default=1
        The number of jobs to run in parallel. `n_jobs=-1` means using all processors.
    
    random_state : int or RandomState instance, default=None
        A seed for controlling the randomness in breaking ties in nearest-neighbor search and finding random subsamples.

    Returns
    ----------
    np.ndarray
        A numeric vector for the total Sobol' indices estimation.

    Notes
    -----
    `Faiss` (Douze et al., 2024) is used for efficient nearest-neighbor search, with the approximate search (`approx_knn=True`) by the inverted file index (IVF). IVF reduces the search scope through first clustering data into Voronoi cells. To further accelerate, we also support the use of subsamples by specifying `n_mc`. Both random and twinning (Vakayil and Joseph, 2022) subsamples are available, where twinning subsamples provide better approximation for the full data. 

    References
    ----------
    Huang, C., & Joseph, V. R. (2025). Factor Importance Ranking and Selection using Total Indices. Technometrics.
    
    Sobol', I. M. (2001). Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates. Mathematics and computers in simulation, 55(1-3), 271-280.
    
    Broto, B., Bachoc, F., & Depecker, M. (2020). Variance reduction for estimation of Shapley effects and adaptation to unknown input distribution. SIAM/ASA Journal on Uncertainty Quantification, 8(2), 693-716.
    
    Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G., Mazaré, P.E., Lomeli, M., Hosseini, L., & Jégou, H., (2024). The Faiss library. arXiv preprint arXiv:2401.08281.
    
    Vakayil, A., & Joseph, V. R. (2022). Data twinning. Statistical Analysis and Data Mining: The ASA Data Science Journal, 15(5), 598-610.

    """

    # argument check for random state
    rng = check_random_state(random_state)  

    # check X and y, and preprocess
    X, y = _check_X_y_and_preprocess(X, y, rescale)

    # check Monte Carlo subsample setting
    n_mc, twin_mc = _check_mc_setting(n_mc, twin_mc, y)

    # check knn setting 
    n_knn, approx_knn = _check_knn_setting(n_knn, approx_knn, y)

    # get basic information for factor
    n, p = X.shape
    factor_nunique = X.nunique().to_numpy()
    factor_non_constant = [i for i in range(p) if factor_nunique[i] > 1]
    
    # compute total Sobol' indices
    if noise:
        noise_var = _exp_var_knn(
            X = X,
            y = y,
            subset = factor_non_constant,
            factor_nunique = factor_nunique,
            n_knn = n_knn, 
            approx_knn = approx_knn,
            n_mc = n_mc, 
            twin_mc = twin_mc,
            random_state = rng.randint(1e9, size=1)[0],
        )
    else:
        noise_var = 0
    y_var = max(np.var(y,ddof=1) - noise_var, 0)
    tsi = np.zeros(p)
    if y_var > 0:
        seeds = rng.randint(1e9, size=len(factor_non_constant))
        xe_var = Parallel(n_jobs=n_jobs,prefer='threads')(delayed(_exp_var_knn)(
            X = X,
            y = y,
            subset = factor_non_constant[:i]+factor_non_constant[(i+1):],
            factor_nunique = factor_nunique,
            n_knn = n_knn, 
            approx_knn = approx_knn,
            n_mc = n_mc, 
            twin_mc = twin_mc,
            random_state = seeds[i],
        ) for i in range(len(factor_non_constant)))
        xe_var = np.array(xe_var)
        xi_var = np.maximum(xe_var - noise_var, 0) 
        tsi[factor_non_constant] = xi_var / y_var

    return tsi

def ShapleySobolKNN(
        X:Union[pd.DataFrame, np.ndarray], 
        y:Union[pd.Series, np.ndarray], 
        noise:bool, 
        n_knn:int = None, 
        approx_knn:bool = False, 
        n_mc:int = None, 
        twin_mc:bool = False, 
        rescale:bool = True,
        n_jobs:int = 1,
        random_state:Union[int,np.random.RandomState] = None,
    ) -> np.ndarray:

    """Estimating Shapley Sobol' Effect from Data 

    `ShapleySobolKNN` provides consistent estimation of Shapley Sobol' Effect (Owen, 2014; Song et al., 2016) directly from scattered data. When the responses are noiseless (`noise=False`), it implements the Nearest-Neighbor estimator in Broto et al. (2020). When the responses are noisy (`noise=True`), it implements the Noise-Adjusted Nearest-Neighbor estimator in Huang and Joseph (2024).

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        A pd.DataFrame or np.ndarray for the factors / predictors.
    
    y : pd.Series or np.ndarray
        A pd.Series or np.ndarray for the responses. 
    
    noise : bool
        A logical indicating whether the responses are noisy.
    
    n_knn : int, default=None
        The number of nearest-neighbor for the inner loop conditional variance estimation. `n_knn=2` is recommended for regression, and `n_knn=3` for binary classification.
    
    approx_knn : bool, default=False
        A logical indicating whether to use approximate nearest-neighbor search, otherwise exact search is used. It is supported when there are at least 10,000 data instances.
    
    n_mc : int, default=None 
        The number of Monte Carlo samples for the outer loop expectation estimation.
    
    twin_mc : bool, default=False
        A logical indicating whether to use twinning subsamples, otherwise random subsamples are used. It is supported when the reduction ratio is at least 2. 
    
    rescale : bool, default=True
        A logical indicating whether to standardize the factors / predictors.
    
    n_jobs : int, default=1
        The number of jobs to run in parallel. `n_jobs=-1` means using all processors.
    
    random_state : int or RandomState instance, default=None
        A seed for controlling the randomness in breaking ties in nearest-neighbor search and finding random subsamples.

    Returns
    ----------
    np.ndarray
        A numeric vector for the Shapley Sobol' effect estimation.

    Notes
    -----
    `Faiss` (Douze et al., 2024) is used for efficient nearest-neighbor search, with the approximate search (`approx_knn=True`) by the inverted file index (IVF). IVF reduces the search scope through first clustering data into Voronoi cells. To further accelerate, we also support the use of subsamples by specifying `n_mc`. Both random and twinning (Vakayil and Joseph, 2022) subsamples are available, where twinning subsamples provide better approximation for the full data. 

    References
    ----------
    Huang, C., & Joseph, V. R. (2025). Factor Importance Ranking and Selection using Total Indices. Technometrics.
    
    Owen, A. B. (2014), “Sobol’indices and Shapley value,” SIAM/ASA Journal on Uncertainty Quantification, 2, 245–251.

    Song, E., Nelson, B. L., & Staum, J. (2016), “Shapley effects for global sensitivity analysis: Theory and computation,” SIAM/ASA Journal on Uncertainty Quantification, 4, 1060-1083.
    
    Broto, B., Bachoc, F., & Depecker, M. (2020). Variance reduction for estimation of Shapley effects and adaptation to unknown input distribution. SIAM/ASA Journal on Uncertainty Quantification, 8(2), 693-716.
    
    Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G., Mazaré, P.E., Lomeli, M., Hosseini, L., & Jégou, H., (2024). The Faiss library. arXiv preprint arXiv:2401.08281.
    
    Vakayil, A., & Joseph, V. R. (2022). Data twinning. Statistical Analysis and Data Mining: The ASA Data Science Journal, 15(5), 598-610.

    """

    # argument check for random state
    rng = check_random_state(random_state)  

    # check X and y, and preprocess
    X, y = _check_X_y_and_preprocess(X, y, rescale)

    # check Monte Carlo subsample setting
    n_mc, twin_mc = _check_mc_setting(n_mc, twin_mc, y)

    # check knn setting 
    n_knn, approx_knn = _check_knn_setting(n_knn, approx_knn, y)

    # get basic information for factor
    n, p = X.shape
    factor_nunique = X.nunique().to_numpy()
    factor_non_constant = [i for i in range(p) if factor_nunique[i] > 1]
    
    # compute Shapley Sobol' Effect
    if noise:
        noise_var = _exp_var_knn(
            X = X,
            y = y,
            subset = factor_non_constant,
            factor_nunique = factor_nunique,
            n_knn = n_knn, 
            approx_knn = approx_knn,
            n_mc = n_mc, 
            twin_mc = twin_mc,
            random_state = rng.randint(1e9, size=1)[0],
        )
    else:
        noise_var = 0    
    y_var = np.var(y,ddof=1)
    if y_var < noise_var:
        # noise dominated, no factor is important
        ssi = np.zeros(p) 
    else:
        q = len(factor_non_constant) # number of non-constant factors
        u = [[bool(int(bit)) for bit in format(i, f'0{q}b')[::-1]] for i in range(2**q-1)]
        seeds = rng.randint(1e9, size=2**q-1)
        nx_var = Parallel(n_jobs=n_jobs,prefer='threads')(delayed(_exp_var_knn)(
            X = X,
            y = y,
            subset = [ind for ind, flag in zip(factor_non_constant, u[i]) if flag],
            factor_nunique = factor_nunique,
            n_knn = n_knn, 
            approx_knn = approx_knn,
            n_mc = n_mc, 
            twin_mc = twin_mc,
            random_state = seeds[i],
        ) for i in range(2**q-1))
        nx_var = np.array(nx_var + [noise_var])
        x_var = np.maximum(y_var - nx_var, 0.0)
        ssi_non_constant = np.zeros(q)
        for i in range(q):
            for s in range(0, 2**q, 2**(i+1)):
                for l in range(s, s+2**i):
                    cardu = sum(u[l])
                    ssi_non_constant[i] += (x_var[l+2**i] - x_var[l]) / comb(q-1, cardu)
        ssi_non_constant = ssi_non_constant / q / (y_var - noise_var)
        ssi = np.zeros(p)
        ssi[factor_non_constant] = ssi_non_constant

    return ssi 


def FIRSTRank(
        X:Union[pd.DataFrame, np.ndarray], 
        y:Union[pd.Series, np.ndarray], 
        noise:bool, 
        n_knn:int = None, 
        approx_knn:bool = False, 
        n_mc:int = None, 
        twin_mc:bool = False, 
        rescale:bool = True,
        n_jobs:int = 1,
        random_state:Union[int,np.random.RandomState] = None,
) -> Dict:
    """Provides Factor Ranking via Cumulative Variance that can be explained. 

    `FIRSTRank` provides factor ranking directly from scattered data via cumulative variance that can be explained. See Huang and Joseph (2025) for details.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        A pd.DataFrame or np.ndarray for the factors / predictors.
    
    y : pd.Series or np.ndarray
        A pd.Series or np.ndarray for the responses. 
    
    noise : bool
        A logical indicating whether the responses are noisy.
    
    n_knn : int, default=None
        The number of nearest-neighbor for the inner loop conditional variance estimation. `n_knn=2` is recommended for regression, and `n_knn=3` for binary classification.
    
    approx_knn : bool, default=False
        A logical indicating whether to use approximate nearest-neighbor search, otherwise exact search is used. It is supported when there are at least 10,000 data instances.
    
    n_mc : int, default=None 
        The number of Monte Carlo samples for the outer loop expectation estimation.
    
    twin_mc : bool, default=False
        A logical indicating whether to use twinning subsamples, otherwise random subsamples are used. It is supported when the reduction ratio is at least 2. 
    
    rescale : bool, default=True
        A logical indicating whether to standardize the factors / predictors.
    
    n_jobs : int, default=1
        The number of jobs to run in parallel. `n_jobs=-1` means using all processors.
    
    random_state : int or RandomState instance, default=None
        A seed for controlling the randomness in breaking ties in nearest-neighbor search and finding random subsamples.

    Returns
    ----------
    A dictionary with the following keys:
        - "ranking": np.ndarray
            Indices of the factors, ordered from most to least important, based on  cumulative variance that can be explained.
        - "explained_variance": np.ndarray
            Cumulative variance explained by the factors in the ranking order.

    Notes
    -----
    `Faiss` (Douze et al., 2024) is used for efficient nearest-neighbor search, with the approximate search (`approx_knn=True`) by the inverted file index (IVF). IVF reduces the search scope through first clustering data into Voronoi cells. To further accelerate, we also support the use of subsamples by specifying `n_mc`. Both random and twinning (Vakayil and Joseph, 2022) subsamples are available, where twinning subsamples provide better approximation for the full data. 

    References
    ----------
    Huang, C., & Joseph, V. R. (2025). Factor Importance Ranking and Selection using Total Indices. Technometrics.
    
    Sobol', I. M. (2001). Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates. Mathematics and computers in simulation, 55(1-3), 271-280.
    
    Broto, B., Bachoc, F., & Depecker, M. (2020). Variance reduction for estimation of Shapley effects and adaptation to unknown input distribution. SIAM/ASA Journal on Uncertainty Quantification, 8(2), 693-716.
    
    Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G., Mazaré, P.E., Lomeli, M., Hosseini, L., & Jégou, H., (2024). The Faiss library. arXiv preprint arXiv:2401.08281.
    
    Vakayil, A., & Joseph, V. R. (2022). Data twinning. Statistical Analysis and Data Mining: The ASA Data Science Journal, 15(5), 598-610.

    """

    # argument check for random state
    rng = check_random_state(random_state)  

    # check X and y, and preprocess
    X, y = _check_X_y_and_preprocess(X, y, rescale)

    # check Monte Carlo subsample setting
    n_mc, twin_mc = _check_mc_setting(n_mc, twin_mc, y)

    # check knn setting 
    n_knn, approx_knn = _check_knn_setting(n_knn, approx_knn, y)

    # get basic information for factor
    _, p = X.shape
    factor_nunique = X.nunique().to_numpy()
    factor_non_constant = [i for i in range(p) if factor_nunique[i] > 1]

    # factor importance ranking by the cumulative variance that can be explained
    if noise:
        noise_var = _exp_var_knn(
            X = X,
            y = y,
            subset = factor_non_constant,
            factor_nunique = factor_nunique,
            n_knn = n_knn, 
            approx_knn = approx_knn,
            n_mc = n_mc, 
            twin_mc = twin_mc,
            random_state = rng.randint(1e9, size=1)[0],
        )
    else:
        noise_var = 0
    y_var = max(np.var(y,ddof=1) - noise_var, 0)

    if y_var == 0:
        return {
            "ranking": np.arange(p),
            "explained_variance": np.zeros(p),
        }
    
    factor_ranking = []
    factor_explained_variance = []
    subset = factor_non_constant
    while len(subset) > 0:
        seeds = rng.randint(1e9, size=len(subset))
        nx_var = Parallel(n_jobs=n_jobs,prefer='threads')(delayed(_exp_var_knn)(
            X = X,
            y = y,
            subset = subset[:i]+subset[(i+1):],
            factor_nunique = factor_nunique,
            n_knn = n_knn, 
            approx_knn = approx_knn,
            n_mc = n_mc, 
            twin_mc = twin_mc,
            random_state = seeds[i],
        ) for i in range(len(subset)))
        nx_var = np.array(nx_var)
        x_var = np.maximum(y_var - nx_var, 0)
        remove_ind = subset[np.argmax(x_var)]
        factor_ranking.append(remove_ind)
        factor_explained_variance.append(x_var.max() / y_var)
        subset.remove(remove_ind)
    
    factor_ranking = factor_ranking[::-1]
    factor_explained_variance = factor_explained_variance[::-1][1:] + [1]
    for i in range(p):
        if factor_nunique[i] == 1:
            factor_ranking.append(i)
            factor_explained_variance.append(1)
    
    return {
        "ranking": np.array(factor_ranking),
        "explained_variance": np.array(factor_explained_variance)
    }

def FIRST(
        X:Union[pd.DataFrame, np.ndarray], 
        y:Union[pd.Series, np.ndarray], 
        n_knn:int = None, 
        approx_knn:bool = False, 
        n_mc:int = None, 
        twin_mc:bool = False, 
        rescale:bool = True,
        n_forward:int = 2,
        n_jobs:int = 1,
        random_state:Union[int,np.random.RandomState] = None,
        return_option:str = 'selection',
        verbose:bool = False,
    ) -> np.ndarray:

    """Factor Importance Ranking and Selection using Total (Sobol') indices

    `FIRST` provides factor importance ranking and selection directly from scattered data without any model fitting.

    `FIRST` is a model-independent factor importance ranking and selection algorithm proposed in Huang and Joseph (2024). Factor importance is computed based on total Sobol' indices (Sobol', 2021), which is connected to the approximation error introduced by excluding the factor of interest (Huang and Joseph, 2024). The estimation procedure adapts from the Nearest-Neighbor estimator in Broto et al. (2020) to account for the noisy data. Integrating it with forward selection and backward elimination allows for factor selection.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        A pd.DataFrame or np.ndarray for the factors / predictors.
    
    y : pd.Series or np.ndarray
        A pd.Series or np.ndarray for the responses. 
    
    n_knn : int, default=None
        The number of nearest-neighbor for the inner loop conditional variance estimation. `n_knn=2` is recommended for regression, and `n_knn=3` for binary classification.
    
    approx_knn : bool, default=False
        A logical indicating whether to use approximate nearest-neighbor search, otherwise exact search is used. It is supported when there are at least 10,000 data instances.
    
    n_mc : int, default=None 
        The number of Monte Carlo samples for the outer loop expectation estimation.
    
    twin_mc : bool, default=False
        A logical indicating whether to use twinning subsamples, otherwise random subsamples are used. It is supported when the reduction ratio is at least 2. 
    
    rescale : bool, default=True
        A logical indicating whether to standardize the factors / predictors.
    
    n_forward : int, default=2
        The number of times to run the forward selection phase to tradeoff between efficiency and accuracy. `n_forward=2` is recommended. To run the complete forward selection, please set `n_forward` to the number of factors / predictors. 
    
    n_jobs : int, default=1
        The number of jobs to run in parallel. `n_jobs=-1` means using all processors.
    
    random_state : int or RandomState instance, default=None
        A seed for controlling the randomness in breaking ties in nearest-neighbor search and finding random subsamples.

    return_option : str, default='selection'
        The options for the output of `FIRST`. Default is `selection`, which returns a binary vector indicating the factor selection, with `True` for selected variables and `False` otherwise. Other options include: (i) `total`, which return a numeric vector for total Sobol' effect and (ii) `shapley`, which return a numeric vector for Shapley Sobol' effect. See Huang and Joseph (2024) for details.
    
    verbose : bool, default=False
        A logical indicating whether to display intermediate results, e.g., the selected factor from each iteration.

    Returns
    ----------
    np.ndarray
        A numeric vector for the factor selection, importance, or ranking depending on the value of the `return_option` argument. 

    Notes
    -----
    `FIRST` belongs to the class of forward-backward selection with early dropping algorithm (Borboudakis and Tsamardinos, 2019). In forward selection, each time we find the candidate that maximizes the output variance that can be explained. For candidates that cannot improve the variance explained conditional on the selected factors, they are removed from the candidate set. This forward selection step is run `n_forward` times to tradeoff between accuracy and efficiency. `n_forward = 2` is recommended in Yu et al. (2020). To run the complete forward selection, please set `n_forward` to the number of factors / predictors. In backward elimination, we again remove one factor at a time, starting with the factor that can improve the explained variance most, till no factor can further improve. 

    `Faiss` (Douze et al., 2024) is used for efficient nearest-neighbor search, with the approximate search (`approx_knn=True`) by the inverted file index (IVF). IVF reduces the search scope through first clustering data into Voronoi cells. To further accelerate, we also support the use of subsamples by specifying `n_mc`. Both random and twinning (Vakayil and Joseph, 2022) subsamples are available, where twinning subsamples provide better approximation for the full data. 

    For more details about `FIRST`, please see Huang and Joseph (2024). 

    References
    ----------
    Huang, C., & Joseph, V. R. (2025). Factor Importance Ranking and Selection using Total Indices. Technometrics.
    
    Sobol', I. M. (2001). Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates. Mathematics and computers in simulation, 55(1-3), 271-280.
    
    Broto, B., Bachoc, F., & Depecker, M. (2020). Variance reduction for estimation of Shapley effects and adaptation to unknown input distribution. SIAM/ASA Journal on Uncertainty Quantification, 8(2), 693-716.
    
    Borboudakis, G., & Tsamardinos, I. (2019). Forward-backward selection with early dropping. The Journal of Machine Learning Research, 20(1), 276-314.
    
    Yu, K., Guo, X., Liu, L., Li, J., Wang, H., Ling, Z., & Wu, X. (2020). Causality-based feature selection: Methods and evaluations. ACM Computing Surveys (CSUR), 53(5), 1-36.
    
    Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G., Mazaré, P.E., Lomeli, M., Hosseini, L., & Jégou, H., (2024). The Faiss library. arXiv preprint arXiv:2401.08281.
    
    Vakayil, A., & Joseph, V. R. (2022). Data twinning. Statistical Analysis and Data Mining: The ASA Data Science Journal, 15(5), 598-610.
    
    """

    # argument check for random state
    rng = check_random_state(random_state)

    # check X and y, and preprocess
    X, y = _check_X_y_and_preprocess(X, y, rescale)

    # check Monte Carlo subsample setting
    n_mc, twin_mc = _check_mc_setting(n_mc, twin_mc, y)

    # check knn setting 
    n_knn, approx_knn = _check_knn_setting(n_knn, approx_knn, y)

    # check return option
    assert return_option in ['selection','total','shapley'], f"return_option ({return_option}) is not available, only selection/total/shapley/ranking is supported."

    # get basic information for factor
    n, p = X.shape
    factor_nunique = X.nunique().to_numpy()
    factor_non_constant = [i for i in range(p) if factor_nunique[i] > 1]
    
    # forward selection 
    if verbose:
        print("Starting forward selection...")
        if len(factor_non_constant) < p:
            print(f"factors removed because of constant value: {' '.join(str(i) for i in range(p) if i not in factor_non_constant)}")
    y_var = np.var(y, ddof=1)
    subset = []
    x_var_max = 0
    for t in range(n_forward):
        if verbose: 
            print(f"\nPhase-{(t+1):d} Forward Selection...")
        none_added_to_subset = True
        candidate = [i for i in factor_non_constant if i not in subset]
        while len(candidate) > 0:
            # compute total Sobol' effect for -x (x for current subset)
            seeds = rng.randint(1e9, size=len(candidate))
            nx_var = Parallel(n_jobs=n_jobs,prefer='threads')(delayed(_exp_var_knn)(
                X = X,
                y = y,
                subset = subset + [candidate[i]],
                factor_nunique = factor_nunique,
                n_knn = n_knn, 
                approx_knn = approx_knn,
                n_mc = n_mc, 
                twin_mc = twin_mc,
                random_state = seeds[i],
            ) for i in range(len(candidate)))
            nx_var = np.array(nx_var)
            x_var = np.maximum(y_var - nx_var, 0)
            if verbose:
                print(f"\ncurrent selection: {' '.join(str(s) for s in subset)}")
                print(f"current variance explained: {x_var_max:.3f}")
                print("candidate to add:", ' '.join(f"{candidate[i]:d}({x_var[i]:.3f})" for i in range(len(candidate))))
            if x_var.max() > x_var_max:
                # find the index to add such that the variance explained is maximized
                add_ind = candidate[np.argmax(x_var)]
                candidate = [candidate[i] for i in range(len(candidate)) if x_var[i] > x_var_max]
                candidate.remove(add_ind)
                subset.append(add_ind)
                none_added_to_subset = False
                x_var_max = x_var.max()
                if verbose: 
                    print(f"add candidate {add_ind:d}({x_var_max:.3f}).")
            else:
                break
        if none_added_to_subset:
            if verbose: 
                print("early termination since none of the candidates can be added in this phase.")
            break
    
    # backward elimination
    if verbose:
        print("\nStarting backward elimination...")
    subset.sort()
    while len(subset) > 0:
        # compute total sobol' effect for -(x/{i}) (x for current subset)
        seeds = rng.randint(1e9, size=len(subset))
        nx_var = Parallel(n_jobs=n_jobs,prefer='threads')(delayed(_exp_var_knn)(
            X = X,
            y = y,
            subset = subset[:i]+subset[(i+1):],
            factor_nunique = factor_nunique,
            n_knn = n_knn, 
            approx_knn = approx_knn,
            n_mc = n_mc, 
            twin_mc = twin_mc,
            random_state = seeds[i],
        ) for i in range(len(subset)))
        nx_var = np.array(nx_var)
        x_var = np.maximum(y_var - nx_var, 0)
        if verbose:
            print(f"\ncurrent selection: {' '.join(str(s) for s in subset)}")
            print(f"current variance explained: {x_var_max:.3f}")
            print("candidate to remove:", ' '.join(f"{subset[i]:d}({x_var[i]:.3f})" for i in range(len(subset))))
        if x_var.max() >= x_var_max:
            # find the index to remove such that the variance explained is maximized
            remove_ind = subset[np.argmax(x_var)]
            subset.remove(remove_ind)
            x_var_max = x_var.max()
            if verbose: 
                print(f"remove candidate {remove_ind:d}({x_var_max:.3f}).") 
        else:
            break

    # compute importance via total Sobol' indices
    imp = np.zeros(p)
    if len(subset) > 0:
        noise_var = y_var - x_var_max
        imp[subset] = (nx_var - noise_var) / x_var_max
    # get the indices for selection
    selection = (imp > 0).astype(bool)

    if return_option == "selection":
        return selection 
    elif return_option == "total":
        return imp 
    elif return_option == "shapley":
        shapley_imp = np.zeros(p)
        shapley_imp[selection] = ShapleySobolKNN(
            X = X.loc[:,selection].copy(),
            y = y,
            noise = True,
            n_knn = n_knn,
            approx_knn = approx_knn,
            n_mc = n_mc,
            twin_mc = twin_mc,
            rescale = rescale,
            n_jobs = n_jobs,
            random_state = random_state,
        )
        return shapley_imp 
    else:
        return None

class SelectByFIRST(SelectorMixin, BaseEstimator):
    
    """Feature selector using FIRST

    This implements the feature selector class for FIRST (Huang and Joseph, 2024), a model-independent feature selection algorithm based on total Sobol' indices (Sobol', 2001). 

    Parameters
    ----------
    regression : bool, default=True
        A logical indicating whether the feature selector is for a regression or classification problem. 
    
    n_knn : int, default=None
        The number of nearest-neighbor for the inner loop conditional variance estimation. `n_knn=2` is recommended for regression, and `n_knn=3` for binary classification.
    
    approx_knn : bool, default=False
        A logical indicating whether to use approximate nearest-neighbor search, otherwise exact search is used. It is supported when there are at least 10,000 data instances.
    
    rescale : bool, default=True
        A logical indicating whether to standardize the factors / predictors.
    
    n_forward : int, default=2
        The number of times to run the forward selection phase to tradeoff between efficiency and accuracy. `n_forward=2` is recommended.
    
    n_jobs : int, default=1
        The number of jobs to run in parallel. `n_jobs=-1` means using all processors.
    
    random_state : int or RandomState instance, default=None
        A seed for controlling the randomness in breaking ties in nearest-neighbor search and finding random subsamples.

    importance_option : str, default='total'
        The options for the feature importance. Default is `total`, which returns a numeric vector for total Sobol' effect and `shapley` for Shapley Sobol' effect. See Huang and Joseph (2024) for details.
    
    verbose : default = False
        A logical indicating whether to display intermediate results, e.g., the selected factor from each iteration.
    
    Attributes
    ----------
    importance_ : np.ndarray
        Factor importance, with zero indicating that the factor is not important for predicting the response.

    Notes
    -----
    FIRST belongs to the class of forward-backward selection with early dropping algorithm (Borboudakis and Tsamardinos, 2019). In forward selection, each time we find the candidate that maximizes the output variance that can be explained. For candidates that cannot improve the variance explained conditional on the selected factors, they are removed from the candidate set. This forward selection step is run `n_forward` times to tradeoff between accuracy and efficiency. `n_forward = 2` is recommended in (Yu et al., 2020). In backward elimination, we again remove one factor at a time, starting with the factor that can improve the explained variance most, till no factor can further improve. 

    The estimation of the importance is via an adaptation of the Nearest-Neighbor estimator of Broto et al. (2020) for the total Sobol' indices. `Faiss` (Douze et al., 2024) is used for efficient nearest-neighbor search, with the approximate search (`approx_knn=True`) by the inverted file index (IVF). IVF reduces the search scope through first clustering data into Voronoi cells.

    For more details about FIRST, please see Huang and Joseph (2024). 

    References
    ----------
    Huang, C., & Joseph, V. R. (2025). Factor Importance Ranking and Selection using Total Indices. Technometrics.
    
    Sobol', I. M. (2001). Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates. Mathematics and computers in simulation, 55(1-3), 271-280.
    
    Broto, B., Bachoc, F., & Depecker, M. (2020). Variance reduction for estimation of Shapley effects and adaptation to unknown input distribution. SIAM/ASA Journal on Uncertainty Quantification, 8(2), 693-716.
    
    Borboudakis, G., & Tsamardinos, I. (2019). Forward-backward selection with early dropping. The Journal of Machine Learning Research, 20(1), 276-314.
    
    Yu, K., Guo, X., Liu, L., Li, J., Wang, H., Ling, Z., & Wu, X. (2020). Causality-based feature selection: Methods and evaluations. ACM Computing Surveys (CSUR), 53(5), 1-36.
    
    Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G., Mazaré, P.E., Lomeli, M., Hosseini, L., & Jégou, H., (2024). The Faiss library. arXiv preprint arXiv:2401.08281.

    """

    def __init__(
            self, 
            regression:bool = True,
            n_knn:int = None,
            approx_knn:bool = False,
            rescale:bool = True,
            n_forward:int = 2,
            n_jobs:int = 1,
            random_state:Union[int,np.random.RandomState] = None,
            importance_option:str = 'total',
            verbose:bool = False,
        ):
        
        self.regression = regression
        self.n_knn = (2 if regression else 3) if n_knn is None else n_knn
        self.approx_knn = approx_knn
        self.rescale = rescale
        self.n_forward = n_forward
        self.n_jobs = n_jobs
        self.random_state = check_random_state(random_state)
        self.verbose = verbose
        self.importance_option = importance_option
    
    def fit(
            self, 
            X:Union[pd.DataFrame, np.ndarray], 
            y:Union[pd.Series, np.ndarray],
            n_mc:int = None, 
            twin_mc:bool = False, 
        ):

        """Compute the factor importance from data

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            A pd.DataFrame or np.ndarray for the factors / predictors.
        
        y : pd.Series or np.ndarray
            A pd.Series or np.ndarray for the responses. 
        
        n_mc : int, default=None 
            The number of Monte Carlo samples for the outer loop expectation estimation.
        
        twin_mc : bool, default=False
            A logical indicating whether to use twinning subsamples, otherwise random subsamples are used. It is supported when the reduction ratio is at least 2. 

        Returns
        -------
        object
            Returns the instance itself.

        Notes
        -----
        To further accelerate the importance computation, we support the use of subsamples by specifying `n_mc`. Both random and twinning (Vakayil and Joseph, 2022) subsamples are available, where twinning subsamples provide better approximation for the full data. 

        References
        ----------
        Vakayil, A., & Joseph, V. R. (2022). Data twinning. Statistical Analysis and Data Mining: The ASA Data Science Journal, 15(5), 598-610.

        """

        if not self.regression:
            assert np.unique(y).size == 2, f"Only binary classification is supported by FIRST."

        self.importance_ = FIRST(
            X = X,
            y = y,
            n_knn = self.n_knn, 
            approx_knn = self.approx_knn,
            n_mc = n_mc, 
            twin_mc = twin_mc, 
            rescale = self.rescale,
            n_forward = self.n_forward,
            n_jobs = self.n_jobs,
            random_state = self.random_state, 
            return_option = self.importance_option,
            verbose = self.verbose,
        )

        return self
    
    def get_feature_importance(self):

        """Get the feature importance
        
        Returns
        -------
        np.ndarray
            A numeric vector for the factor importance, with zero indicating that the factor is not important for predicting the response.
        """

        check_is_fitted(self)

        return self.importance_

    def _get_support_mask(self):

        """Get the boolean mask indicating which features are selected

        Returns
        -------
        np.ndarray
            A boolean vector with True indicating the feature is selected.
        """

        check_is_fitted(self)
        
        return self.importance_ > 0   