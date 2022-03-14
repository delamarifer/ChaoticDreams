import numpy as np
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR

def compute_VAR(window_data, unit_indices=None, PCA_dim=-1):
    if unit_indices is None:
        chunk = window_data
    else:
        chunk = window_data[:, unit_indices]
    k = chunk.shape[0]

    results = {}
    results['explained_variance'] = None
    if PCA_dim > 0:
        if PCA_dim < 2:
            raise ValueError(f"PCA dimension must be greater than 1; provided value was {PCA_dim}")
        pca = PCA(n_components=PCA_dim)
        chunk = pca.fit_transform(chunk)
        results['explained_variance'] = pca.explained_variance_ratio_
    
    model = VAR(chunk)
    VAR_results = model.fit(1)
    results['A_mat'] = VAR_results.coefs[0]
    results['A_mat_with_bias'] = VAR_results.params
    e,_ = np.linalg.eig(VAR_results.coefs[0])   
    results['eigs'] = e   
    results['criticality_inds'] = np.abs(e)

    results['sigma2_ML'] = np.linalg.norm(VAR_results.endog[1:] - (VAR_results.endog_lagged @ VAR_results.params), axis=1).sum()/(k - 2)
    results['AIC'] = k*np.log(results['sigma2_ML']) + 2
    results['sigma_norm'] = np.linalg.norm(VAR_results.sigma_u, ord=2)

    return results