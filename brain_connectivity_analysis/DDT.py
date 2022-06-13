import numpy as np
import scipy as sp
from scipy import stats

import argparse
import copy
import pickle

def create_arg_parser():
    parser = argparse.ArgumentParser(prog=__file__, description="""Difference Degree Test (DDT)""", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return parser

def add_arguments(parser):
    parser.add_argument('-m', '--method', type=str, required=False, help='Method to use to select the threshold. Should be "aDDT" or "eDDT"')
    parser.add_argument('-U', '--number', type=int, required=False, help='Number of null networks to generate')
    
    return parser
    
def aDDT(m, μ, σsq):
    """
    Compute a threshold from the 95th quantile of the theoretical critical value based on the parametric mixture distribution defined by: .. math:: H = \frac{2*\\sigma^2}{4}T - \frac{2*\\sigma^2}{4}Q 

    Parameters
    ----------
    m : int
        degree of freedom.
    μ : float
        first moment of the observed difference network.
    σsq : float
        second moment of the observed difference network.

    Returns
    -------
    thresh : float
        threshold.

    """
    df = m
    ncp = m * (4 * (μ ** 2)) / (2 * σsq) # non-centrality parameter
    mcon = 2 * σsq / 4 # constant
    H = mcon * sp.stats.ncx2.rvs(df, ncp, size=1000000) - mcon * sp.stats.chi2.rvs(df, size=1000000) # mixture distribution

    ll = np.quantile(H, .975) # theoretical critical value
    thresh = sp.special.expit(ll)
    
    return thresh

def eDDT(n, m, μ, σsq, U):
    """
    Compute a threshold from the 95th quantile of the empirical critical value based on the empirical distribution.

    Parameters
    ----------
    n : int
        number of regions
    m : int
        max(2, \\(e^2 - \bar{e}^2) / \bar{v})
    μ : float
        first moment of the observed difference network.
    σsq : float
        second moment of the observed difference network.
    U : int
        number of null networks.

    Returns
    -------
    thresh : float
        threshold.
        
    Notes
    -----
    For details in the procedure, see Hirschberger-Qi-Steuer algorithm.
    Section 6.1 of paper: Hirschberger, Markus & Qi, Yue & Steuer, Ralph. (2007). Randomly Generating PortfolioSelection Covariance Matrices with Specified Distributional Characteristics. European Journal of Operational Research. 177. 

    """
    C = np.zeros((n, n, U))
    null = np.zeros((n, n ,U))
    quant = np.zeros((U, ))
    for i in range(U):
        l = μ + np.sqrt(σsq) * np.random.normal(size=(n, m)) # standard normally distributed variables
        C[:, :, i] = l @ l.T
        null[:, :, i] = sp.special.expit(C[:, :, i])
        quant[i] = np.percentile(C[:, :, i][np.triu_indices(n, 1)], 97.5) # empirical critical value
        
    thresh = np.exp(np.max(quant)) / (1 + np.exp(np.max(quant)))
    
    return thresh

def DDT(x, y, method='aDDT', U=1000):
    """
    Difference degree test. 

    Parameters
    ----------
    x : np.ndarray
        connectivity matrices of patients
    y : np.ndarray
        connectivity matrices of controls
    method : str, optional
        method to use to select the threshold. The default is 'aDDT'.
    U : int, optional
        number of networks to generate. The default is 2.

    Returns
    -------
    d_obs_pvalued : np.ndarray
        differentially connected nodes where the differentially weighted edges are greater or equal than 3
        
    Notes
    -----
    See details of the method in: Higgins IA, Kundu S, Choi KS, Mayberg HS, Guo Y. A difference degree test for comparing brain networks. Hum Brain Mapp. 2019 Oct 15;40(15):4518-4536. doi: 10.1002/hbm.24718. Epub 2019 Jul 26

    """
    n = len(x)
    
    # 1. Difference network 
    _, u_pvalue = sp.stats.mannwhitneyu(x, y, axis=-1)
    D = 1 - u_pvalue
    np.fill_diagonal(D, 0)
    
    # 2. First and second moments
    D_bar = sp.special.logit(D)
    # D_bar[np.diag_indices(n)[0], np.diag_indices(n)[1]] = np.diag(D) # to check
    
    ## Convert -inf and +inf to random values
    idxinf = np.argwhere(D_bar <= -12)
    idxsup = np.argwhere(D_bar >= 12)
    neg = -12 - 1 * np.random.rand(1)
    pos = 12 + 1 * np.random.rand(1)
    D_bar[idxinf[:, 0], idxinf[:, 1]] = neg
    D_bar[idxsup[:, 0], idxsup[:, 1]] = pos
    
    e_bar = np.mean(D_bar[np.triu_indices(n, 1)]) # mean of off-diagonal elements
    v_bar = np.var(D_bar[np.triu_indices(n, 1)]) # variance of off-diagonal elements
    e = np.mean(np.diag(D_bar))
    m = max(2, np.floor((e_bar ** 2 - e ** 2) / v_bar)) # if min (like in paper), returns negative value
    μ = np.sqrt(e_bar/m)
    σsq = -(μ ** 2) + np.sqrt(μ ** 4 + (v_bar / m))
    
    # 3. Generate U null Difference Networks
    C = np.zeros((n, n, U))
    null = np.zeros((n, n ,U))
    
    for i in range(U):
        l = μ + np.sqrt(σsq) * np.random.normal(size=(n, m))
        C[:, :, i] = l @ l.T
        null[:, :, i] = sp.special.expit(C[:, :, i])
        
    for i in range(U):
        print("Number {} network \n-----------------".format(i+1))   
        print("mean of off-diagonal elements: {:,.2f}, expected value: {:,.2f}".format(np.mean(C[:, :, i][np.triu_indices(n, 1)]), e_bar))
        print("variance of off-diagonal elements: {:,.2f}, expected value: {:,.2f}".format(np.var(C[:, :, i][np.triu_indices(n, 1)]), v_bar))
        print("mean of diagonal elements: {:,.2f}, expected value: {:,.2f} \n".format(np.mean(np.diag(C[:, :, i])), e))
    
    # 4. Adaptive threshold  
    thresh = aDDT(m, μ, σsq) if method=='aDDT' else eDDT(n, m, μ, σsq, U)
    
    # 5. Apply threshold 
    γ = sp.special.logit(thresh)
    A = np.where(D_bar > γ, 1, 0)
    d_obs = A @ np.ones(n)
    
    # 6. Generate null distribution for di
    sum_A_thresh = np.zeros((n, ))
    for u in range(U):
        A_null_thresh = np.where(null[:, :, u] >= thresh, 1, 0)
        sum_A_thresh = sum_A_thresh + A_null_thresh @ np.ones(n)
    p_null = (1 / (U * (n - 1))) * sum_A_thresh
    
    d_null = np.random.binomial(n-1, p_null)
    
    # 7. Assess the statistical significance of the number of DWE at each node
    result = np.where(d_obs > d_null, 1, 0)
    
    # Binomial probability density function under the null
    p_obs=sp.stats.binom.pmf(d_obs, n-1, p_null)
    # Select regions where the DWE is statistically significant
    p_obs[result == 0] = 1
    # p values that are greater or equal than 0.05 are discarded
    pvalue_DDT = copy.deepcopy(p_obs)
    pvalue_DDT[pvalue_DDT >= 0.05] = 1
    # Discard regions where the corresponding pvalue is greater or equal than 0.05
    d_obs_pvalued = copy.deepcopy(d_obs)
    d_obs_pvalued[pvalue_DDT == 1] = 0
    # Discard regions incident to less than 3 DWE
    # d_obs_pvalued[d_obs_pvalued < 3] = 0
    
    return d_obs_pvalued

if __name__ == '__main__':
    with open('../manage_data/connection_analysis.pickle', 'rb') as f:
        x, y = pickle.load(f)
        
    parser = create_arg_parser()
    add_arguments(parser)
    args = parser.parse_args()
    
    res = DDT(x, y, args.method, args.number)
    print("Differentially connected nodes for each region: ", res)