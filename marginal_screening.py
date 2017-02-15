import numpy as np, pandas as pd
from selection.constraints.affine import constraints, stack
#from selection.tests.instance import gaussian_instance
from scipy.stats import norm, t as tdist

def marginal_screening(X, Y, sigma, thresh=3):

    n, p = X.shape

    diagX = np.sqrt((X**2).sum(0))
    # A hack, treat Y_norm as constant since it has relatively small variance.
    Y_norm = np.linalg.norm(Y - Y.mean())
    # Z is the correlations, use Pearson's correlation test
    Z = X.T.dot(Y - Y.mean()) / (diagX * Y_norm)
    signZ = np.sign(Z)
    above_thresh = np.abs(Z) > thresh

    # surviving the threshold

    cons = []

    if above_thresh.sum():

        A_above = -X[:,above_thresh].T / (diagX[above_thresh] * Y_norm * signZ[above_thresh])[:,None]
        b_above = -np.ones(above_thresh.sum()) * thresh

        above_con = constraints(A_above,
                                b_above)
        cons.append(above_con)

    # below threshold

    if (~above_thresh).sum():

        A_below = np.vstack([X[:,~above_thresh].T / (diagX[~above_thresh] * Y_norm)[:,None],
                             -X[:,~above_thresh].T / (diagX[~above_thresh] * Y_norm)[:,None]])
        b_below = np.ones(2*(~above_thresh).sum()) * thresh

        below_con = constraints(A_below,
                                b_below)
        cons.append(below_con)

    con = stack(*cons)
    con.covariance = sigma**2 * np.identity(n)

    # check

    con(Y)

    # form a model

    if above_thresh.sum():

        X_E = X[:,above_thresh]
        X_Ei = np.linalg.pinv(X_E)

        pvalues = []
        intervals = []
        above_thresh_set = np.nonzero(above_thresh)[0]
        for j, v in enumerate(above_thresh_set):
            pvalues.append(con.pivot(X_Ei[j], Y, alternative='twosided'))
            #intervals.append(con.interval(X_Ei[j], Y))

        #intervals = np.array(intervals)

        df = pd.DataFrame({'pvalue':pvalues},
                           #'lower':intervals[:,0],
                           #'upper':intervals[:,1]},
                           index=above_thresh_set)
        return df

def test():
    n, p = 2000, 20000
    X, y, _, sigma = instance(n=2000, p=20000, s=50, snr=0.1, sigma=10.) 
    bonferroni = get_correlation_cutoff(n, 5./p) 
    return marginal_screening(X, y, sigma, thresh=bonferroni)

def get_correlation_cutoff(n, pval):
    # using Student t-distribution to test Pearson's correlation test
    t = -tdist.ppf(pval/2, n-2)
    return t/np.sqrt(n-2+t**2)
                                
def instance(n=100, p=200, s=10, snr=0.3, sigma=1.):
    X = np.random.standard_normal((n,p))
    X -= X.mean(0)
    beta = np.zeros(p)
    beta[:s] = snr * sigma
    y = np.dot(X, beta) + np.random.standard_normal(n) * sigma

    return X, y, beta, sigma

def test_bonferroni():
    n, p = 2000, 20000
    select_num = []
    false_discovery = []
    y_norm = [] 
    for _ in range(20):
        X, y, _, sigma = instance(n=2000, p=20000, s=50, snr=0.1, sigma=10.) 
        diagX = np.sqrt((X**2).sum(0))
        Z = X.T.dot(y - y.mean()) / (diagX * np.linalg.norm(y - y.mean()))
        bonferroni = get_correlation_cutoff(n, 0.05/p) 
        select_num.append(sum(np.abs(Z) > bonferroni))
        false_discovery.append(sum(np.abs(Z[50:]) > bonferroni))
        y_norm.append(np.linalg.norm(y - y.mean()))

    return select_num, false_discovery, y_norm
