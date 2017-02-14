import numpy as np, pandas as pd
from selection.constraints.affine import constraints, stack
#from selection.tests.instance import gaussian_instance
from scipy.stats import norm

def marginal_screening(X, Y, sigma, thresh=3):

    n, p = X.shape

    diagX = np.sqrt((X**2).sum(0))
    Z = X.T.dot(Y) / (diagX * sigma)
    signZ = np.sign(Z)
    above_thresh = np.abs(Z) > thresh

    # surviving the threshold

    cons = []

    if above_thresh.sum():

        A_above = -X[:,above_thresh].T / (diagX[above_thresh] * sigma * signZ[above_thresh])[:,None]
        b_above = -np.ones(above_thresh.sum()) * thresh

        above_con = constraints(A_above,
                                b_above)
        cons.append(above_con)

    # below threshold

    if (~above_thresh).sum():

        A_below = np.vstack([X[:,~above_thresh].T / (diagX[~above_thresh] * sigma)[:,None],
                             -X[:,~above_thresh].T / (diagX[~above_thresh] * sigma)[:,None]])
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
    X, y, beta, sigma = instance(n=2000, p=20000, s=50, snr=0.1, sigma=10.) 
    bonferroni = norm.ppf(1-0.025/20000)
    print X.shape, y.shape, beta.shape, np.where(beta)[0], bonferroni
    return marginal_screening(X, y, sigma, thresh=bonferroni)
                                
def instance(n=100, p=200, s=10, snr=0.3, sigma=1.):
    X = np.random.standard_normal((n,p))
    beta = np.zeros(p)
    beta[:s] = snr * sigma
    y = np.dot(X, beta) + np.random.standard_normal(n) * sigma

    return X, y, beta, sigma

