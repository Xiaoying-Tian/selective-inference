import numpy as np
from sklearn.linear_model import Lasso
from sklearn.cross_validation import KFold

from selection.algorithms.lasso import instance
import selection.sampling.randomized.api as randomized
import selection.constraints.affine as affine

class Error:
    def __init__(self, s=5, n=100, p=50, scale=0.5, lam_frac=1):
        self.s, self.scale = s, scale
        self.X, self.y, _, _, self.sigma = instance(n=n, p=p, s=s, sigma=1.)
        sigma_star = np.sqrt(1 + scale**2) * self.sigma
        self.lam = sigma_star * lam_frac * np.mean(
                np.fabs(np.dot(self.X.T, np.random.standard_normal((n, 10000)))).max(0))
        self.y_star = self.y + np.random.standard_normal(n)*scale*self.sigma
        self.L = Lasso(alpha=self.lam/n, fit_intercept=False)
        self.L.fit(self.X, self.y_star)

    def cv(self, n_folds=10):
        k_fold = KFold(self.X.shape[0], n_folds = n_folds)
        error = 0
        n, p = self.X.shape
        for train, test in k_fold:
            X_train, y_train = self.X[train,:], self.y[train]
            X_test, y_test = self.X[test,:], self.y[test]
            ls = Lasso(alpha=self.lam/n, fit_intercept=False)
            ls.fit(X_train, y_train)
            error += avg_pe(y_test, X_test.dot(ls.coef_)) / n_folds
        return error

    def rand(self, nsim=100):
        error = 0
        n, p = self.X.shape
        for _ in range(nsim): 
            rvs = np.random.standard_normal(n)*self.sigma*self.scale
            ls = Lasso(alpha=self.lam/n, fit_intercept=False)
            ls.fit(self.X, self.y+rvs)
            error += avg_pe(self.y - 1./(self.scale**2) * rvs, self.X.dot(ls.coef_)) - \
                    (self.sigma**2) / (self.scale**2) 
        return error/nsim

    def var(self, ndraw=10000):
        n, p = self.X.shape
        X, y, lam = self.X, self.y, self.lam
        active = (self.L.coef_ != 0)
        nactive = active.sum()
        signs = np.sign(self.L.coef_[active])
        X_E, X_notE = X[:, active], X[:, ~active]
        X_Einv = np.linalg.pinv(X_E)
        Q_Einv = np.linalg.pinv(np.dot(X_E.T, X_E))
        R_E = np.identity(n) - X_E.dot(X_Einv)
        linear_part = np.zeros((2*p-nactive, n))
        offset = np.zeros(2*p-nactive)
        linear_part[:nactive, :] = -np.diag(signs).dot(X_Einv)
        linear_part[nactive:p, :] = np.dot(X_notE.T, R_E)
        linear_part[p:, :] = -np.dot(X_notE.T, R_E)
        offset[:nactive] = np.diag(signs).dot(X_Einv).dot(y) - \
            lam*signs*np.dot(Q_Einv, signs)
        offset[nactive:p] = lam - np.dot(X_notE.T, R_E).dot(y) - \
            lam * np.dot(X_notE.T, np.dot(X_Einv.T, signs))
        offset[p:] = lam + np.dot(X_notE.T, R_E).dot(y) + \
            lam * np.dot(X_notE.T, np.dot(X_Einv.T, signs))
        con = affine.constraints(linear_part, offset,
                                 (self.scale*self.sigma)**2 * np.identity(n))
        #print (linear_part.dot(y_star - y) <= offset).all()
        S = affine.sample_from_constraints(con, self.y_star-y, ndraw=ndraw)
        pe = avg_pe(y - 1./(self.scale**2) * S, X.dot(self.L.coef_)) - \
                (self.sigma**2) / (self.scale**2)
        return pe
    
    def unbiased(self, nsim=10):
        n, p = self.X.shape
        beta0 = np.zeros(p)
        beta0[:self.s] = 7
        y_new = self.X.dot(beta0) + np.random.standard_normal((nsim,n)) * self.sigma
        return avg_pe(y_new, self.X.dot(self.L.coef_))

def avg_pe(y, y_hat):
    return ((y-y_hat)**2).mean()

for _ in range(10):
    errors = Error()
    print "true: ", errors.unbiased(), " cv: ", errors.cv(), " rand: ", errors.rand(), \
            " variables: ", errors.var()

