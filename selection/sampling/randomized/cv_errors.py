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
        n, p = self.X.shape
        k_fold = KFold(n, n_folds = n_folds)
        error = 0
        for train, test in k_fold:
            X_train, y_train = self.X[train,:], self.y[train]
            X_test, y_test = self.X[test,:], self.y[test]
            ls = Lasso(alpha=self.lam/n, fit_intercept=False)
            ls.fit(X_train, y_train)
            error += avg_pe(y_test, X_test.dot(ls.coef_)) / n_folds
        return error

    def rand(self, nsim=50):
        n, p = self.X.shape
        error = 0
        for _ in range(nsim): 
            rvs = np.random.standard_normal(n)*self.sigma*self.scale
            ls = Lasso(alpha=self.lam/n, fit_intercept=False)
            ls.fit(self.X, self.y+rvs)
            error += avg_pe(self.y - 1./(self.scale**2) * rvs, self.X.dot(ls.coef_)) - \
                    (self.sigma**2) / (self.scale**2) 
        return error/nsim

    def form_constraint(self):
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
        self.con = affine.constraints(linear_part, offset,
                                 (self.scale*self.sigma)**2 * np.identity(n))
        #print (linear_part.dot(y_star - y) <= offset).all()


    def var(self, ndraw=10000):
        n, p = self.X.shape
        self.form_constraint()
        S = affine.sample_from_constraints(self.con, self.y_star-self.y, ndraw=ndraw)
        error = 0
        period = 20
        for i in range(ndraw):
            if i % period  == 0:
                rvs = S[i,:].reshape(n)
                ls = Lasso(self.lam/n, fit_intercept=False)
                ls.fit(self.X, self.y+rvs)
                pe = avg_pe(self.y - 1./(self.scale**2) * rvs, self.X.dot(ls.coef_)) - \
                        (self.sigma**2) / (self.scale**2)
                error += pe / (ndraw/period)
        return error 
    
    def unbiased(self, nsim=100):
        n, p = self.X.shape
        beta0 = np.zeros(p)
        beta0[:self.s] = 7
        error = 0
        for _ in range(nsim):
            rvs = np.random.standard_normal(n)*self.sigma*self.scale
            ls = Lasso(self.lam/n, fit_intercept=False)
            ls.fit(self.X, self.y+rvs)
            y_new = self.X.dot(beta0) + np.random.standard_normal((nsim,n)) * self.sigma
            error += avg_pe(y_new, self.X.dot(ls.coef_)) / nsim
        return error

def avg_pe(y, y_hat):
    return ((y-y_hat)**2).mean()

for _ in range(20):
    errors = Error()
    avg = []
    print "true: ", errors.unbiased(), " cv: ", errors.cv(), " rand: ", errors.rand(), \
            " variables: ", errors.var()
    avg.append([errors.unbiased(), errors.cv(), errors.rand(), errors.var()]) 
print np.array(avg).mean(0)
