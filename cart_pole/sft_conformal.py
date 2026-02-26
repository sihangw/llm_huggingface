import numpy as np

class SplitConformalRegressor:
    """
    Wraps a trained point predictor f(x) and produces conformal prediction intervals.

    predictor: object with predict(X)->(N,) array (or callable)
    alpha: miscoverage level (0.1 -> 90% intervals)
    clip: optional bounds for interval endpoints, e.g. (0,1)
    """
    def __init__(self, predictor, alpha: float = 0.1, clip=None):
        self.predictor = predictor
        self.alpha = float(alpha)
        self.clip = clip
        self.q_ = None
        self.cal_residuals_ = None

    def _predict(self, X):
        if callable(self.predictor):
            yhat = self.predictor(X)
        else:
            yhat = self.predictor.predict(X)
        yhat = np.asarray(yhat).reshape(-1)
        return yhat

    def fit_calibration(self, X_cal, y_cal):
        y_cal = np.asarray(y_cal).reshape(-1)
        yhat_cal = self._predict(X_cal)
        res = np.abs(y_cal - yhat_cal)
        self.cal_residuals_ = res

        n = res.shape[0]
        # conformal quantile index: ceil((n+1)*(1-alpha)) / n
        k = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        k = min(max(k, 1), n)  # clamp
        self.q_ = np.partition(res, k - 1)[k - 1]
        return self

    def predict_interval(self, X, return_pred=True):
        if self.q_ is None:
            raise RuntimeError("Call fit_calibration() first.")

        yhat = self._predict(X)
        lo = yhat - self.q_
        hi = yhat + self.q_

        if self.clip is not None:
            lo = np.clip(lo, self.clip[0], self.clip[1])
            hi = np.clip(hi, self.clip[0], self.clip[1])

        if return_pred:
            if self.clip is not None:
                yhat = np.clip(yhat, self.clip[0], self.clip[1])
            return yhat, lo, hi
        return lo, hi

    def empirical_coverage(self, X_test, y_test):
        y_test = np.asarray(y_test).reshape(-1)
        _, lo, hi = self.predict_interval(X_test, return_pred=True)
        return float(np.mean((y_test >= lo) & (y_test <= hi)))
