import numpy as np
from scipy.ndimage.filters import convolve
from scipy.special import gamma
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
from scipy import stats


def mae(y, f):
    return mean_absolute_error(y, f)


def rmse(y, f):
    rmse = sqrt(mean_squared_error(y, f))
    return rmse


def mse(y, f):
    mse = mean_squared_error(y, f)
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def r2(y, f):
    return r2_score(y, f)
