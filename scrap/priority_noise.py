from pyanp.priority import *
import numpy as np
from scipy.stats.mstats import gmean
a = np.array([
    [1, 2, 5],
    [1/2, 1, 3],
    [1/5, 1/3, 1]
])

print(pri_eigen(a))
vals = [
    [2, 5/3],
    [2*3, 5],
    [3, 5/2]
]

means = [gmean(row) for row in vals]
b = utmrowlist_to_npmatrix(means)
print(b)
print(means)
print(incon_std(b))

means = [np.mean(row) for row in vals]
b = utmrowlist_to_npmatrix(means)
print(b)
print(means)
print(incon_std(b))