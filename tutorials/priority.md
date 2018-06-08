# Priority calculations Tutorial

This tutorial covers:

1. Importing the necessary libraries
2. Loading a matrix from a spreadsheet or directly inputting
3. Calculating the standard largest eigenvector priority, eigenvalue, and inconsistency
4. New priority calculations
5. Further references
6. Jupyter notebook and references for this tutorial

## 1. Importing the necessary libraries
The library you need is `pyanp.priority`, but we could also make use of `numpy` and `pandas` so we will import those as well.

```python
# Pandas has DataFrames and Series, very useful things
import pandas as pd
# numpy has lots of useful things in it
import numpy as np
# lastly import our ahptree python code.  If you haven't already installed the pyanp library do
# pip install pyanp
# to get it
from pyanp import priority
```

## 2. Loading a matrix from a spreadhseet or directly inputting

To load from a CSV or Excel (it is the same function), with or without headers
```python
mat3 = priority.get_matrix("pairwise3x3-1.csv")
#this gives the same matrix but with headers
mat3 = priority.get_matrix("pairwise3x3-1-headers.csv")
```
To directly input the matrix
```python
mat4 = np.array([
    [1, 2, 3, 4],
    [1/2, 1, 5, 6],
    [1/3, 1/5, 1, 7],
    [1/4, 1/6, 1/7, 1]
])
```

## 3. Calculating the largest eigenvector priority, eigenvalue, and inconsistency
```python
priority.pri_eigen(mat3)
```
result is:
```
array([0.5816, 0.309 , 0.1095])
```
Now let's calculate the eigenvalue
```python
priority.pri_eigen(mat3, return_eigenval=True)
```
result is:
```
3.0036945980662293
```
And finally calculate the inconsistency
```python
priority.incon_std(mat3)
```
the result is:
```
0.0035524981406050895
```

## 4. New priority calculations
To beter see the differences, we will use the `mat4` 4x4 example matrix.

### 4.1 The original largest eigenvector calculation
```python
priority.pri_eigen(mat4)
```
the result is:
```
array([0.4082, 0.3758, 0.1632, 0.0528])
```

### 4.2 New exponential eigenvector calculation
```python
priority.pri_expeigen(mat4)
```
the result is:
```
array([0.2244, 0.1985, 0.0689, 0.5081])
```

### 4.3 Geometric mean of columns AKA llsm
```python
priority.pri_llsm(mat4)
```
the result is:
```
array([0.2672, 0.1841, 0.0642, 0.4845])
```

## 5. Further references

[The Programmers Reference for `pyanp.priority`](https://pyanp.readthedocs.io/en/latest/refs/priority.html)

## 6. Jupyter notebook and references for this tutorial

* [Jupyter notebook with these examples](../examples/PriorityCalculations.ipynb)
* [3x3 example spreadsheet without headers](../examples/pairwise3x3-1.csv)
* [3x3 example spreadsheet with headers](../examples/pairwise3x3-1-headers.csv)
* [4x4 example without headers](../examples/pairwise4x4-1.csv)
