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
matrix = priority.get_matrix("pairwise3x3-1.csv")
#this is the same matrix but with headers
matrix = priority.get_matrix("pairwise3x3-1-headers.csv")
```
To directly input the matrix
```python
mat4a = np.array([
    [1, 2, 3, 4],
    [1/2, 1, 5, 6],
    [1/3, 1/5, 1, 7],
    [1/4, 1/6, 1/7, 1]
])
```

## 3. Calculating the standard largest eigenvector priority, eigenvalue, and inconsistency

## 4. New priority calculations

## 5. Further references

## 6. Jupyter notebook and references for this tutorial

