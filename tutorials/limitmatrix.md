# Limit matrix tutorial for `pyanp`
In this tutorial you will be shown:

1. How to import the limit matrix module, where all of the ANP limit matrix related calculations are.
2. How to load a matrix from an excel/csv file and how to directly input a matrix
3. How to perform some of the basic limit matrix calculations
4. Where to find more information on all of the limit matrix calculations available
5. Reference notebooks and Excel/CSV files used in this  tutorial

## 1. Importing the `pyanp.limitmatrix` module

```python
# Pandas has DataFrames and Series, very useful things
import pandas as pd
# numpy has lots of useful things in it
import numpy as np
# lastly import our ahptree python code.  If you haven't already installed the pyanp library do
# pip install pyanp
# to get it
from pyanp import limitmatrix as lm
```

## 2. Loading data from an excel / csv file

```python
# For excel / csv file with headers or without it is the same function
matrix = lm.get_matrix("PATH_TO_YOUR_EXCEL_OR_CSV_FILE")
```
To create a matrix directly in python use
```python
hierarhcyMatrix = np.array([
    [0, 0, 0, 0, 0],
    [0.6, 0, 0, 0, 0],
    [0.4, 0, 0, 0, 0],
    [0, 0.9, 0.2, 0, 0],
    [0, 0.1, 0.8, 0, 0]
])
```

## 3. Standard limit matrix calculations
