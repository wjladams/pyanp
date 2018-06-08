# ANP row sensitivity tutorial

In this tutorial we will learn how to:

1. Load the necessary libraries
2. Read in super matrices from csv/excel files
3. Directly input super matrices
4. Standard calculations
5. Create row sensitivity graphs
6. Find the programmers reference for all row sensitivity functions
7. Find the resources for this tutorial

# 1. Load the necessary libraries
The library you need is `pyanp.rowsens`, but we will also make use of `numpy` and `pandas` so we will import those as well.

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


# 2. Read in super matrices from csv/excel files

```python
# CSV/sheet sheet with no headers, handle this way
mat4 = rowsens.get_matrix("supermatrix1.csv")
# CSV/sheet with headers is the handled the same way
mat4 = rowsens.get_matrix("supermatrix4x4-headers.csv")
```

# 3. Directly input super matrices
We can also directly input the supermatrix as
```python
mat4 = np.array([[0.5 , 0.1 , 0.  , 0.  ],
       [0.2 , 0.6 , 0.  , 0.  ],
       [0.1 , 0.05, 0.75, 0.1 ],
       [0.2 , 0.25, 0.25, 0.9 ]])
```

# 4. Standard calculations
## 4.1 Calculate row adjust matrix
```python
rowsens.row_adjust(mat=mat4, row=0, p=0.1, p0mode=0.5)
```
the result is:
```
array([[0.1   , 0.02  , 0.    , 0.    ],
       [0.36  , 0.6533, 0.    , 0.    ],
       [0.18  , 0.0544, 0.75  , 0.1   ],
       [0.36  , 0.2722, 0.25  , 0.9   ]])
```
To use the smartp0, pass parameter p0mode=#of_the_row_you_are_adjusting, e.g.
```python
rowsens.row_adjust(mat=mat4, row=0, p=0.1, p0mode=0)
```
the result is:
```
array([[0.1788, 0.076 , 0.    , 0.    ],
       [0.3285, 0.616 , 0.    , 0.    ],
       [0.1642, 0.0513, 0.75  , 0.1   ],
       [0.3285, 0.2567, 0.25  , 0.9   ]])
```
If you want to scale by cluster, tell us the nodes in the cluster being adjusted
```python
rowsens.row_adjust(mat=mat4, row=0, p=0.1, p0mode=0, cluster_nodes=(0,1))
```
the result is:
```
array([[0.1109, 0.0692, 0.    , 0.    ],
       [0.5891, 0.6308, 0.    , 0.    ],
       [0.1   , 0.05  , 0.75  , 0.1   ],
       [0.2   , 0.25  , 0.25  , 0.9   ]])
```
## 4.2 Calculate row adjusted priorities
We need a more complicate matrix to show interesting results, so let's load that
```python
mat42 = rowsens.get_matrix("supermatrix4x4-2.csv")
```
Let's get the original priority of this matrix
```python
rowsens.priority(mat42)
```
which results in:
```
array([0.2209, 0.3505, 0.1031, 0.3255])
```
If we then want to calculate the new priorities after adjusting the first row, the code is:
```python
rowsens.row_adjust_priority(mat=mat42, row=0, p=0.1, p0mode=0.5)
```
the result is:
```
array([0.0308, 0.3479, 0.0947, 0.3364])
```
To use smart p0, it is the same as the row_adjust function:
```python
rowsens.row_adjust_priority(mat=mat42, row=0, p=0.1, p0mode=0)
```
and the result is:
```
array([0.1158, 0.3511, 0.0998, 0.3282])
```
# 5. Create row sensitivity graphs


# 6. Find the programmers reference for all row sensitivity functions


# 7. Find the resources for this tutorial

