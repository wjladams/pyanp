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

## 3. Some limit matrix calculations

### 3.1 The calculaus type limit matrix calculation

The standard limit matrix calculation used in Super Decisions

```python
lm.calculus(matrix)
```
the result is:
```
array([[0.    , 0.    , 0.    , 0.    ],
       [0.    , 0.    , 0.    , 0.    ],
       [0.2857, 0.2857, 0.2857, 0.2857],
       [0.7143, 0.7143, 0.7143, 0.7143]])
```

### 3.2 The hiearchy formula
We need need to calculate on the hierarchy matrix we defined above.
```python
hierarhcyMatrix = np.array([
    [0, 0, 0, 0, 0],
    [0.6, 0, 0, 0, 0],
    [0.4, 0, 0, 0, 0],
    [0, 0.9, 0.2, 0, 0],
    [0, 0.1, 0.8, 0, 0]
])
lm.hiearhcy_formula(hierarhcyMatrix)
```
the result is:
```
array([[0.  , 0.  , 0.  , 0.  , 0.  ],
       [0.3 , 0.  , 0.  , 0.  , 0.  ],
       [0.2 , 0.  , 0.  , 0.  , 0.  ],
       [0.31, 0.9 , 0.2 , 0.  , 0.  ],
       [0.19, 0.1 , 0.8 , 0.  , 0.  ]])
```

### 3.3 The new hierarchy formula
To see how this calculation can be different we use a different matrix
```python
matrix2 = np.array([
    [0.5, 0.3, 0.4, 0.0, 0.0],
    [0.1, 0.2, 0.2, 0.0, 0.0],
    [0.1, 0.1, 0.1, 0.0, 0.0],
    [0.2, 0.3, 0.1, 0.0, 0.0],  
    [0.1, 0.1, 0.2, 0.0, 0.0],  
])
```
Now let's calculate without using the extra limit
```python
lm.limit_newhierarchy(matrix2, with_limit=False)
```
the result is:
```
array([[0.3277, 0.3277, 0.3277, 0.    , 0.    ],
       [0.0988, 0.0988, 0.0988, 0.    , 0.    ],
       [0.0735, 0.0735, 0.0735, 0.    , 0.    ],
       [0.3206, 0.3206, 0.3206, 0.    , 0.    ],
       [0.1794, 0.1794, 0.1794, 0.    , 0.    ]])
```
Next let's do it with the extra limit
```python
lm.limit_newhierarchy(matrix2, with_limit=False)
```
the result is:
```
array([[0.4965, 0.4965, 0.4965, 0.    , 0.    ],
       [0.1498, 0.1498, 0.1498, 0.    , 0.    ],
       [0.1114, 0.1114, 0.1114, 0.    , 0.    ],
       [0.1554, 0.1554, 0.1554, 0.    , 0.    ],
       [0.0869, 0.0869, 0.0869, 0.    , 0.    ]])
```
Lastly, let us look at the calculus type calculation:
```python
lm.calculus(matrix2)
```
the result is:
```
array([[0.4458, 0.4458, 0.4458, 0.    , 0.    ],
       [0.1345, 0.1345, 0.1345, 0.    , 0.    ],
       [0.1   , 0.1   , 0.1   , 0.    , 0.    ],
       [0.2051, 0.2051, 0.2051, 0.    , 0.    ],
       [0.1147, 0.1147, 0.1147, 0.    , 0.    ]])
```

## 4. Additional limit matrix calculations
You can find [all limit matrix calculations here.](https://pyanp.readthedocs.io/en/latest/refs/limitmatrix.html)

## 5. Reference notebooks and Excel/CSV files used in this  tutorial

* [Jupyter notebook with these calculations](../examples/ANPSupermatrixCalcs.ipynb)
* [First supermatrix excel file (has headers)](../examples/supermatrix1.xlsx)
* [First supermatrix CSV file version (has headers)](../examples/supermatrix1.csv)
* [The same supermatirx without headers as excel](../examples/supermatrix2.xlsx)
* [The same supermatirx without headers as CSV](../examples/supermatrix2.csv)

