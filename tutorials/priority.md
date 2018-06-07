# Priority calculations Tutorial

This tutorial covers:

1. Importing the necessary libraries
2. Loading a matrix from a spreadsheet or directly inputting
3. Calculating the standard largest eigenvector priority
4. Calculating the standard largest eigen value and inconsistency
5. New priority calculations
6. Further references
7. Jupyter notebook and references for this tutorial

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

