# Tutorial on AHP usage in pyanp
In this tutorial we will:

1. Show you how to import the AHPTree class (which is what handles all AHP tree calculations)
1. Read in a full AHP tree's data from a single excel spreadsheet
1. Review the format of the Excel file
1. Show some standard calculations and their results

## Importing the AHPTree class
And a few other classes you will need for later parts of this tutorial

```python
# Pandas has DataFrames and Series, very useful things
import pandas as pd
# numpy has lots of useful things in it
import numpy as np
# lastly import our ahptree python code.  If you haven't already installed the pyanp library do
# pip install pyanp
# to get it
from pyanp import ahptree
```

## Loading data from an excel file

```python
excel_file = 'PATH_TO_YOUR_EXCEL_FILE'
ahp = ahptree.ahptree_fromdf(excel_file)
```
