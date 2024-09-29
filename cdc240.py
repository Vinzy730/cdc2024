import numpy as np
import pandas as pd

ds = pd.read_csv('data_SS.csv')
df = pd.DataFrame()

i = 1
while i < 25: 
    rslt_ds =ds[ds[f"Category {i}"] > 0]
    pd.DataFrame("Category{i}", rslt_ds)
    i += 1 
  
print('\nResult dataframe :\n', df) 