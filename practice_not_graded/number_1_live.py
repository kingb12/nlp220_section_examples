"""
Consider this dataset: 

https://github.com/jackiekazil/data-wrangling/blob/master/data/chp3/data-text.csv

Links to an external site.

Read this data as CSV and print the first 1000 samples. 

Use csv reader and pandas read_csv - both. 
Prepare another file where you save the 1000 samples as a tab separated value 
"""

# Read this data as CSV and print the first 1000 samples. 
import csv
from typing import Any, List

# same, in pandas
import pandas as pd

df: pd.DataFrame = pd.read_csv('number-1-data-text.csv', nrows=1000)

df.to_csv('number-1-pandas-tsv.tsv', sep='\t', index=False)