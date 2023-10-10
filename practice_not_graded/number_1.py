"""
Consider this dataset: 

https://github.com/jackiekazil/data-wrangling/blob/master/data/chp3/data-text.csv

Links to an external site.

Read this data as CSV and print the first 1000 samples. 

Use csv reader and pandas read_csv - both. 
Prepare another file where you save the 1000 samples as a tab separated value 
"""

import os
from typing import Any, List
import requests

# I've done some non-essential steps for my own convenience:
# 1) Downloading data from within the script, only if not already saved to a file (which I otherwise trust implicitly)
# 2) Saving the data to a file named specific to this practice example, so it can co-exist in the folder with others
RAW_GITHUBUSERCONTENT_LINK: str = "https://raw.githubusercontent.com/jackiekazil/data-wrangling/master/data/chp3/data-text.csv"
MY_FILE_NAME: str = "number-1-data-text.csv"

def download_if_not_present(file_name: str = MY_FILE_NAME):
    if not os.path.exists(file_name):
        print(f"Downloading {RAW_GITHUBUSERCONTENT_LINK} to {file_name} and saving...")
        data: bytes = requests.get(RAW_GITHUBUSERCONTENT_LINK).content

        # using 'wb' to allow writing of bytes, instead of a str, etc
        with open(file_name, "wb") as f:
            f.write(data)
        print("Saved.")

if __name__ == '__main__':
    download_if_not_present()

    # Read this data as CSV and print the first 1000 samples. (using csv)
    import csv

    data: List[List[Any]] = []
    with open(MY_FILE_NAME, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            print(row)
            data.append(row)
            # i starts at 0, but there is also the header row, so this loop does 1001 prints which should be correct
            if i >= 1000:
                print("printed 1000 rows + header")
                break
    
    # Read this data as CSV and print the first 1000 samples. (using pandas)
    import pandas as pd
    df: pd.DataFrame = pd.read_csv(MY_FILE_NAME)
    
    # technically doesn't print all data row by row, you can use a for loop over df[:1000] to do this
    print(df[:1000])

    # could also just load the number of rows you want with nrows. Will be faster!
    small_df: pd.DataFrame = pd.read_csv(MY_FILE_NAME, nrows=1000)

    # Prepare another file where you save the 1000 samples as a tab separated value (from  csv parse)
    with open("number_1_1000_elem_as_tsv.tsv", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(data)

    # Prepare another file where you save the 1000 samples as a tab separated value (from pandas)
    # index = False prevents leading integers!
    df.to_csv("number_1_1000_elem_as_tsv_pandas.tsv", sep="\t", index=False)
