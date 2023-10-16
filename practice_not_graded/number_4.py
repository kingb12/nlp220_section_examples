"""
Build a binary classifier using Naive Bayes using this dataset. https://archive.ics.uci.edu/dataset/53/iris

Test the accuracy on 80-20 split.
"""
import io
from typing import Dict, List
import zipfile
import requests
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# right-click 'Copy Link Address' on the Download button from link above
zip_file_url: str = "https://archive.ics.uci.edu/static/public/53/iris.zip"

data_dir: str = "data/iris"

if not os.path.exists(os.path.join(data_dir, "iris.data")):
    response = requests.get(zip_file_url)

    # Create a file-like object from the response content
    zip_data = io.BytesIO(response.content)

    # read from our file-like object holding the contents of 'iris.zip', and extract it into the data dir
    # same as:
    # > wget https://archive.ics.uci.edu/static/public/53/iris.zip
    # > mkdir -p data/iris
    # > unzip -d data/iris iris.zip 
    os.makedirs(data_dir, exist_ok=True)
    with zipfile.ZipFile(zip_data, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

# read in the data (its a CSV despite the name)
df: pd.DataFrame = pd.read_csv(os.path.join(data_dir, "iris.data"), names=
                                ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "species"])

# now we need to make this a binary classification problem somehow, despite 3 species. Wee'll do setosa vs. non-setosa
df['label'] = df['species'] == 'Iris-setosa'
assert 1 > df['label'].mean() >= 0, "expected points in each class"

# another problem: we have continuous features. Two choices: 
# 1) convert to discrete features and use the familiar Multinomial NB
# 2) GaussianNB: handles continuous features by assuming the feature likelihoods p(x_i | y) are Gaussians,
# with learnable mean and variance 

# ======= GaussianNB ===========

# Step 1: Perform an 80/20 train-test split
X = df[["sepal_len", "sepal_wid", "petal_len", "petal_wid"]]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: feature engineering. Skipped! Working directly with continuou smeasurements as features

# Step 3: train model
gaussian_nb = GaussianNB()
gaussian_nb.fit(X_train, y_train)

# Step 4: predict and score accuracy
y_pred_gaussian = gaussian_nb.predict(X_test)
acc: float = accuracy_score(y_test, y_pred_gaussian)
print(f"Accuracy (GaussianNB): {acc}")

# ======= Discretized features ===========



# We'll loop over our hyper-paramters for demonstration, since this is fast:
for num_buckets in (1, 2, 3, 4, 5, 6, 1000):
    # 1 bucket is a degenerate case: we should do no better than the class prior!

    # Step 1: Perform an 80/20 train-test split
    X = df[["sepal_len", "sepal_wid", "petal_len", "petal_wid"]]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 2: feature engineering: for each feature, break into buckets. 
    # Number of buckets is a hyper-parameter
    def bucketize_column(column: pd.Series, num_buckets: int = 2, bins: List[float] = None) -> pd.Series:
        min_val = column.min()
        max_val = column.max()
        bucket_width = (max_val - min_val) / num_buckets
        # not sure this is actually the most 'even' way to assign them, but it works enough
        bins: List[float] = bins or [min_val + ((i + 1) * bucket_width) for i in range(num_buckets - 1)]
        def assign_bucket(val: float) -> int:
            i = 0
            while i < len(bins) and val >= bins[i]:
                i += 1
            assert i >= 0 and i < num_buckets
            return i
        return column.map(assign_bucket), bins

    all_bins: Dict[str, List[float]] = {}
    for column in X_train.columns:
        X_train[column], all_bins[column] = bucketize_column(X_train[column], num_buckets=num_buckets)

    # need to featurize test data with bins 'learned'/engineered w/ training data! Similar to vocab 
    for column in X_test:
        X_test[column], all_bins[column] = bucketize_column(X_test[column], bins=all_bins[column], num_buckets=num_buckets)

    # Step 3: train model
    multinomial_nb = MultinomialNB()
    multinomial_nb.fit(X_train, y_train)

    # Step 4: predict and score accuracy
    y_pred_multi = multinomial_nb.predict(X_test)
    acc: float = accuracy_score(y_test, y_pred_multi)

    # note: on this tiny dataset, we can get perfect test accuracy with just 3 buckets
    print(f"Accuracy (MultinomialNB, # buckets={num_buckets}): {acc}")
