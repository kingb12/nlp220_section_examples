import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

# =========================================== 1. Loading/Data Processing/train_test_split ===========================================

# Neat trick: you can pass a url directly to read_csv. This works well for small datasets like this one, which 
# can be downloaded efficiently on each run. Make sure to use the 'Raw' link, or replace this with a local path.
PATH_TO_TSV: str = "https://raw.githubusercontent.com/kingb12/nlp220_section_examples/main/SMSSpamCollection.tsv"

# Loading the TSV into pandas
df: pd.DataFrame = pd.read_csv(PATH_TO_TSV, delimiter="\t", names=["label", "text"])

# Creating a numeric label (1 for spam, 0 for ham)
df['label'] = df['label'] == 'spam'
assert 0 < df['label'].mean() < 1

# Splitting the data into a train/test split. Be sure to do in one call! Calling separately on (X, y) implicitly shuffles supervised pairs
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# =========================================== 2. Feature Engineering ================================================================

# We'll use a simple CountVectorizer to build a unigram BOW model. Note I did not lowercase, in this dataset
# capitalization actually helps differentiate. No way to know this in advance without looking at data or just trying both!
vectorizer: CountVectorizer = CountVectorizer(lowercase=False)


# important: only 'fit' with training data. Defining your features with test data gives an unrealistic evaluation of unseen data 
# (e.g. you might add words to your vocabulary that aren't in training data, and should otherwise be treated as unseen)
X_train_feat = vectorizer.fit_transform(X_train)
X_test_feat = vectorizer.transform(X_test)

# =========================================== 3 & 4. Model Training & Evaluation ====================================================

# Create a SVM classifier using LinearSVC. Also acceptable: SGDClassifier with default params/hinge loss (this is an identical model optimized by other means)
svm = LinearSVC()
svm.fit(X_train_feat, y_train)
y_svm = svm.predict(X_test_feat)

print(f"SVM Accuracy: {accuracy_score(y_test, y_svm):3f}")
print(f"SVM F1: {f1_score(y_test, y_svm):3f}")

# Create a Multinomial Naive Bayes Classifer using MultinomialNB. We gave points for GaussianNB, but this is actually a different model!
# When choosing between these, consider whether your features are predominately discrete (MultinomialNB) or continuous (GaussianNB)
nb = MultinomialNB()
nb.fit(X_train_feat, y_train)
y_nb = nb.predict(X_test_feat)

print(f"NB Accuracy: {accuracy_score(y_test, y_nb):3f}")
print(f"NB F1: {f1_score(y_test, y_nb):3f}")
