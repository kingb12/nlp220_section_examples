"""
Import Twitter Corpus from NLTK. 

 

For each of the tweets in the tweets.20150430-223406.json corpus, apply tokenization and POS tagging. 

Try using both NLTK and Spacy 
 

Find the total number of adjectives and nouns, and output them.

Find the top-10 nouns.
"""

from collections import Counter
from typing import List, Tuple
import nltk
from nltk.corpus import twitter_samples
from pprint import pprint

# check out what files are included in the corpus:
nltk.download('twitter_samples')
print("File IDs:", twitter_samples.fileids())

# load the tweets in file "tweets.20150430-223406.json"
tweets: List[str] = twitter_samples.strings("tweets.20150430-223406.json")

# tokenizing and tagging in one step with spacy
import spacy
from spacy.tokens.doc import Doc

# nlp(...) typically does the following steps, all as one 'pipeline'
# tokenize -> tag -> parse -> NER ... to produce a Doc, which idisable heavy pipeline steps
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])



# Since in our analysis we don't actually care about tweet-level statistics, to speed up processing I'll first join 
# them to be one string
all_tweets: str = "\n".join(tweets)

# The max length is set to prevent unreasonable memory allocation (read more in the error message from removing this line).
# Since we disabled NER and parsing, we should be ok to parse our whole document.
nlp.max_length = len(all_tweets) + 1
doc: Doc = nlp.__call__(all_tweets)

# Counting parts of speech
pos_counts: Counter[str] = Counter(token.pos_ for token in doc)
print(f"# NOUN: {pos_counts['NOUN']}, # ADJ: {pos_counts['ADJ']}")


# print the top-10 nouns.
# Careful! a token is a Token object, and likely unique within the corpus. str(token)
# returns the expected string representation of the token itself, without added attributes
# like .pos_, etc.
noun_counts: Counter[str] = Counter(str(token) for token in doc if token.pos_ == 'NOUN')

# then, we'll print the most common ones using most_common(10)
pprint(noun_counts.most_common(10))