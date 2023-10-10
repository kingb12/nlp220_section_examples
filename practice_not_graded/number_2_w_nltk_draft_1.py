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
nltk.download('averaged_perceptron_tagger')
print("File IDs:", twitter_samples.fileids())

# load the tweets in file "tweets.20150430-223406.json"
tweets: List[str] = twitter_samples.strings("tweets.20150430-223406.json")

# tokenizing with nltk (word_tokenize)
from nltk import word_tokenize, sent_tokenize

# data[tweet][token]
# [word_tokenize(sent) for sent in sent_tokenize(big_text)]
tokenized_tweets: List[List[str]] = [word_tokenize(tweet) for tweet in tweets]

from nltk import pos_tag

# data[tweet][index] -> (token, tag)
tagged_tweets: List[List[Tuple[str, str]]] = [pos_tag(tweet) for tweet in tokenized_tweets]

# Find the total number of adjectives and nouns, and output them.
# BK Note: maybe there is a fancy way within nltk, but we can use use Counter!

pos_counter: Counter[str] = Counter()

for tweet in tagged_tweets:
    for token, tag in tweet:
        # count all parts of speech, then print N, ADJ
        pos_counter[tag] += 1

# It turns out, nltk distinguishes between common singular, common plural, and proper nouns. All these tags start with NN
# Similarly for adjectives, all sub-categories start with JJ. We'll use this to sum across groups.
noun_tags: List[str] = [k for k in pos_counter if k.startswith('NN')]
adj_tags: List[str] = [k for k in pos_counter if k.startswith('JJ')]

# call sum with the value of each of these tags in the counter
total_nouns: int = sum(pos_counter[tag] for tag in noun_tags)
total_adjs: int = sum(pos_counter[tag] for tag in adj_tags)

print(f"# NOUN ({', '.join(noun_tags)}): {total_nouns}, # ADJ ({', '.join(adj_tags)}): {total_adjs}")

# print the top-10 nouns
# we'll create a new counter, and only add nouns to it to save space
noun_counter: Counter[str] = Counter()
for tweet in tagged_tweets:
    for token, tag in tweet:
        if tag.startswith('NN'):
            noun_counter[token] += 1

# then, we'll print the most common ones using most_common(10)
pprint(noun_counter.most_common(10))


