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
nltk.download('universal_tagset')
print("File IDs:", twitter_samples.fileids())

# load the tweets in file "tweets.20150430-223406.json"
tweets: List[str] = twitter_samples.strings("tweets.20150430-223406.json")

# tokenizing with nltk (word_tokenize)
from nltk import word_tokenize

# data[tweet][token]
# [word_tokenize(sent) for sent in sent_tokenize(big_text)]
tokenized_tweets: List[List[str]] = [word_tokenize(tweet) for tweet in tweets]

# Change 1: using pos_tag_sents for more efficient tagging of multiple sentences, see the pos_tag docs!
from nltk import pos_tag_sents

# data[tweet][index] -> (token, tag)

# Change 2: using tagset=universal: now all nouns are under NOUN, adjectives under ADJ, instead of multiple tags.
tagged_tweets: List[List[Tuple[str, str]]] = pos_tag_sents(tokenized_tweets, tagset='universal')


from nltk import FreqDist

# Change 3: use FreqDist and a generator comprehension (like a list comprehension that does not allocate memory for a new list)
# Thanks to Deo & others for suggesting FreqDist!
# list comprehensions with nested iterators are written sort of oddly: you write them in the order you would write the for loops, moving inward. E.g. this is the same as:
# for tweet in tagged_tweets:
#     for token, tag in tweet:
#         yield tag
pos_dist: FreqDist = FreqDist(samples=(tag for tweet in tagged_tweets for (token, tag) in tweet))

print(f"# NOUN: {pos_dist['NOUN']}, # ADJ: {pos_dist['ADJ']}")


# print the top-10 nouns
# FreqDist seems to support a very similar API to Counter, with additional helpers and fast counting on init. We'll count 
# nounts and view with most_common. We've modified the generator here to only include tags which are 'NOUN', and to yield 
# the token instead of the tag
noun_dist: FreqDist = FreqDist(
    samples=(token for tweet in tagged_tweets for (token, tag) in tweet if tag == 'NOUN')
)

# then, we'll print the most common ones using most_common(10)
pprint(noun_dist.most_common(10))


