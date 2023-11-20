from collections import Counter
from typing import Any, Dict, List
from datasets import load_dataset
from pprint import pprint

if __name__ == '__main__':  

    dataset: List[Dict[str, Any]] = load_dataset("SetFit/20_newsgroups", split="train")

    # instead of the default value, the type argument to the counter is the type of the keys. In SetFit/20_newsgroups,
    # label_text are strings
    label_counter: Counter[str] = Counter()
    for item in dataset:
        # key miss: count is, increment by 1 and save to key
        # key hit: count is retrieved, incremented by 1, saved
        label_counter[item['label_text']] += 1  # Does not need to be an int! Can use floats (e.g. log-probs)
    
    # pprint() == pretty print. pprint module also has functions like pformat, which is pprint, but returns the string instead of printing
    print("\n===== All label counts in SetFit/20_newsgroups (train) =====\n")
    pprint(label_counter)

    # helper method: most_common(n): return the top-n (key, value):
    print("\n===== Top-5 most common labels =====\n")
    pprint(label_counter.most_common(5))