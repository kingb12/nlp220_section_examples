from typing import Dict, Iterable
import pandas as pd


PATH_TO_TSV: str = "https://raw.githubusercontent.com/kingb12/nlp220_section_examples/main/SMSSpamCollection.tsv"

def get_longest_text(texts: Iterable[str]) -> str:
    max_len: int = -1
    longest_text: str = None
    for text in texts:
        if len(text) > max_len:
            max_len = len(text)
            longest_text = text
    return longest_text

# Loading the TSV into pandas
df: pd.DataFrame = pd.read_csv(PATH_TO_TSV, delimiter="\t", names=["label", "text"])

# Using groupby to find the longest text in each label:
result: Dict[str, str] = df.groupby('label').apply(lambda frame: get_longest_text(frame['text'])).to_dict()
print("\n\nLongest ham:", result['ham'])
print("\n\nLongest spam:", result['spam'])
