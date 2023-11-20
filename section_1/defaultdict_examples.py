from collections import defaultdict
from typing import Any, Dict, List
from datasets import load_dataset
from pprint import pprint

# some type hints, we'll just say a Turn is a Dict from str to anything else,
# but we'll really assume it has a 'dialogue_id' key, mapping to a string
Turn = Dict[str, Any]

if __name__ == '__main__':
    # our list of dictionaries (real example)
    dataset: List[Turn] = load_dataset("Brendan/icdst_multiwoz_turns_v24", split="train[0:200]")

    # create our result dictionary as a defaultdict which yields an empty list on missing keys:
    by_dialogue_id: Dict[str, List[Turn]] = defaultdict(list)

    # iterate and add to our index:
    for turn in dataset:
        # on key miss: creates a new list by calling `list`, then appends to it
        # on key hit: returns existing list, then appends to it
        by_dialogue_id[turn['dialogue_id']].append(turn)

    pprint(list(by_dialogue_id.keys()))