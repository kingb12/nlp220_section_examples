import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt


sent_id_to_sent: pd.DataFrame = pd.read_csv("class_assignments/ca1/section5/stanfordSentimentTreebankRaw/sentlex_exp12.txt",
                                            names=["id", "sentence"])

frame_content = []
# This CSV is more annoying, there is an id followed by an arbitrary number of raw scores, which we will average, to build a frame
# from sent_id to raw_score average
with open("/home/bking2/nlp220/section_examples/class_assignments/ca1/section5/stanfordSentimentTreebankRaw/rawscores_exp12.txt", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        
        frame_content.append({"id": int(row[0]), "raw_score": np.mean([int(score) for score in row[1:]])})
sent_id_to_raw_score: pd.DataFrame = pd.DataFrame(frame_content)



all_raw: pd.DataFrame = pd.merge(sent_id_to_sent, sent_id_to_raw_score, on='id', how='inner')


phrase_id_to_score: pd.DataFrame = pd.read_csv("/home/bking2/nlp220/section_examples/class_assignments/ca1/section5/stanfordSentimentTreebank/sentiment_labels.txt",
                                               delimiter='|', names=["phrase_id", "score"], skiprows=1)
phrase_to_id: pd.DataFrame = pd.read_csv("/home/bking2/nlp220/section_examples/class_assignments/ca1/section5/stanfordSentimentTreebank/dictionary.txt", 
                                         delimiter='|', names=["phrase", "phrase_id"])

all_main: pd.DataFrame = pd.merge(phrase_id_to_score, phrase_to_id, on="phrase_id", how="inner")

# inner will lose some rows, but as long as we get most of them we should have enough 'samples' from which to infer score
everything: pd.DataFrame = pd.merge(all_main, all_raw, left_on="phrase", right_on="sentence", how="inner")

ax = everything.plot.scatter(x="raw_score", y="score", title="Scores vs. Raw Scores in Stanford Sentiment Tree Bank")

plt.savefig("class_assignments/ca1/section_6/scores_vs_raw_scores.png")
plt.close()