from oval.io import read_polis
from oval.variable import DiffusionVariable
import pandas as pd
from sklearn.model_selection import train_test_split

conversation = read_polis(
    "data/vtaiwan/comments_translated.csv",
    "data/vtaiwan/participant-votes.csv",
    body_column="translated_comment",
)
labels = pd.read_csv("data/vtaiwan/ratings.csv")
labels = dict(zip(labels["comment-id"], labels["rating"]))

train_labels, test_labels = train_test_split(
    list(labels.items()), test_size=0.4, random_state=42
)

variable = DiffusionVariable(conversation=conversation)
variable.fit(anchors=dict(train_labels))

r = variable.score_comments(dict(test_labels))
print(f"Test R: {r:.4f}")