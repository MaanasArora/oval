from oval.io import read_polis
from oval.variable import Variable
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

variable = Variable(conversation=conversation, name="Risk Tolerance")
variable.fit(labels=dict(train_labels), ndim=3)

test_comment_ids = [cid for cid, _ in test_labels]
r = variable.score_comments(test_comment_ids, dict(test_labels))
print(f"Test R: {r:.4f}")