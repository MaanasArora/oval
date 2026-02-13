import numpy as np
import pandas as pd
from oval.conversation import Conversation, Comment, User


def read_polis(path_comments, path_votes, body_column="comment-body"):
    comments = pd.read_csv(path_comments)
    votes = pd.read_csv(path_votes)

    users = {}
    for user_id in pd.concat([comments["author-id"], votes["participant"]]).unique():
        users[user_id] = User(id=user_id)

    comments = comments.rename(
        columns={
            "comment-id": "id",
            "author-id": "author_id",
            body_column: "content",
        }
    )

    comments = [
        Comment(
            id=row["id"],
            author=users[row["author_id"]],
            content=row["content"],
        )
        for row in comments.to_dict(orient="records")
    ]

    votes_matrix = np.zeros((len(users), len(comments)), dtype=float)
    for i, comment in enumerate(comments):
        comment_votes = votes[str(comment.id)]
        votes_matrix[:, i] = comment_votes.values

    return Conversation(
        comments=comments, votes_matrix=votes_matrix, users=list(users.values())
    )
