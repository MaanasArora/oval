from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from sklearn.linear_model import LinearRegression
from oval.conversation import Conversation, Comment, User
from oval.decomposition import decompose_votes


@dataclass
class Variable:
    conversation: Conversation
    name: str

    labels: Optional[dict[int, float]] = None

    coef_: Optional[np.ndarray] = None
    participant_pred: Optional[np.ndarray] = None

    def fit(
        self,
        labels: dict[int, float],
        decomposed_votes: Optional[np.ndarray] = None,
        ndim: int = 3,
    ):
        comment_indices = np.array(
            list(map(self.conversation.comment_id_to_index.get, labels.keys()))
        )
        label_values = np.array(list(labels.values()))
        votes_matrix_labels = self.conversation.votes_matrix[:, comment_indices]

        label_values = np.nan_to_num(label_values, nan=0)
        votes_matrix_labels = np.nan_to_num(votes_matrix_labels, nan=0)

        participant_prop_labels = votes_matrix_labels @ label_values

        if decomposed_votes is None:
            decomposed_votes, pca = decompose_votes(
                self.conversation.votes_matrix, num_components=ndim
            )
        if decomposed_votes is None:
            raise ValueError("Decomposed votes could not be computed.")

        X = decomposed_votes[:, :ndim]
        y = participant_prop_labels

        model = LinearRegression()
        model.fit(X, y)

        pred = model.predict(X)

        self.labels = labels
        self.participant_pred = pred

        return model.coef_

    def predict_comments(self, comment_ids: List[int]) -> np.ndarray:
        if self.labels is None or self.participant_pred is None:
            raise ValueError("Variable must be fitted before scoring comments.")

        comment_indices = np.array(
            list(map(self.conversation.comment_id_to_index.get, comment_ids))
        )

        votes_matrix_labels = self.conversation.votes_matrix[:, comment_indices]
        votes_matrix_labels = np.nan_to_num(votes_matrix_labels, nan=0)

        pred = votes_matrix_labels.T @ self.participant_pred
        return pred

    def score_comments(self, comment_ids: List[int], labels: dict[int, float]) -> float:
        pred = self.predict_comments(comment_ids)
        label_values = np.array([labels.get(cid, np.nan) for cid in comment_ids])
        label_values = np.nan_to_num(label_values, nan=0)

        r = np.corrcoef(pred, label_values)[0, 1]
        return r
