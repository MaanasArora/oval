from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from sklearn.linear_model import LinearRegression
from oval.conversation import Conversation, Comment, User
from oval.decomposition import decompose_votes


@dataclass
class Variable(ABC):
    conversation: Conversation

    @abstractmethod
    def fit(self, anchors: dict[int, float]) -> None:
        pass

    @abstractmethod
    def predict_comments(self, comment_ids: Optional[List[int]] = None) -> np.ndarray:
        pass

    @abstractmethod
    def predict_users(self, user_ids: Optional[List[int]] = None) -> np.ndarray:
        pass

    def score_comments(self, anchors: dict[int, float]) -> float:
        pred = self.predict_comments(list(anchors.keys()))
        anchor_values = np.array(list(anchors.values()))
        anchor_values = np.nan_to_num(anchor_values, nan=0)

        r = np.corrcoef(pred, anchor_values)[0, 1]
        return r


class LinearVariable(Variable):
    def fit(
        self,
        anchors: dict[int, float],
        ndim: int = 3,
        relevance_scoring: bool = False,
    ):
        comment_indices = self.conversation.comment_ids_to_indices(list(anchors.keys()))
        anchor_values = np.array(list(anchors.values()))
        votes_matrix_anchors = self.conversation.votes_matrix[:, comment_indices]

        anchor_values = np.nan_to_num(anchor_values, nan=0)
        votes_matrix_anchors = np.nan_to_num(votes_matrix_anchors, nan=0)

        participant_prop_anchors = votes_matrix_anchors @ anchor_values
        if relevance_scoring:
            participant_relevance = np.abs(votes_matrix_anchors) @ np.abs(anchor_values)
            participant_prop_anchors *= np.sqrt(participant_relevance)

        decomposed_votes, pca = decompose_votes(
            self.conversation.votes_matrix, num_components=ndim
        )

        model = LinearRegression()
        model.fit(decomposed_votes, participant_prop_anchors)

        pred = model.predict(decomposed_votes)

        self.labels = anchor_values
        self.participant_pred = pred
        self.coef_ = model.coef_

    def predict_comments(self, comment_ids: Optional[List[int]] = None) -> np.ndarray:
        if self.labels is None or self.participant_pred is None:
            raise ValueError("Variable must be fitted before scoring comments.")

        if comment_ids is None:
            comment_ids = list(self.conversation.comment_id_to_index.keys())

        comment_indices = self.conversation.comment_ids_to_indices(comment_ids)

        votes_matrix_anchors = self.conversation.votes_matrix[:, comment_indices]
        votes_matrix_anchors = np.nan_to_num(votes_matrix_anchors, nan=0)

        pred = (
            votes_matrix_anchors.T
            @ self.participant_pred
            / len(self.conversation.users)
        )
        return pred

    def predict_users(self, user_ids: Optional[List[int]] = None) -> np.ndarray:
        if user_ids is None:
            user_ids = list(self.conversation.user_id_to_index.keys())

        if self.labels is None or self.participant_pred is None:
            raise ValueError("Variable must be fitted before scoring users.")

        user_indices = self.conversation.user_ids_to_indices(user_ids)
        pred = self.participant_pred[user_indices]
        return pred


class DiffusionVariable(Variable):
    def _diffuse(
        self, votes_matrix: np.ndarray, anchors: np.ndarray, alpha: float
    ) -> np.ndarray:
        num_nodes = votes_matrix.shape[0] + votes_matrix.shape[1]
        graph = np.zeros((num_nodes, num_nodes))

        graph[: votes_matrix.shape[0], votes_matrix.shape[0] :] = votes_matrix
        graph[votes_matrix.shape[0] :, : votes_matrix.shape[0]] = votes_matrix.T

        mask = ~np.isnan(graph)
        deg = np.sum(mask, axis=1)
        P = np.nan_to_num(graph, nan=0) / np.where(deg > 0, deg, 1)[:, np.newaxis]

        anchor_mask = ~np.isnan(anchors)
        y = np.concatenate(
            [np.zeros(votes_matrix.shape[0]), np.nan_to_num(anchors, nan=0)]
        )
        f = y.copy()

        delta = np.inf
        while delta > 1e-6:
            f_new = P @ f * (1 - alpha) + y * alpha
            delta = np.linalg.norm(f_new - f, ord=1)
            f = f_new

        comment_scores = f[votes_matrix.shape[0] :]

        comment_scores_anchor = comment_scores[anchor_mask]
        comment_scores_nonanchor = comment_scores[~anchor_mask]

        anchor_mean, anchor_std = np.mean(comment_scores_anchor), np.abs(
            np.std(comment_scores_anchor)
        )
        nonanchor_mean, nonanchor_std = np.mean(comment_scores_nonanchor), np.abs(
            np.std(comment_scores_nonanchor)
        )

        comment_scores[anchor_mask] = (comment_scores_anchor - anchor_mean) / (
            anchor_std + 1e-10
        )
        comment_scores[~anchor_mask] = (comment_scores_nonanchor - nonanchor_mean) / (
            nonanchor_std + 1e-10
        )

        participant_scores = f[: votes_matrix.shape[0]]
        participant_scores = (participant_scores - np.mean(participant_scores)) / (
            np.abs(np.std(participant_scores)) + 1e-10
        )

        f[votes_matrix.shape[0] :] = comment_scores
        f[: votes_matrix.shape[0]] = participant_scores
        return f

    def fit(
        self,
        anchors: dict[int, float],
        alpha: float = 0.5,
    ):
        comment_indices = self.conversation.comment_ids_to_indices(list(anchors.keys()))
        anchor_values = np.array(list(anchors.values()))

        labels = np.full(len(self.conversation.comments), np.nan)
        labels[comment_indices] = anchor_values

        self.anchors = anchors
        self.pred = self._diffuse(self.conversation.votes_matrix, labels, alpha)

    def predict_comments(self, comment_ids: Optional[List[int]] = None) -> np.ndarray:
        if not comment_ids:
            return self.pred[self.conversation.votes_matrix.shape[0] :]

        comment_indices = self.conversation.comment_ids_to_indices(comment_ids)
        comment_indices = [
            self.conversation.votes_matrix.shape[0] + idx for idx in comment_indices
        ]
        return self.pred[comment_indices]

    def predict_users(self, user_ids: Optional[List[int]] = None) -> np.ndarray:
        if not user_ids:
            return self.pred[: len(self.conversation.users)]

        user_indices = self.conversation.user_ids_to_indices(user_ids)
        user_indices = [idx + len(self.conversation.comments) for idx in user_indices]
        return self.pred[user_indices]
