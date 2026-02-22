from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class User:
    id: int


@dataclass
class Comment:
    id: int
    author: User
    content: str


@dataclass
class Vote:
    comment: Comment
    user: User
    value: int


class Conversation:
    id: Optional[int] = None
    comments: List[Comment]
    users: List[User] = field(default_factory=list)
    votes: Optional[List[Vote]] = None

    _user_id_to_index_cache: Optional[dict] = None
    _comment_id_to_index_cache: Optional[dict] = None
    _votes_matrix_cache: Optional[np.ndarray] = None

    def __init__(
        self,
        comments: List[Comment],
        users: List[User],
        votes: Optional[List[Vote]] = None,
        votes_matrix: Optional[np.ndarray] = None,
    ):
        self.comments = comments
        self.votes = votes
        self.users = users

        if votes_matrix is not None:
            self._votes_matrix_cache = votes_matrix

    @property
    def user_id_to_index(self):
        if self._user_id_to_index_cache is not None:
            return self._user_id_to_index_cache
        self._user_id_to_index_cache = {
            user.id: idx for idx, user in enumerate(self.users)
        }
        return self._user_id_to_index_cache

    @property
    def comment_id_to_index(self):
        if self._comment_id_to_index_cache is not None:
            return self._comment_id_to_index_cache
        self._comment_id_to_index_cache = {
            comment.id: idx for idx, comment in enumerate(self.comments)
        }
        return self._comment_id_to_index_cache

    def comment_ids_to_indices(self, comment_ids: List[int]) -> List[int]:
        indices = [self.comment_id_to_index.get(cid) for cid in comment_ids]

        if None in indices:
            missing_ids = [cid for cid, idx in zip(comment_ids, indices) if idx is None]
            raise ValueError(f"Comment IDs not found in conversation: {missing_ids}")

        return [idx for idx in indices if idx is not None]

    def user_ids_to_indices(self, user_ids: List[int]) -> List[int]:
        indices = [self.user_id_to_index.get(uid) for uid in user_ids]

        if None in indices:
            missing_ids = [uid for uid, idx in zip(user_ids, indices) if idx is None]
            raise ValueError(f"User IDs not found in conversation: {missing_ids}")

        return [idx for idx in indices if idx is not None]

    def _build_vote_matrix_from_votes(self):
        if not self.votes:
            return None

        matrix = np.zeros(
            (len(set(vote.user.id for vote in self.votes)), len(self.comments))
        )

        for vote in self.votes:
            user_idx = self.user_id_to_index[vote.user.id]
            comment_idx = self.comment_id_to_index[vote.comment.id]
            matrix[user_idx, comment_idx] = vote.value

        return matrix

    @property
    def votes_matrix(self) -> np.ndarray:
        if self._votes_matrix_cache is not None:
            return self._votes_matrix_cache
        elif self.votes is not None:
            self._votes_matrix_cache = self._build_vote_matrix_from_votes()

            if self._votes_matrix_cache is None:
                raise ValueError(
                    "Votes provided but no valid vote matrix could be built."
                )
        else:
            raise ValueError("No votes or votes matrix available to build vote matrix.")

        return self._votes_matrix_cache
