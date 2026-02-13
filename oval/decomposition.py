import numpy as np
from sklearn.decomposition import PCA


def decompose_votes(vote_matrix: np.ndarray, num_components: int = 2):
    pca = PCA(n_components=num_components)

    vote_matrix_nonan = np.nan_to_num(vote_matrix, nan=0)
    transformed = pca.fit_transform(vote_matrix_nonan)

    total_votes = np.sum(~np.isnan(vote_matrix_nonan), axis=1)
    vote_scale = np.sum(~np.isnan(vote_matrix_nonan), axis=1)
    vote_scale = np.sqrt(total_votes / (vote_scale + 1e-10))

    return transformed * vote_scale[:, None], pca
