import numpy as np
from sklearn.decomposition import PCA
import umap


def decompose_votes(
    vote_matrix: np.ndarray, num_components: int = 2, method: str = "umap"
):
    vote_matrix_nonan = np.nan_to_num(vote_matrix, nan=0)
    if method == "umap":
        reducer = umap.UMAP(n_components=num_components, random_state=42)
    elif method == "pca":
        reducer = PCA(n_components=num_components)
    transformed = reducer.fit_transform(vote_matrix_nonan)

    total_votes = np.sum(~np.isnan(vote_matrix), axis=1)
    vote_scale = np.sum(~np.isnan(vote_matrix), axis=1)
    vote_scale = np.sqrt(total_votes / (vote_scale + 1e-10))

    return transformed * vote_scale[:, None], reducer
