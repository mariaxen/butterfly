import joblib
import numpy as np
from scipy.spatial import ConvexHull
import sklearn.manifold
# import torch.utils.data
import tqdm


def Rotate2D(pts, cnt, ang=np.pi / 4):
    """
    pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian
    """
    return np.dot(pts - cnt, np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])) + cnt


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    #    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi / 2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    # edges = np.zeros((len(hull_points) - 1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    # angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles - pi2),
        np.cos(angles + pi2),
        np.cos(angles)]).T

    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


class AlbumTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, size, embedding_algorithm=None, layers=None, store_embeddings=False):
        self.size = size
        self.embedding_algorithm = embedding_algorithm
        self.layers = layers
        self.store_embeddings = store_embeddings

    def fit(self, X, y=None, groups=None):

        if isinstance(self.size, int):
            self.size_ = (self.size, self.size)
        else:
            self.size_ = self.size

        if self.embedding_algorithm is None:
            self.embedding_algorithm_fit_ = sklearn.manifold.TSNE(n_components=2, perplexity=25)
        elif isinstance(self.embedding_algorithm, dict):
            self.embedding_algorithm_fit_ = sklearn.manifold.TSNE(n_components=2, **self.embedding_algorithm)
        else:
            self.embedding_algorithm_fit_ = sklearn.base.clone(self.embedding_algorithm)

        if self.layers is None:
            self.layers_ = [len, np.mean]

        # fit embedding
        X_embedded = self.embedding_algorithm_fit_.fit_transform(X.T)
        if self.store_embeddings:
            self.X_embedded_ = X_embedded

        # rotate to fill maximum space
        bbox = minimum_bounding_rectangle(X_embedded)
        xDiff = bbox[2, 0] - bbox[1, 0]
        yDiff = bbox[2, 1] - bbox[1, 1]
        angle = np.arctan2(xDiff, yDiff)
        X_rotated = Rotate2D(X_embedded, np.array([bbox[2, 0], bbox[2, 1]]), angle)
        if self.store_embeddings:
            self.X_rotated_ = X_rotated

        # grid intervals
        x = np.linspace(min(X_rotated[:, 0]), max(X_rotated[:, 0]), self.size_[0] + 1)
        y = np.linspace(min(X_rotated[:, 1]), max(X_rotated[:, 1]), self.size_[1] + 1)

        self.feature_idx_ = dict()  # saving this as an array is probably slightly faster but a lot more inefficient
        for i in range(self.size_[0]):
            for j in range(self.size_[1]):
                self.feature_idx_[(i, j)] = np.argwhere(
                    (X_rotated[:, 0] >= x[i]) & ((X_rotated[:, 0] < x[i + 1])
                                                 | ((i + 1 == self.size_[1]) & (X_rotated[:, 0] <= x[i + 1]))) &
                    (X_rotated[:, 1] >= y[j]) & ((X_rotated[:, 1] < y[j + 1])
                                                 | ((j + 1 == self.size_[1]) & (X_rotated[:, 1] <= y[j + 1]))))
        return self

    @staticmethod
    def _transform(X, size, feature_idx, layers):
        album = np.empty((X.shape[0], len(layers), size[0], size[1]))
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(X.shape[0]):
                    values = X[k, :][feature_idx[(i,j)]]
                    for l, method in enumerate(layers):
                        if values.size == 0:
                            album[k, l, i, j] = 0
                        else:
                            album[k, l, i, j] = method(values)
        return album

    def transform_parallel(self, X, y=None, groups=None, n_jobs=None):
        batches = sklearn.model_selection.KFold(n_splits=2 if n_jobs is None else n_jobs)
        pages = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(AlbumTransformer._transform)(
            X[split, :], self.size_, self.feature_idx_, self.layers_)
            for _, split in batches.split(X))
        return np.concatenate(pages)

    def transform(self, X):
        return AlbumTransformer._transform(X, self.size_, self.feature_idx_, self.layers_)


class SingleCellTransformer(sklearn.base.BaseEstimator):

    def __init__(self, size, embedding_algorithm=None, means=False, store_embeddings=False):
        self.size = size
        self.embedding_algorithm = embedding_algorithm
        self.means = means
        self.store_embeddings = store_embeddings

    def fit_transform(self, X, groups):

        if isinstance(self.size, int):
            self.size_ = (self.size, self.size)

        if self.embedding_algorithm is None:
            self.embedding_algorithm_fit_ = sklearn.manifold.TSNE(n_components=2, perplexity=25)
        elif isinstance(self.embedding_algorithm, dict):
            self.embedding_algorithm_fit_ = sklearn.manifold.TSNE(n_components=2, **self.embedding_algorithm)
        elif isinstance(self.embedding_algorithm, sklearn.base.TransformerMixin):
            self.embedding_algorithm_fit_ = sklearn.base.clone(self.embedding_algorithm)
        else:
            self.embedding_algorithm_fit_ = self.embedding_algorithm

        # fit embedding
        print("Embedding")
        if isinstance(self.embedding_algorithm_fit_, sklearn.base.BaseEstimator):
            X_embedded = self.embedding_algorithm_fit_.fit_transform(X)
        else:
            X_embedded = self.embedding_algorithm_fit_(X)
        if self.store_embeddings:
            self.X_embedded_ = X_embedded

        # rotate to fill maximum space
        print("Rotating")
        bbox = minimum_bounding_rectangle(X_embedded)
        xDiff = bbox[2, 0] - bbox[1, 0]
        yDiff = bbox[2, 1] - bbox[1, 1]
        angle = np.arctan2(xDiff, yDiff)
        X_rotated = Rotate2D(X_embedded, np.array([bbox[2, 0], bbox[2, 1]]), angle)
        if self.store_embeddings:
            self.X_rotated_ = X_rotated

        # grid intervals
        x = np.linspace(min(X_rotated[:, 0]), max(X_rotated[:, 0]), self.size_[0] + 1)
        y = np.linspace(min(X_rotated[:, 1]), max(X_rotated[:, 1]), self.size_[1] + 1)

        print("Images")
        unique_groups = np.unique(groups)
        if self.means:
            album = np.empty((unique_groups.size, 1 + X.shape[1], *self.size_))
        else:
            album = np.empty((unique_groups.size, 1, *self.size_))

        pbar = tqdm.tqdm(enumerate(unique_groups))
        for i_g, g in pbar:
            group_data = X[groups == g, :]
            group_emb = X_rotated[groups == g, :]
            for i in range(self.size_[0]):
                pbar.set_description(f"group: {i_g:5d}, row: {i:3d}")
                for j in range(self.size_[1]):
                    # pbar.set_description(f"group: {i_g:5d}, row: {i:3d}, col: {j:3d}")
                    feature_idx = (
                        (group_emb[:, 0] >= x[i]) & ((group_emb[:, 0] < x[i + 1])
                                                     | ((i + 1 == self.size_[1]) & (group_emb[:, 0] <= x[i + 1]))) &
                        (group_emb[:, 1] >= y[j]) & ((group_emb[:, 1] < y[j + 1])
                                                     | ((j + 1 == self.size_[1]) & (group_emb[:, 1] <= y[j + 1]))))
                    album[i_g, 0, i, j] = feature_idx.sum()
                    if self.means:
                        album[i_g, 1:, i, j] = group_data[feature_idx, :].mean(axis=0)

        # normalize
        album[:, 0, :, :] /= np.max(album[:, 0, :, :])

        return album

# class AlbumDataset(torch.utils.data.Dataset):

#     def __init__(self, album, labels):
#         self.album = torch.tensor(album).float()
#         self.labels = torch.tensor(labels).long()

#     def __getitem__(self, item):
#         return self.album[item], self.labels[item]

#     def __len__(self):
#         return self.album.shape[0]

#
# def create_album(data, size, embedding=None, layers=None):
#     """
#     Create your album that contains all the pictures you are training on
#     """tesnor
#
#     if isinstance(size, int):
#         size = (size, size)
#
#     if embedding is None:
#         embedding = sklearn.manifold.TSNE(perplexity=25)
#
#     if layers is None:
#         layers = [
#             lambda values, n_points: len(values),
#             lambda values, n_points: np.mean(values)
#         ]
#
#     embedded = embedding.fit_transform(data.T)
#     bbox = minimum_bounding_rectangle(embedded)
#
#     xDiff = bbox[2, 0] - bbox[1, 0]
#     yDiff = bbox[2, 1] - bbox[1, 1]
#     angle = np.arctan2(xDiff, yDiff)
#
#     embedded = Rotate2D(embedded, np.array([bbox[2, 0], bbox[2, 1]]), angle)
#
#     # grid intervals
#     x = np.linspace(min(embedded[:, 0]), max(embedded[:, 0]), size[0] + 1)
#     y = np.linspace(min(embedded[:, 1]), max(embedded[:, 1]), size[1] + 1)
#
#     album = np.empty((data.shape[0], len(layers), size[0], size[1]))
#     for row_idx in range(data.shape[0]):
#         print(row_idx, "========================================")
#         for i in range(size[0]):
#             for j in range(size[1]):
#                 pixel_idx = \
#                     (embedded[:, 0] >= x[i]) & ((embedded[:, 0] < x[i + 1])
#                                                 | ((i + 1 == size[1]) & (embedded[:, 0] <= x[i + 1]))) & \
#                     (embedded[:, 1] >= y[j]) & ((embedded[:, 1] < y[j + 1])
#                                                 | ((j + 1 == size[1]) & (embedded[:, 1] <= y[j + 1])))
#                 values = data[row_idx, pixel_idx]
#                 for l, method in enumerate(layers):
#                     if values.size == 0:
#                         album[row_idx, l, i, j] = 0
#                     else:
#                         album[row_idx, l, i, j] = method(values, embedded.shape[0])
#
#     return album, embedding, embedded, x, y
