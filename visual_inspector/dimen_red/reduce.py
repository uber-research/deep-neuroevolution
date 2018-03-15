"""dimensionality reduction"""
from sklearn import decomposition, manifold, discriminant_analysis
import numpy as np


def reduce_dim(X, *, labels, method='pca'):
    """dimensionality reduction"""
    print("Reducing ...")

    if method == 'downsampling':
        X_r = X
    elif method == 'lda':
        X2 = X.copy()
        X2.flat[::X.shape[1] + 1] += 0.01
        X_r = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, labels)
    elif method == 'tsne':
        X_pca = decomposition.PCA(n_components=50).fit_transform(X)
        X_r = manifold.TSNE(n_components=2, perplexity=30,
                            verbose=2, random_state=0, n_iter=1000).fit_transform(X_pca)
    elif method == 'pca':
        X_r = decomposition.PCA(n_components=2).fit_transform(X)
    elif method == 'two_end':
        nrow, ncol = X.shape
        idx_last_x, idx_last_y = int(ncol / 2 - 1), -1
        X_r = np.hstack((X[:, idx_last_x].reshape(nrow, 1), X[:, idx_last_y].reshape(nrow, 1)))
    else:
        raise NotImplementedError

    print('Reduction Completed! X.shape={} X_r.shape={}'.format(X.shape, X_r.shape))
    return X_r
