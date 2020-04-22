"""
Sat Mar 14 21:54:45 2020
Written by: Rojan Shrestha 
"""

import numpy as np
import pandas as pd


class npPCA:
    """
    Simple numpy implementation for PCA 
    """

    def __init__(self):
        """ initialize variables
        
        Returns:
        """
        self.X = []
        self.y = []

    def load_data(
        self,
        path_to_data="",
        target_column_idx=-1,
        title_exist=None,
        dtypes={},
        use_iris_data=True,
    ):
        """ Load a data from given path 
        
        Args:
            path_to_data (string): path to data file 
            target_column (int): column index of target variable 
            title_exists (string, optional): indicate column names is defined. Defaults to None.
            data_type (dict, optional): column data types. Defaults to {}.
            use_data (bool, optional): use well tested data. Defaults to True.
        """
        # title_exists: None, if the data does not have column title,

        if use_iris_data:
            dtypes = {
                "sepallength": np.float64,
                "sepalwidth": np.float64,
                "petallength": np.float64,
                "petalwidth": np.float64,
                "target": np.str,
            }
            path_to_file = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

        df_data = pd.read_csv(path_to_file, dtype=dtypes)

        # find the column indexes of dependent variables
        columns_idxes = [
            i for i in range(len(df_data.columns)) if i != target_column_idx
        ]

        # Dependent variables are set X and target variable is y
        self.X = df_data.iloc[:, columns_idxes].values
        self.y = df_data.iloc[:, target_column_idx].values
        print("# of features:%d " % self.X.shape[1])
        print("# of records:%d " % self.X.shape[0])

    def use_cov(self):
        """ compute PCA using covariance matrix

        Returns:
            [floats float]: list or numpy array list/numpy 
        """

        # standarized
        X_norm = self.X - np.mean(self.X.T, axis=1)  # / np.std(self.X.T, axis=1)

        # covariance matrix of transpose matrix to compute the relation
        # between features using data
        X_cov = np.cov(X_norm.T)

        # eigendecomposition of d*d matrix where d represents # of features
        ei_vals, ei_vecs = np.linalg.eig(X_cov)

        # project original normalzied data into PC spaces
        self.X_trans = np.dot(ei_vecs.T, X_norm.T).T
        return ei_vals, ei_vecs

    def use_svd(self):
        """ Use SVD decomposition to get variance and principal component.

        Returns:
            [float, float]: pricipal component, variance 
        """
        X_norm = self.X - np.mean(self.X.T, axis=1)  # / np.std(self.X.T, axis=1)
        u, sigma, V = np.linalg.svd(X_norm, full_matrices=False)
        self.X_trans = np.dot(u, np.diag(sigma))
        return np.square(sigma) / (u.shape[0] - 1), V

    def fit(self, method="eig"):
        """ Eigenvectors and eigenvalues of a covariance matrix 
        are core of a PCA. The principal components (eigenvector)
        determine the direction of the new features, and eigenvalues
        holds the variance of the data along the new feature axes
        that determine their magnitude.

        Args:
            method (str, optional): Method choose to compute PCA. Defaults to "eig".

        Returns:
            [float]: projected matrix  
        """

        if method == "eig":
            self._ei_vals, self._ei_vecs = self.use_cov()
        elif method == "svd":
            self._ei_vals, self._ei_vecs = self.use_svd()

        # extract eigenvalue and eigenvector pair
        self._e_pairs = [
            ([np.abs(self._ei_vals[i]), self._ei_vecs[:, i]])
            for i in range(len(self._ei_vals))
        ]

        # select a top eigen vectors
        self._e_pairs.sort(key=lambda x: x[0], reverse=True)
        return self.X_trans

    def largest_eigenvalues(self, k=5):
        """ After sorting based on eigenvalue (variance), return
        top five eigenvalue and its corresponding eigenvectors.
        
        Args:
            k (int, optional): top k to select eigenvalues and eigenvectors. Defaults to 5.
        
        Returns:
            int: top k eigen vectors and values  
        """

        if len(self._e_pairs) < k:
            return self._e_pairs
        else:
            return self._e_pairs[:k]

    def explained_variance(self, k=5):
        """Compute explained variance

        Args:
            k (int, optional): top k. Defaults to 5.
        
        Returns:
            float: cumulative variance captured by each principal component
        """
        total_variance = sum([self._e_pairs[i][0] for i in range(len(self._e_pairs))])
        variance = [
            self._e_pairs[i][0] / total_variance for i in range(len(self._e_pairs))
        ]
        if k > len(variance):
            k = len(variance) - 1
        return np.cumsum(variance[:k])


if __name__ == "__main__":
    pca = npPCA()
    pca.load_data(target_column_idx=4)
    pca.fit()
    pca.fit(method="svd")
    # explained_variances = pca.explained_variance()
