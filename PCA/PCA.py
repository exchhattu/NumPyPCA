

"""
Sat Mar 14 21:54:45 2020
Written by: Rojan Shrestha 
"""

import pandas as pd
import numpy as np

class NP_PCA:

    def __init__(self):
        self.X = []
        self.y = []
        

    def load_data(self):
            """load a data from given path """
            dtypes = {"sepallength":np.float64,  
                      "sepalwidth":np.float64, 
                      "petallength":np.float64, 
                      "petalwidth":np.float64, 
                      "target":np.str}
            path_to_file = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
            df_data = pd.read_csv(path_to_file, header = None, 
                                  dtype= dtypes)

            print(df_data.info())
            self.X = df_data.iloc[:,0:4].values
            self.y = df_data.iloc[:, 4].values
            print("# of features:%d " %self.X.shape[0])
            
    
    def fit(self):
        """
        Eigenvectors and eigenvalues of a covariance matrix 
        is core of a PCA. The principal components (eigenvector)
        determine the direction of the new features, and eigenvalues
        holds the variance of the data along the new feature axes
        that determine their magnitude.
        """
        
        # standarized
        print(np.mean(self.X, axis=1).shape)
        X_norm = (self.X - np.mean(self.X, axis=0)) / np.std(self.X, axis=0)
        
        # covariance matrix
        # X_mean = np.mean(X_norm, axis=0) # featurewise mean
        # X_cov_test = (X_norm - X_mean).T.dot(X_norm - X_mean)/(X_norm.shape[0] -1)
        # print("X_cov_test ", X_cov_test)
        X_cov = np.cov(X_norm.T)
        print("X_cov ", X_cov)
        
        # eigendecomposition of d*d matrix where d represents # of features
        self._ei_vals, self._ei_vecs = np.linalg.eig(X_cov)
        print("test ", self._ei_vals, self._ei_vecs)
        
        # extract eigenvalue and eigenvector pair
        self._e_pairs = [([np.abs(self._ei_vals[i]) , self._ei_vecs[:,i]]) \
                        for i in range(len(self._ei_vals))]
        # select a top eigen vectors
        self._e_pairs.sort(key=lambda x: x[0], reverse=True)
    
    def largest_eigenvalues(self, k=5):
        """
        After sorting based on eigenvalue (variance), return
        top five eigenvalue and its corresponding eigenvectors.
        """
        if len(self._e_pairs) < k:
            return self._e_pairs
        else:
            return self._e_pairs[:k] 
        
    def explained_variance(self):
        """Computed explained variance"""
        total_variance = sum([self._e_pairs[i][0] for i in range(len(self._e_pairs))])
        variance = [self._e_pairs[i][0]/total_variance for i in range(len(self._e_pairs))]
        return np.cumsum(variance)
        
        
    
    def SVD_matrix_decomp(self):
        # standarized
        X_norm = (X - np.mean(self.X, axis=1)) / np.std(self.X, axis=1)
        X_mean = np.mean(X_norm, axis=0) # featurewise mean
        
        # SVD 
        # X_cov = (X_norm - X_mean).T.dot(X_norm - X_mean)/(X_norm.shape[0] -1)
        U, S, V = np.linalg.svd(X_norm.T)
        # where U and V are unity matrix, dot product of the inverse
        # gives identify matrix 


class NumPyInterface(object):

    def __init__(self):
        print("init")

    def test(self):
        return "test"
