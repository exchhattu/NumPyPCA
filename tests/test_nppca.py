
import numpy as np
import pandas as pd
import numpy.testing as npt
from sklearn.decomposition import PCA

import np_pca
 
def test_scikitlearn():
    """ compare the result of PCA using numpy library with scikit. 
        parse data and use it first with scikit learn and then
        compare with result from the library using covariance 
        matrix
    """

    target_column_idx = 4
    path_to_file = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    dtypes = {
                "sepallength": np.float64,
                "sepalwidth": np.float64,
                "petallength": np.float64,
                "petalwidth": np.float64,
                "target": np.str,
            }
    df_data = pd.read_csv(path_to_file, dtype=dtypes)

    # find the column indexes of dependent variables
    columns_idxes = [
        i for i in range(len(df_data.columns)) if i != target_column_idx
    ]

    # Dependent variables are set to X and target variable is ignored 
    X = df_data.iloc[:, columns_idxes].values
    print("Unlucky ", X.shape)
    pca = PCA()
    pca.fit(X)
    X_trans_scikit  = pca.transform(X)
    variance_scikit = pca.explained_variance_
    comp_scikit     = pca.components_

    # this is a class to compute PCA using numpy libraries
    # and linear algebra 
    oj_np_pca = np_pca.npPCA() 
    oj_np_pca.load_data(target_column_idx=4)
    X_trans_np = oj_np_pca.fit()

    # compare two transformed matrixes 
    npt.assert_array_equal(X_trans_np.shape, X_trans_scikit.shape)
    npt.assert_almost_equal(np.abs(X_trans_np), np.abs(X_trans_scikit), decimal=4)

    # compare eigen values and their corresponding vectors
    # eigen vector of np implementation should be transposed 
    npt.assert_almost_equal(variance_scikit, oj_np_pca._ei_vals)
    npt.assert_almost_equal(np.abs(comp_scikit), np.abs(oj_np_pca._ei_vecs.T))


if __name__ == '__main__':
    test_scikitlearn()