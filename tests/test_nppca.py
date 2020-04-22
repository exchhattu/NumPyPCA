
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
    x_trans_scikit = pca.transform(X)
    print(pca.explained_variance_)
    y1 = pca.explained_variance_

    # this is a class to compute PCA using numpy libraries
    # and linear algebra 
    oj_np_pca = np_pca.npPCA() 
    oj_np_pca.load_data(target_column_idx=4)
    X_trans_np = oj_np_pca.fit()

    # compare transformed matrix 
    npt.assert_array_equal(X_trans_np.shape, x_trans_scikit.shape)
    npt.assert_almost_equal(np.abs(X_trans_np), np.abs(x_trans_scikit), decimal=4)

    # y = oj_np_pca.explained_variance(3)
    # y3 = oj_np_pca.largest_eigenvalues(3)
    # y2 = pca.explained_variance_ratio_
    # print(y)
    # print(y2)
    # print(y3)
    # npt.assert_array_equal(y1, y)


if __name__ == '__main__':
    test_scikitlearn()