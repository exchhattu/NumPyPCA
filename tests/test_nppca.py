
import numpy as np
import numpy.testing as npt

import np_pca
 
def test_np_pca():
    oj_np_pca = np_pca.npPCA() 
    oj_np_pca.load_data()
    oj_np_pca.fit()

    # compare the eigenvalues, which are variance of
    # principal axes
    l_evalues = oj_np_pca.largest_eigenvalues(2)
    gtruth_evals = [0.7262, 0.2314, 0.0371, 0.0052]
    npt.assert_allclose(gtruth_evals, l_evalues,rtol=1e-10, atol=0)

    # compare the cumsum of variances, the better 
    # coverage by first few components are good 
    cum_sums = oj_np_pca.explained_variance(2)
    gtruth_csums = [0.7262, 0.9576, 0.9947, 1.]
    npt.assert_allclose(gtruth_csums, cum_sums, rtol=1e-10, atol=0)

if __name__ == '__main__':
    test_np_pca()