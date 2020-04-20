
import numpy as np
import numpy.testing as npt

import np_pca
 
def test_np_pca():
    oj_np_pca = np_pca.npPCA() 
    oj_np_pca.load_data(target_column_idx=4)
    oj_np_pca.fit()

    # compare the eigenvalues, which are variance of
    # principal axes

    # only two first components selected
    l_evalues = oj_np_pca.largest_eigenvalues(2)
    npt.assert_equal(len(l_evalues), 2)

    # first variance and corresponding vectors  
    npt.assert_almost_equal(2.9244283,l_evalues[0][0], decimal=4)
    gtruth_evals = [ 0.52308496, -0.25956935,  0.58184289,  0.56609604]
    npt.assert_almost_equal(l_evalues[0][1], gtruth_evals, decimal=4)

    # second variance and corresponding vectors  
    npt.assert_almost_equal(0.9321523302535063, l_evalues[1][0], decimal=4)
    gtruth_evals = [-0.36956962, -0.92681168, -0.01912775, -0.06381646]
    npt.assert_almost_equal(l_evalues[1][1], gtruth_evals, decimal=4)

    # compare the cumsum of variances, the better 
    # coverage by first few components are good 
    cum_sums = oj_np_pca.explained_variance(2)
    gtruth_csums = [0.72620033, 0.9576744]
    npt.assert_almost_equal(cum_sums, gtruth_csums, decimal=4)

if __name__ == '__main__':
    test_np_pca()