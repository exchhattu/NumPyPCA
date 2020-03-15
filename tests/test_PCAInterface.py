
import numpy as np
import numpy.testing as npt

import PCA 
 
def test_class():
    a = PCA.NumPyInterface() 
    val = a.test()
    npt.assert_equal(val, "test") 
