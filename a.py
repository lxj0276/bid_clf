import scipy.sparse as sp
import numpy as np
c= sp.csr_matrix([[1,0,2],[1,0,0],[1,0,0]], dtype=np.float64, copy=True)
print(c.data)
c.toarray()