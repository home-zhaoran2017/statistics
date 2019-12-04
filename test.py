import numpy as np
import pandas as pd

from statistical_test import ChiSquareTest

data = pd.read_csv("mess_time_feature",header=None,sep='|',dtype=object)
d=data.iloc[:,1:].values
d = np.array(d,dtype=float)
d = np.array(d,dtype=int)

Y=d[:,-1]

for n in range(280):
    X=d[:,n]
    chi, p = ChiSquareTest.chi_p(X,Y)
    print("%4d %12.6f %.6f"%(n, chi,p))
