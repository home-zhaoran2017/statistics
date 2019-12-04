from collections import Counter
import numpy as np
from scipy import special

class ChiSquareTest():
    def _cdf(x,k):
        return special.gammainc(k/2,x/2)

    @classmethod
    def chi_p(cls, X, Y):
        X=np.array(X)
        Y=np.array(Y)
        N=X.shape[0]
        x_uniq = list(np.unique(X))
        y_uniq = list(np.unique(Y))
        I = len(x_uniq)
        J = len(y_uniq)

        O = np.zeros((I,J),dtype=int)
        E = np.zeros((I,J),dtype=int)

        for j,y_v in enumerate(y_uniq):
            count = Counter(X[Y==y_v])
            for item in count.items():
                i=x_uniq.index(item[0])
                O[i,j]=item[1]

        R = np.sum(O,axis=1)
        R = R.reshape((len(R),1))
        C = np.sum(O,axis=0)
        C = C.reshape((1,len(C)))

        E = np.kron(C,R)/float(N)

        chi = (O-E)**2/E
        chi = np.sum(chi)

        p_value = 1-cls._cdf(chi,(I-1)*(J-1))

        return chi, p_value
