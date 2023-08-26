import numpy as np
from scipy.stats import pearsonr

if __name__ == '__main__':
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)

    print(X.shape, y.shape)

    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y must have the same number of rows.')
    
    print('X and y have the same number of rows.')

    corr, p = pearsonr(X[:, 0], y)

    print('Pearsons correlation: %.3f' % corr)

