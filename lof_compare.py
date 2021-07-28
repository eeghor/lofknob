import lofknob
import pandas as pd
import numpy as np

X = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
c_grid = list(np.arange(0.01, 0.20, 0.02))
print('c_grid=', c_grid)
k_grid = list(range(5, 30, 5))
print('k_grid=', k_grid)

print(lofknob.lofknob().tune(X, c_grid, k_grid))
