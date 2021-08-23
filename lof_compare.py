from lofknob import lofknob
import pandas as pd
from collections import Counter
import numpy as np

X = pd.DataFrame({'a': np.random.choice(a=[-5,1], size=370, p=[0.06, 0.94])})

_counter = Counter(X['a'])

c_grid = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.15, 0.20]
k_grid = list(range(5,31))

if res := lofknob().tune(X, c_grid, k_grid):
    c_opt, k_opt = res
    print(f'optimal contamination: {c_opt:.4f}, number of neighbours: {k_opt:.0f}')
else:
    print("optimisation failed")

print(f"total data points: {len(X.index)}")
print(f"contamination: {_counter[-5]} samples, {_counter[-5]/len(X.index):.4f}")
print()