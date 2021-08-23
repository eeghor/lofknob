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

# https://github.com/scikit-learn/scikit-learn/blob/114616d9f6ce9eba7c1aacd3d4a254f868010e25/examples/neighbors/plot_lof_outlier_detection.py

np.random.seed(42)

# Generate train data
X_inliers = 0.3 * np.random.randn(100, 2)
X_inliers = np.r_[X_inliers + 2, X_inliers - 2]

# Generate some outliers
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = pd.DataFrame(np.r_[X_inliers, X_outliers])

n_outliers = len(X_outliers)
ground_truth = np.ones(len(X), dtype=int)
ground_truth[-n_outliers:] = -1

if res := lofknob().tune(X, c_grid, k_grid):
    c_opt, k_opt = res
    print(f'optimal contamination: {c_opt:.4f}, number of neighbours: {k_opt:.0f}')
else:
    print("optimisation failed")

print(f"outliers: {n_outliers}, contamination: {n_outliers/len(X_inliers):.4f}")

# fit the model for outlier detection (default)
# clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the estimator has no predict,
# decision_function and score_samples methods).
# y_pred = clf.fit_predict(X)
# n_errors = (y_pred != ground_truth).sum()
# X_scores = clf.negative_outlier_factor_