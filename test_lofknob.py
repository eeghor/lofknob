import unittest
from lofknob import lofknob
import pandas as pd
from collections import Counter
import numpy as np

class ObviousOutliers(unittest.TestCase):

    def setUp(self):

        self.OUTLIER = -1
        self.INLIER = 1
        self.OUTLIER_PROPORTION = 0.06
        self.TOTAL_DATAPOINTS = 400

        while 1:
            self.data = pd.DataFrame({'a': np.random.choice(a=[self.OUTLIER, self.INLIER], 
                                                            size=self.TOTAL_DATAPOINTS, 
                                                            p=[self.OUTLIER_PROPORTION, 
                                                            1 - self.OUTLIER_PROPORTION])})

            self.REAL_CONTAMINATION = Counter(self.data['a']).get(self.OUTLIER, 0)/len(self.data)
            if abs(self.REAL_CONTAMINATION - self.OUTLIER_PROPORTION) < 0.001:
                break

        self.c_grid = np.linspace(0.01, 0.20, 20)
        self.k_grid = [5, 10, 15, 20, 25, 30]

    def test_sanity(self):

        self.assertAlmostEqual(self.OUTLIER_PROPORTION, self.REAL_CONTAMINATION, places=2)
        self.assertLessEqual(max(self.c_grid), 0.5)
        self.assertGreater(min(self.c_grid), 0)

    def test_optimal_contamination(self):

        c_opt, k_opt = lofknob().tune(X=self.data, c_grid=self.c_grid, k_grid=self.k_grid, verbose=True)
        print(f"found c_opt={c_opt}, k_opt={k_opt}")

        self.assertAlmostEqual(c_opt, self.REAL_CONTAMINATION, places=2)


if __name__ == "__main__":
    unittest.main()

# print(f"total data points: {len(X.index)}")
# print(f"real contamination: {_counter[-5]/len(X.index):.4f}, {_counter[-5]} samples")
# print()

# # https://github.com/scikit-learn/scikit-learn/blob/114616d9f6ce9eba7c1aacd3d4a254f868010e25/examples/neighbors/plot_lof_outlier_detection.py

# np.random.seed(42)

# # Generate train data
# X_inliers = 0.3 * np.random.randn(100, 2)
# X_inliers = np.r_[X_inliers + 2, X_inliers - 2]

# # Generate some outliers
# X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
# X = pd.DataFrame(np.r_[X_inliers, X_outliers])

# n_outliers = len(X_outliers)
# ground_truth = np.ones(len(X), dtype=int)
# ground_truth[-n_outliers:] = -1

# if res := lofknob().tune(X, c_grid, k_grid, verbose=True):
#     c_opt, k_opt = res
#     print(f'optimal contamination: {c_opt:.4f}, number of neighbours: {k_opt:.0f}')
# else:
#     print("optimisation failed")

# print(f"real contamination: {n_outliers/len(X_inliers):.4f}, outliers: {n_outliers}")

# # fit the model for outlier detection (default)
# # clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# # use fit_predict to compute the predicted labels of the training samples
# # (when LOF is used for outlier detection, the estimator has no predict,
# # decision_function and score_samples methods).
# # y_pred = clf.fit_predict(X)
# # n_errors = (y_pred != ground_truth).sum()
# # X_scores = clf.negative_outlier_factor_