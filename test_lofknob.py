import unittest
from lofknob import lofknob
import pandas as pd
from collections import Counter
import numpy as np
from scipy.stats import norm, uniform


class ObviousOutliers(unittest.TestCase):
    def setUp(self):

        self.OUTLIER = -1
        self.INLIER = 1

    def test_one(self):

        samples = 400
        outliers = 32
        sample_contamination = outliers / samples
        _actual_contamination = 0

        while abs(_actual_contamination - sample_contamination) > 0.001:
            data = np.random.choice(
                        a=[-1, 1],
                        size=samples,
                        p=[sample_contamination, 1 - sample_contamination]).reshape(-1, 1)

            _actual_contamination = Counter(data.flatten()).get(-1, 0) / len(data)

        c_grid = [0.01, 0.02, 0.05, 0.06, 0.07, 0.10, 0.12, 0.15, 0.20]
        k_grid = [5, 8, 10, 15, 20, 30]

        if candidates_with_scores := lofknob().tune(
            X=data,
            c_grid=c_grid,
            k_grid=k_grid,
            verbose=True,
            return_scores=True,
        ):

            c_opt, k_opt, _ = max(
                candidates_with_scores, key=lambda x: x.probability_score
            )

            self.assertAlmostEqual(c_opt, _actual_contamination, places=2)

    # def test_two(self):

    #     samples = 500
    #     outliers = 25
    #     sample_contamination = outliers / samples

    #     data = pd.concat(
    #         [
    #             pd.DataFrame({"a": norm().rvs(size=samples - outliers)}),
    #             pd.DataFrame({"a": uniform(loc=5).rvs(size=outliers)}),
    #         ]
    #     )

    #     c_grid = [0.01, 0.02, 0.05, 0.06, 0.07, 0.10, 0.12, 0.15, 0.20]
    #     k_grid = [5, 8, 10, 15, 20, 30]

    #     candidates_with_scores = lofknob().tune(
    #         X=data,
    #         c_grid=c_grid,
    #         k_grid=k_grid,
    #         verbose=True,
    #         return_scores=True,
    #     )

    #     c_opt, k_opt, _ = max(candidates_with_scores, key=lambda x: x.probability_score)

    #     self.assertAlmostEqual(c_opt, sample_contamination, places=2)


if __name__ == "__main__":
    unittest.main()
