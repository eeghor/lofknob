import unittest
from lofknob import lofknob
import numpy as np
from scipy.stats import uniform


class FindObviousOutliers(unittest.TestCase):

    def test_one(self):

        # total samples to generate
        SAMPLES = 50
        # we want this many outliers
        OUTLIERS = 10

        _actual_contamination = OUTLIERS/SAMPLES
        
        # data
        X = np.hstack((np.ones(SAMPLES - OUTLIERS), -np.ones(OUTLIERS) + np.random.normal(0,1e-6, OUTLIERS))).reshape(-1, 1) 

        c_grid = [0.01, 0.02, 0.05, 0.06, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25]
        k_grid = [5, 8, 10, 15, 20, 25, 30]

        if candidates_with_scores := lofknob().tune(
            X=X,
            c_grid=c_grid,
            k_grid=k_grid,
            verbose=True,
            return_scores=True,
        ):

            c_opt, k_opt, _ = max(
                candidates_with_scores, key=lambda x: x.probability_score
            )

            self.assertAlmostEqual(c_opt, _actual_contamination, places=2)

    def test_two(self):

        SAMPLES = 500
        OUTLIERS = 25

        _actual_contamination = OUTLIERS / SAMPLES

        X = np.hstack((uniform().rvs(size=SAMPLES - OUTLIERS),
                       uniform(loc=3).rvs(size=OUTLIERS))).reshape(-1,1)

        c_grid = [0.01, 0.05, 0.06, 0.07, 0.10, 0.12, 0.15, 0.20]
        k_grid = [10, 12, 14, 20, 30, 50]

        c_opt, k_opt = lofknob().tune(
            X=X,
            c_grid=c_grid,
            k_grid=k_grid,
            verbose=True,
            return_scores=False,
        )

        self.assertAlmostEqual(c_opt, _actual_contamination, places=2)


if __name__ == "__main__":
    unittest.main()
