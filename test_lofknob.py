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
        self.EXPECTED_OUTLIERS = int(
            np.floor(self.OUTLIER_PROPORTION * self.TOTAL_DATAPOINTS)
        )

        while 1:
            self.data = pd.DataFrame(
                {
                    "a": np.random.choice(
                        a=[self.OUTLIER, self.INLIER],
                        size=self.TOTAL_DATAPOINTS,
                        p=[self.OUTLIER_PROPORTION, 1 - self.OUTLIER_PROPORTION],
                    )
                }
            )

            self.REAL_CONTAMINATION = Counter(self.data["a"]).get(
                self.OUTLIER, 0
            ) / len(self.data)
            if abs(self.REAL_CONTAMINATION - self.OUTLIER_PROPORTION) < 0.001:
                break

        self.c_grid = np.linspace(0.01, 0.20, 20)
        self.k_grid = [5, 10, 15, 20, 25, 30]

    def test_settings_sanity(self):

        self.assertAlmostEqual(
            self.OUTLIER_PROPORTION, self.REAL_CONTAMINATION, places=2
        )
        self.assertEqual(
            self.EXPECTED_OUTLIERS / self.TOTAL_DATAPOINTS, self.REAL_CONTAMINATION
        )
        self.assertLessEqual(max(self.c_grid), 0.5)
        self.assertGreater(min(self.c_grid), 0)

    def test_optimal_contamination(self):

        c_opt, k_opt = lofknob().tune(
            X=self.data, c_grid=self.c_grid, k_grid=self.k_grid, verbose=True
        )
        print(f"optimal contamination: {c_opt}, optimal neighbours: {k_opt}")

        self.assertAlmostEqual(c_opt, self.REAL_CONTAMINATION, places=2)

    def test_optimal_contamination_scores(self):

        if candidates_with_scores := lofknob().tune(
            X=self.data,
            c_grid=self.c_grid,
            k_grid=self.k_grid,
            verbose=True,
            return_scores=True,
        ):

            c_opt, k_opt, _ = max(
                candidates_with_scores, key=lambda x: x.probability_score
            )

            print(candidates_with_scores)
            print(f"optimal contamination: {c_opt}, optimal neighbours: {k_opt}")

            self.assertAlmostEqual(c_opt, self.REAL_CONTAMINATION, places=2)


if __name__ == "__main__":
    unittest.main()
