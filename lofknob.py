from typing import List, Tuple, Optional, NamedTuple, Union, Dict
from operator import attrgetter
import heapq
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.stats import nct  # type: ignore
from sklearn.neighbors import LocalOutlierFactor  # type: ignore
from collections import Counter


class lofknob:

    """
    Hyperparameter Tuning for the Local Outlier Factor Algorithm
    (see the original paper at https://arxiv.org/pdf/1902.00567.pdf)

    usage example
    -------------

    optimal_contamination, optimal_number_of_neighbours =
                                                                    lofknob().tune(X, c_grid, k_grid, return_scores, min_outlier_rows, useful_digits, verbose)

    where

    X:
       is a pandas DataFrame with training data
    c_grid:
       is a list of contamination values to try, e.g. [0.01, 0.02, ..]
    k_grid:
       is a list of the number of neighbours to try, e.g. [5, 10, 20, ..]
    return_scores:
       if True, return a list of parameter pairs competing for being "optimal" along
       with their probability scores; default: False
    min_outlier_rows:
       minimum number of expected outlier rows for any selected contamination;
       default: 2
    useful_digits:
       how many decimal digits to round to in output;
       default: 6
    verbose:
       if True, show some intermediate output;
       default: False
    """

    # scikit-learn restricts contamination to range (0, 0.5]
    MIN_CONTAMINATION = 0.001
    MAX_CONTAMINATION = 0.500
    # scikit-learn uses 20 neighbours by default
    MIN_NEIGHBORS = 5
    # very small number to use instead of zero
    VERY_SMALL = 1e-8
    # scikit-learn outlier labels
    OUTLIER_LABEL = -1
    INLIER_LABEL = 1

    class Candidate(NamedTuple):
        contamination: float
        number_of_neighbours: float
        probability_score: float

    def tune(
        self,
        X: pd.DataFrame,
        c_grid: Optional[List[float]] = None,
        k_grid: Optional[List[int]] = None,
        return_scores: bool = False,
        min_outlier_rows: int = 2,
        useful_digits: int = 6,
        verbose: bool = False,
    ) -> Union[Tuple[float, int], List[Candidate], None]:

        self.min_outlier_rows = min_outlier_rows
        self.useful_digits = useful_digits

        if c_grid is None:
            c_grid = np.linspace(
                start=lofknob.MIN_CONTAMINATION, stop=lofknob.MAX_CONTAMINATION, num=20
            ).tolist()

        if ((c_min := np.min(c_grid)) < lofknob.MIN_CONTAMINATION) or (
            (c_max := np.max(c_grid)) > lofknob.MAX_CONTAMINATION
        ):
            raise Exception(
                f"contamination in [{c_min}, {c_max}] while it must be in [{lofknob.MIN_CONTAMINATION}, {lofknob.MAX_CONTAMINATION}]!"
            )

        if k_grid is None:
            k_grid = np.arange(
                start=lofknob.MIN_NEIGHBORS,
                stop=(10 + 1) * lofknob.MIN_NEIGHBORS,
                step=lofknob.MIN_NEIGHBORS,
            ).tolist()

        if (k_min := np.min(k_grid)) < lofknob.MIN_NEIGHBORS:
            raise Exception(
                f"number of neighbors is {k_min} while it must be at least {lofknob.MIN_NEIGHBORS}!"
            )

        n_rows = len(X.index)

        candidates = []

        for c in c_grid:

            # with contamination c, expected_outlier_rows=floor(c*n_rows)
            # e.g. if n_rows=98 and c=0.12 then
            # expected_outlier_rows=floor(c*n_rows=0.12*98=11.76)=11

            expected_outlier_rows = np.floor(c * n_rows).astype(
                int
            )  # np.floor() returns a float

            if expected_outlier_rows < self.min_outlier_rows:
                continue

            out_mean_lls_all_ks = []
            out_var_lls_all_ks = []
            in_mean_lls_all_ks = []
            in_var_lls_all_ks = []

            # well store pairs (k, score) here
            dist_scores_all_k = []

            for k in k_grid:

                # fit LOF with c and k, return -1 if outlier and 1 otherwise
                lof = LocalOutlierFactor(contamination=c, n_neighbors=k, novelty=False)
                labels = lof.fit_predict(X)

                label_counter = Counter(labels)  # type:ignore

                number_of_predicted_outliers = label_counter.get(
                    lofknob.OUTLIER_LABEL, 0
                )
                if verbose:
                    print(
                        f"LocalOutlierFactor predicted {number_of_predicted_outliers} outliers: cont. {c:.4f}, exp. outliers {expected_outlier_rows}, neighbours: {k}"
                    )

                if not number_of_predicted_outliers:
                    continue

                if not label_counter.get(lofknob.INLIER_LABEL, None):
                    if verbose:
                        print(
                            f"LocalOutlierFactor found no inliers for contamination {c:.4f} and {k} neighbours"
                        )
                    continue

                # negative_outlier_factor_ = -[LOF score]; [LOF score] for
                # outliers >> 1 and for inliers it's close to 1;
                # we calculate natural logarithms of LOF scores here
                lof_scores = -lof.negative_outlier_factor_
                lof_scores = np.where(
                    lof_scores < lofknob.VERY_SMALL, lofknob.VERY_SMALL, lof_scores
                )

                lls = np.log(lof_scores)

                # select natural logarithms of LOF scores for outliers
                out_lls = lls[labels == lofknob.OUTLIER_LABEL]

                # ..and for inliers but these will be sorted smallest to largest;
                # then we pick the last expected_outlier_rows of them since after the sorting
                # these will be the largest expected_outlier_rows LOF scores
                inl_lls = heapq.nlargest(
                    expected_outlier_rows, lls[labels == lofknob.INLIER_LABEL]
                )

                # calculate mean and variance of lls for outliers
                out_mean_lls_this_k = np.mean(out_lls).tolist()
                out_mean_lls_all_ks.append(out_mean_lls_this_k)
                out_var_lls_this_k = np.var(out_lls).tolist()
                out_var_lls_all_ks.append(out_var_lls_this_k)

                # ..and inliers
                in_mean_lls_this_k = np.mean(inl_lls).tolist()
                in_mean_lls_all_ks.append(in_mean_lls_this_k)
                in_var_lls_this_k = np.var(inl_lls).tolist()
                in_var_lls_all_ks.append(in_var_lls_this_k)

                # measure distance between outliers and the nearest expected_outlier_rows inliers
                if (
                    diff_means_this_k := out_mean_lls_this_k - in_mean_lls_this_k
                ) > lofknob.VERY_SMALL:

                    sum_vars_this_k = (
                        out_var_lls_this_k + in_var_lls_this_k + lofknob.VERY_SMALL
                    )
                    dist_score_this_k = diff_means_this_k / np.sqrt(
                        sum_vars_this_k / expected_outlier_rows
                    )

                else:
                    dist_score_this_k = 0

                dist_scores_all_k.append((k, dist_score_this_k))

            if not out_mean_lls_all_ks:
                if verbose:
                    print(f"no outliers found for contamination {c}")
                continue

            if (
                diff_mean_means_over_ks := np.mean(out_mean_lls_all_ks)
                - np.mean(in_mean_lls_all_ks)
            ) > lofknob.VERY_SMALL:

                sum_var_means = (
                    np.mean(out_var_lls_all_ks)
                    + np.mean(in_var_lls_all_ks)
                    + lofknob.VERY_SMALL
                )
                ncp_c = diff_mean_means_over_ks / np.sqrt(
                    sum_var_means / expected_outlier_rows
                )

            else:
                ncp_c = 0

            # now we want to find out which choice of k resulted in
            # the largest distance score (and what that score was);
            # priority to smalled k's
            if dist_scores_all_k:
                k_opt_this_c, opt_dist_score = max(
                    sorted(dist_scores_all_k, key=lambda _: _[0]), key=lambda x: x[1]
                )
            else:
                continue

            # degrees of freedom: must be positive, otherwise nct.cdf just returns nans
            if (
                degrees_of_freedom := 2 * (expected_outlier_rows - 1)
            ) <= lofknob.VERY_SMALL:
                return None
            else:

                p_c = nct.cdf(x=opt_dist_score, df=degrees_of_freedom, nc=ncp_c)

                candidates.append(
                    lofknob.Candidate(
                        contamination=round(c, self.useful_digits),
                        number_of_neighbours=k_opt_this_c,
                        probability_score=round(p_c, self.useful_digits),
                    )
                )

        # now that we've gone through all combinations of c and k,
        # find optimal c - it's the one corresponding to the largest probability_score
        if candidates:
            if return_scores:
                return sorted(
                    sorted(candidates, key=lambda x: x.contamination),
                    key=lambda x: x.probability_score,
                    reverse=True,
                )
            else:
                return attrgetter("contamination", "number_of_neighbours")(  # type: ignore
                    max(candidates, key=lambda x: x.probability_score)
                )
        else:
            return None
