from collections import namedtuple
from typing import List, Tuple, Optional

import numpy as np    # type: ignore
import pandas as pd    # type: ignore
from scipy.stats import nct    # type: ignore
from sklearn.neighbors import LocalOutlierFactor    # type: ignore


def lofknob(
    X: pd.DataFrame,
    c_grid: Optional[List[float]] = None,
    k_grid: Optional[List[int]] = None,
) -> Tuple[float, int]:

    # scikit-learn restricts contamination to range [0, 0.50]
    MAX_CONTAMINATION = 0.50
    # minimum number of expected outlier rows
    MIN_OUTLIER_ROWS = 2
    # scikit-learn uses 20 neighbours by default
    MIN_NEIGHBORS = 5
    # very small number to use instead of zero
    VERY_SMALL = 1e-8
    # scikit-learn outlier labels
    OUTLIER_LABEL = -1
    INLIER_LABEL = 1

    if c_grid is None:
        c_grid = np.arange(
            start=0.01, stop=MAX_CONTAMINATION + 0.01, step=0.01
        ).tolist()

    if (np.min(c_grid) < 0) or (np.max(c_grid) > MAX_CONTAMINATION):
        raise Exception(f"contamination must be in [0.00, {MAX_CONTAMINATION}]!")

    if k_grid is None:
        k_grid = np.arange(start=MIN_NEIGHBORS, stop=46, step=2).tolist()

    if np.min(k_grid) < MIN_NEIGHBORS:
        raise Exception(f"number of neighbors must be at least {MIN_NEIGHBORS}!")

    n_rows = len(X.index)

    Cand = namedtuple("Cand", "c k_c_opt p_c")
    candidates = []

    for c in c_grid:

        # with contamination c, cn=floor(c*n_rows) rows to be labelled outliers
        # e.g. if n_rows=98 and c=0.12 then
        # cn=floor(c*n_rows=0.12*98=11.76)=11

        cn = np.floor(c * n_rows).astype(int)  # np.floor() returns a float

        if cn < MIN_OUTLIER_ROWS:
            continue

        out_mean_lls_all_ks = []
        out_var_lls_all_ks = []
        in_mean_lls_all_ks = []
        in_var_lls_all_ks = []

        dist_scores_all_k = []

        k_grid_len = len(k_grid)

        for k in k_grid:

            # fit LOF with c and k, return -1 if outlier and 1 otherwise
            lof = LocalOutlierFactor(contamination=c, n_neighbors=k)
            labels = lof.fit_predict(X)

            # negative_outlier_factor_ = -[LOF score]; [LOF score] for
            # outliers >> 1 and for inliers it's close to 1;
            # we calculate natural logarithms of LOF scores here
            lls = np.log(-lof.negative_outlier_factor_)

            # select natural logarithms of LOF scores for outliers
            out_lls = lls[labels == OUTLIER_LABEL]
            # ..and for inliers but these will be sorted smallest to largest;
            # then we pick the last cn of them since after the sorting
            # these will be the largest cn LOF scores
            inl_lls = np.sort(lls[labels == INLIER_LABEL])[-cn:]

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

            # measure distance between outliers and the nearest cn inliers
            if (
                diff_means_this_k := out_mean_lls_this_k - in_mean_lls_this_k
                > VERY_SMALL
            ) and (
                sum_vars_this_k := out_var_lls_this_k + in_var_lls_this_k > VERY_SMALL
            ):
                dist_score_this_k = diff_means_this_k / np.sqrt(sum_vars_this_k / cn)
            else:
                dist_score_this_k = 0

            dist_scores_all_k.append(dist_score_this_k)

        if (
            diff_mean_means_over_ks := np.mean(out_mean_lls_all_ks)
            - np.mean(in_mean_lls_all_ks)
            > VERY_SMALL
        ) and (
            sum_mean_vars_over_ks := np.mean(out_var_lls_all_ks)
            + np.mean(in_var_lls_all_ks)
            > VERY_SMALL
        ):
            ncp_c = diff_mean_means_over_ks / np.sqrt(sum_mean_vars_over_ks / cn)
        else:
            ncp_c = 0

        # now we want to find out which choice of k resulted in
        # the largest distance score;
        # first we find index where the largest (optimal) score is sitting
        idx_opt_dist_score = np.argmax(dist_scores_all_k).tolist()
        # ..then we pick the value of the largest score..
        opt_dist_score = dist_scores_all_k[idx_opt_dist_score]
        # ..and finally pick the value of k corresponding to this score
        k_opt_this_c = k_grid[idx_opt_dist_score]

        # degrees of freedom
        df_this_c = 2 * cn - 2

        p_c = nct.cdf(x=opt_dist_score, df=df_this_c, nc=ncp_c, loc=0, scale=1)

        candidates.append(Cand(c=c, k_c_opt=k_opt_this_c, p_c=p_c))

    # now that we've gone through all combinations of c and k,
    # find optimal c_opt - it's the one corresponding to the largest p_c
    c_opt, k_opt, _ = max(candidates, key=lambda x: x.p_c)

    return (c_opt, k_opt)
