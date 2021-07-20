import numpy as np
import pandas as pd
from scipy.stats import nct
from collections import namedtuple
from sklearn.neighbors import LocalOutlierFactor
from typing import List, Tuple


def lofknob(
    X: pd.DataFrame,
    c_grid: List[float],
    k_grid: List[int],
) -> Tuple[float, int]:

    if (np.min(c_grid) < 0) or (np.max(c_grid) > 0.50):
        raise Exception("ERROR: contamination must be in [0.00, 0.50]!")

    if np.min(k_grid) < 1:
        raise Exception("ERROR: number of neighbors must be at least 1!")

    n_rows = len(X.index)

    Cand = namedtuple("Cand", "c k_c_opt p_c")
    candidates = []

    VERY_SMALL = 1e-8

    OUTLIER_LABEL = -1
    INLIER_LABEL = 1

    for i in range(len(c_grid)):

        c = c_grid[i]

        # with contamination c, floor(c*n_rows) rows to be labelled as outliers
        # e.g. if n_rows=98 and c=0.12 then
        # floor(c*n_rows=0.12*98=11.76)=11 rows will be ouliers

        cn = np.floor(c * n_rows).astype(int)  # np.floor() returns a float

        if cn < 2:
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
            if ((diff_means_this_k := out_mean_lls_this_k - in_mean_lls_this_k > VERY_SMALL) and 
                    (sum_vars_this_k := out_var_lls_this_k + in_var_lls_this_k> VERY_SMALL)):
                dist_score_this_k = diff_means_this_k / np.sqrt(sum_vars_this_k / cn)
            else:
                dist_score_this_k = 0

            dist_scores_all_k.append(dist_score_this_k)

        if ((diff_mean_means_over_ks := np.mean(out_mean_lls_all_ks) - np.mean(in_mean_lls_all_ks) > VERY_SMALL) and 
                (sum_mean_vars_over_ks := np.mean(out_var_lls_all_ks) + np.mean(in_var_lls_all_ks) > VERY_SMALL)):
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
