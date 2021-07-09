import numpy as np
from scipy.stats import nct
from collections import namedtuple
from sklearn.neighbors import LocalOutlierFactor
from typing import Tuple, Iterable


def lofknob(X, c_grid: Iterable = None, k_grid: Iterable = None) -> Tuple[float, int]:

    if not c_grid:
        c_grid = np.linspace(0.01, 0.15, num=10)
    if not k_grid:
        k_grid = np.arange(10, 50, step=2)

    if (np.min(c_grid) < 0) or (np.max(c_grid) > 0.50):
        raise Exception(f"ERROR: contamination must be in [0.00, 0.50]!")

    if np.min(k_grid) < 1:
        raise Exception(f"ERROR: number of neighbors must be at least 1!")

    n = X.shape[0]  # number of rows

    Cand = namedtuple("Cand", "c k_c_opt p_c")
    candidates = []

    VERY_SMALL = 1e-8

    for c in c_grid:

        # with contamination c, floor(c*n) rows will be labelled as outliers; e.g. if n=98 and c=0.12 then
        # floor(c*n=0.12*98=11.76)=11 rows will be ouliers

        cn = np.floor(c * n).astype(int)  # np.floor() returns a float

        if cn < 2:
            continue

        m_c_out = v_c_out = m_c_in = v_c_in = 0

        t_c = np.empty_like(k_grid)

        k_grid_len = k_grid.shape[0]

        for k in k_grid:

            # fit LOF with c and k and return labels: -1 if outlier and 1 otherwise
            lof = LocalOutlierFactor(contamination=c, n_neighbors=k)
            labels = lof.fit_predict(X)

            # ln(LOFs for each row)
            ln_lofs = np.log(-lof.negative_outlier_factor_)

            # LOFs for outliers
            out_ln_lofs = ln_lofs[labels == -1]
            # top cn LOFs for inliers
            in_ln_lofs = np.sort(ln_lofs[labels == 1])[-cn:]

            # mean and variance of outlier LOFs
            m_c_k_out = np.mean(out_ln_lofs)
            m_c_out += m_c_k_out / k_grid_len
            v_c_k_out = np.var(out_ln_lofs)
            v_c_out += v_c_k_out / k_grid_len

            # mean and variance of inlier LOFs
            m_c_k_in = np.mean(in_ln_lofs)
            m_c_in += m_c_k_in / k_grid_len
            v_c_k_in = np.var(in_ln_lofs)
            v_c_in += v_c_k_in / k_grid_len

            t_c_k = (
                (m_c_k_out - m_c_k_in) / np.sqrt((v_c_k_out + v_c_k_in) / cn)
                if (m_c_k_out - m_c_k_in) * (v_c_k_out + v_c_k_in) > VERY_SMALL
                else 0
            )

            t_c = np.append(t_c, t_c_k)

        # non-centrality parameter
        ncp_c = (
            (v_c_out + v_c_in) / np.sqrt((v_c_out + v_c_in) / cn)
            if (v_c_out + v_c_in) * (v_c_out + v_c_in) > VERY_SMALL
            else 0
        )

        # degrees of freedom
        df_c = 2 * cn - 2

        t_c_opt_idx = np.argmax(t_c)
        t_c_k_opt = t_c[t_c_opt_idx]
        k_c_opt = k_grid[t_c_opt_idx]

        p_c = nct.cdf(x=t_c_k_opt, df=df_c, nc=ncp_c, loc=0, scale=1)

        candidates.append(Cand(c=c, k_c_opt=k_c_opt, p_c=p_c))

    # find optimal c_opt, it's the one corresponding to the largest p_c
    c_opt, k_opt, _ = max(candidates, key=lambda x: x.p_c)

    return (c_opt, k_opt)
