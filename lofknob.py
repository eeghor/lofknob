import numpy as np
from sklearn.neighbors import LocalOutlierFactor

c_grid = np.linspace(0.01, 0.15, num=10)
k_grid = np.linspace(10, 50, num=20)

# data
X
# number of rows
n = X.shape[0]

for c in c_grid:
    # with this contamination, floor(c*n) rows will be labelled as outliers; e.g. if n=98 and c=0.12 then 
    # floor(c*n=0.12*98=11.76)=11 rows will be ouliers
    cn = np.floor(c*n)
    for k in k_grid:
        # fit LOF with the current c and k and return labels, -1 if outlier and 1 otherwise
        lof = LocalOutlierFactor(contamination=c, n_neighbors=k).fit(X)
        labels = lof.predict(X)
        # ln(LOFs for each row)
        ln_lofs = np.log(-lof.negative_outlier_factor_)
        # LOFs for outliers
        out_ln_lofs = ln_lofs[labels==-1] 
        # mean of these outlier LOFs
        m_c_k_out = np.mean(out_ln_lofs)
        # and their variance
        v_c_k_out = np.var(out_ln_lofs)
        # top cn LOFs for inliers
        in_ln_lofs = np.sort(ln_lofs[labels==1])[-cn:]
        # mean of these inlier LOFs
        m_c_k_in = np.mean(in_ln_lofs)
        # and their variance
        v_c_k_in = np.var(in_ln_lofs)
        # standardized difference
        t_c_k = (m_c_k_out - m_c_k_in)/np.sqrt((v_c_k_out + v_c_k_in)/cn)
        