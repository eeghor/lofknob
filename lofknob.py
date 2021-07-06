import numpy as np
from scipy.stats import nct
from sklearn.neighbors import LocalOutlierFactor

def lofknob(X, c_grid=None, k_grid=None):

    c_grid = np.linspace(0.01, 0.15, num=10)
    k_grid = np.linspace(10, 50, num=20)
    
    # number of rows
    n = X.shape[0]
    
    candidates = []
    
    for c in c_grid:
        
        # with contamination c, floor(c*n) rows will be labelled as outliers; e.g. if n=98 and c=0.12 then 
        # floor(c*n=0.12*98=11.76)=11 rows will be ouliers
    
        cn = np.floor(c*n)
    
        m_c_out = v_c_out = 0
        m_c_in = v_c_in = 0
        
        t_c = np.zero_like(k_grid)
    
        k_grid_len = k_grid.shape[1]
        
        for k in k_grid:
            
            # fit LOF with c and k and return labels, -1 if outlier and 1 otherwise
            lof = LocalOutlierFactor(contamination=c, n_neighbors=k).fit(X)
            labels = lof.predict(X)
            
            # ln(LOFs for each row)
            ln_lofs = np.log(-lof.negative_outlier_factor_)
            # LOFs for outliers
            out_ln_lofs = ln_lofs[labels==-1] 
            # top cn LOFs for inliers
            in_ln_lofs = np.sort(ln_lofs[labels==1])[-cn:]
            
            # mean of these outlier LOFs
            m_c_k_out = np.mean(out_ln_lofs)
            m_c_out += m_c_k_out/k_grid_len
            # and their variance
            v_c_k_out = np.var(out_ln_lofs)
            v_c_out += v_c_k_out/k_grid_len
            
            m_c_k_in = np.mean(in_ln_lofs)
            m_c_in += m_c_k_in/k_grid_len
            v_c_k_in = np.var(in_ln_lofs)
            v_c_in += v_c_k_in/k_grid_len
            
            t_c_k = (m_c_k_out - m_c_k_in)/np.sqrt((v_c_k_out + v_c_k_in)/cn)
            
            t_c = t_c.append(t_c_k)
            
        # non-centrality parameter
        ncp_c = (m_c_out - m_c_in)/np.sqrt((v_c_out + v_c_in)/cn)
    
        # degrees of freedom
        df_c = 2*cn - 2
    
        t_c_opt_idx = np.argmax(t_c)
        t_c_k_opt = t_c[t_c_opt_idx]
        k_c_opt = k_grid[t_c_opt_idx]
    
        p_c = nct.cdf(x=t_c_k_opt, df=df_c, nc=ncp_c, loc=0, scale=1)
    
        candidates.append((c, k_c_opt, p_c))
    
    # find optimal c
    c_opt, k_opt, _ = max(candidates, key=lambda x: x[2])

    return (c_opt, k_opt)
    

        