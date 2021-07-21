### Automatic Hyperparameter Tuning Method for Local Outlier Factor

Implementation of an algorithm described in [this](https://arxiv.org/abs/1902.00567) paper.  It’s using LOF  from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor)  where the two parameters being optimised are  

* **n_neighbors** and
* **contamination**

The algorithm is trying to find a parameter combination that gives most clear-cut outliers.