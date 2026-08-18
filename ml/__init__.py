"""``ml`` package — HousingIQ model code.

Sub-packages:
    - cleaning       (parse + dedup + outlier flagging + parquet writer)
    - features       (schema mapping, fit_preprocessor, persistence, split)
    - training       (regression + classification training scripts)
    - evaluation     (Spec 15 — fixed protocol gate, see ml.evaluation)
    - recommender    (TF-IDF + NearestNeighbors index)
"""

from ml import evaluation  # noqa: F401  (Spec 15 — fixed evaluation gate)
