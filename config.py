"""Central configuration. Everything is tuned to be light on an old laptop."""
import os
from pathlib import Path

# Keep CPU/memory usage low: cap BLAS/OpenMP threads BEFORE numpy/xgboost are imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DATA_DIR = ROOT / "data"

SEED = 42
N_JOBS = 2                 # max CPU threads used by any model

# --- synthetic data -------------------------------------------------------
N_TXN = 60_000
FRAUD_RATE = 0.008         # 0.8 % -> "fraud is rare (<1%)"

# --- time-based split (no shuffling => no look-ahead leakage) -------------
TRAIN_FRAC, VAL_FRAC = 0.60, 0.20     # remaining 20 % = test

# --- business cost model --------------------------------------------------
REVIEW_COST = 5.0          # cost of manually reviewing one flagged txn (false positive)
# a missed fraud (false negative) costs the full transaction amount

# --- focal loss -----------------------------------------------------------
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.75         # weight of the positive (fraud) class

CATEGORIES = ["grocery", "fuel", "restaurant", "electronics",
              "travel", "online_retail", "entertainment", "utilities"]
