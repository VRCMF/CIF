from pathlib import Path

import yaml

with open(Path(__file__).resolve().parent / "rome" / "globals.yml", "r") as stream:
    data = yaml.safe_load(stream)

(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR,) = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]

# Set True at runtime by intervention_fused.py --zero_shot_tuning (identity MLP patch: mlp_dires = mlp_val).
ZERO_SHOT_TUNING = False
