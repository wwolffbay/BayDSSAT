# Domino runs
from domino import Domino
import os
import time

domino = Domino(
    "godcp/BayCropMod",
    api_key=os.environ["DOMINO_USER_API_KEY"],
    host=os.environ["DOMINO_API_HOST"],
)

n_jobs = 36 

for i in range(n_jobs):
    domino_run = domino.runs_start(
        ["DSSATcal.py", str(i), str(n_jobs)], title="Scaling DSSAT calibration, job number: " + str(i)
    )
    time.sleep(5)


