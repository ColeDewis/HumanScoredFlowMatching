import pandas as pd
import numpy as np
import os

# TASK = "CutBanana"
TASK = "CupBallGood"
# TASK = "BottleCubes"

os.makedirs(TASK, exist_ok=True)

score_df = pd.read_csv(f"720 Demos - {TASK}.csv")

value_cols = {"Jasper": [0, 2], "Cole":[1, 2], "Sergey":[0, 1]}

for i, row in score_df.iterrows():
    mean_score = row[2:][value_cols[row["Demonstrator"]]].sum() / 2
    print(row["Demonstration"], mean_score)
    fname = f"{TASK}/{row['Demonstration']}.npy"
    with open(fname, 'wb+') as f:
        np.save(f, np.array(mean_score))
