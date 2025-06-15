import pickle

import numpy as np
from sklearn.metrics import r2_score

with open("/home/results/save_finetune_single_gpu_test.out.pkl", "rb") as f:
    results = pickle.load(f)
import pandas as pd

df = pd.DataFrame(results)
print(df.head())
df.to_excel("/home/results.xlsx", sheet_name="Predictions")
all_preds = []
all_targets = []
for batch in results:
    all_preds.extend(batch["predict"].cpu().numpy().flatten())
    all_targets.extend(batch["target"].cpu().numpy().flatten())
all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
r2 = r2_score(all_targets, all_preds)
print(f"R² Score: {r2:.4f}")
