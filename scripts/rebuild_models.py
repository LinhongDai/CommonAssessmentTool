# scripts/rebuild_models.py

import os
import pickle
import numpy as np
from sklearn.linear_model import Ridge

MODEL_DIR = "app/clients/service/MLmodels"
os.makedirs(MODEL_DIR, exist_ok=True)

model = Ridge()
X = np.random.rand(50, 4)
y = np.random.rand(50)
model.fit(X, y)

with open(os.path.join(MODEL_DIR, "ridge_model.pkl"), "wb") as f:
    pickle.dump(model, f)

print("✅ Model re-pickled successfully.")
