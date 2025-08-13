# dummy_model_creator.py
import dill
import numpy as np
import os

MODEL_FILENAME = "model_data.pkl"

class DummyModel:
    """
    Simple model: returns 4 per-set suggested weights based on Avg_reps_workout (or default).
    Output shape: (n_samples, 4)
    """
    def predict(self, X):
        # X may be a DataFrame-like; try to read Avg_reps_workout if present
        try:
            # If X is a pandas DataFrame-like with column access via X.get
            if hasattr(X, "get"):
                avg_list = X.get("Avg_reps_workout", [8]*len(X)) if hasattr(X, "get") else [8] * len(X)
                avgv = np.array(avg_list).astype(float)
            else:
                # fallback: len(X)
                n = X.shape[0]
                avgv = np.full(n, 8.0)
        except Exception:
            n = X.shape[0] if hasattr(X, "shape") else 1
            avgv = np.full(n, 8.0)

        result = []
        for i in range(len(avgv)):
            base = 10.0 + (avgv[i] - 8.0) * 0.5
            # returns [set1, set2, set3, set4]
            result.append(np.array([base, base + 2.0, base + 4.0, base + 6.0], dtype=float))
        return np.array(result)

# Minimal features to match app expectations (app will only set ones that exist)
features = [
    "Set3_Hit_Target", "Set4_Hit_Target", "month", "day",
    "Avg_reps_workout", "progress_rate", "Health_by_setweight",
    # one-hot placeholders
    "Body_Part_Back", "Body_Part_Chest", "Body_Part_Legs", "Body_Part_Shoulders", "Body_Part_Arms",
    "Exercise_Name_Bent-over Row", "Exercise_Name_Bicep Curl", "Exercise_Name_Deadlift",
    "Exercise_Name_Dumbbell Fly", "Exercise_Name_Front Raise", "Exercise_Name_Hammer Curl",
    "Exercise_Name_Incline Press", "Exercise_Name_Lat Pulldown", "Exercise_Name_Lateral Raise",
    "Exercise_Name_Leg Press", "Exercise_Name_Lunges", "Exercise_Name_Shoulder Press", "Exercise_Name_Squat",
    "Exercise_Name_Tricep Extension", "Equipment_Type_Weight Plate", "Equipment_Type_Dumbbell",
    "Health condition_Fever", "Health condition_Healthy", "Health condition_Injury", "Health condition_Periods",
    "Max_weight_per_reps"
]

model_data = {
    "model": DummyModel(),
    "features": features,
    "scaler": None,
    "cols_to_scale": []
}

with open(MODEL_FILENAME, "wb") as f:
    dill.dump(model_data, f)

print(f"✅ Dummy model saved as {MODEL_FILENAME} in {os.getcwd()}")
