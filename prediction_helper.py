# prediction_helper.py
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Union

MODEL_FILENAME = "model.joblib"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

def _load_model_optional():
    if not os.path.exists(MODEL_PATH):
        return None
    data = joblib.load(MODEL_PATH)
    if not isinstance(data, dict):
        return None
    required = {"model", "features", "scaler", "cols_to_scale"}
    if not required.issubset(data.keys()):
        return None
    return data

def get_weight_recommendations(
    health_condition: str,
    sets: List[Dict[str, Union[float, int]]],
    target_reps: Dict[str, Union[int, None]],
    body_part: str,
    exercise_name: str,
    equipment_type: str
) -> Dict[str, Union[str, List[Dict[str, Union[int, float, str]]]]]:
    """
    - sets: list of 4 dicts [{"weight": float, "reps": int}, ...]
    - target_reps: {"set_3": int, "set_4": int}
    Returns dict with keys: recommendation, message, suggested_weights, hit_targets (bools), health_warning
    """
    # Basic validation
    if not sets or len(sets) < 4:
        raise ValueError("Provide data for all 4 sets.")

    # Unpack set values
    s3_reps = int(sets[2].get("reps", 0))
    s4_reps = int(sets[3].get("reps", 0))
    t3 = int(target_reps.get("set_3") or 0)
    t4 = int(target_reps.get("set_4") or 0)

    hit3 = (t3 > 0 and s3_reps >= t3)
    hit4 = (t4 > 0 and s4_reps >= t4)

    # Try to use model if available
    model_data = _load_model_optional()
    if model_data:
        try:
            model = model_data["model"]
            features = list(model_data["features"])
            scaler = model_data["scaler"]
            cols_to_scale = list(model_data["cols_to_scale"])

            # Build input row with zeros
            input_df = pd.DataFrame(np.zeros((1, len(features))), columns=features)
            # Set common numeric features if exist
            if "Avg_reps_workout" in input_df.columns:
                input_df.loc[0, "Avg_reps_workout"] = np.mean([s.get("reps", 0) for s in sets])
            if "Set3_Reps" in input_df.columns:
                input_df.loc[0, "Set3_Reps"] = s3_reps
            if "Set4_Reps" in input_df.columns:
                input_df.loc[0, "Set4_Reps"] = s4_reps
            if "Set3_Weight" in input_df.columns:
                input_df.loc[0, "Set3_Weight"] = float(sets[2].get("weight", 0))
            if "Set4_Weight" in input_df.columns:
                input_df.loc[0, "Set4_Weight"] = float(sets[3].get("weight", 0))
            # one-hot columns
            kp = f"Body_Part_{body_part}"
            if kp in input_df.columns:
                input_df.loc[0, kp] = 1
            ke = f"Exercise_Name_{exercise_name}"
            if ke in input_df.columns:
                input_df.loc[0, ke] = 1
            kq = f"Equipment_Type_{equipment_type}"
            if kq in input_df.columns:
                input_df.loc[0, kq] = 1
            kh = f"Health condition_{health_condition}"
            if kh in input_df.columns:
                input_df.loc[0, kh] = 1

            # scale numeric cols if scaler present
            if scaler is not None and cols_to_scale:
                present = [c for c in cols_to_scale if c in input_df.columns]
                if present:
                    input_df[present] = scaler.transform(input_df[present])

            pred = model.predict(input_df)
            # Handle per-set numeric output
            arr = np.asarray(pred)
            if arr.ndim == 2 and arr.shape[1] >= 4:
                row = arr[0][:4]
            elif arr.ndim == 1 and arr.size >= 4:
                row = arr[:4]
            elif arr.size >= 1:
                val = float(arr.ravel()[0])
                # build suggested weights: last set predicted, others maintain
                suggested = []
                for i in range(4):
                    if i == 3:
                        suggested.append({"Set": 4, "Weight (kg)": round(val,1), "Target Reps": t4 or "-", "Action": "Predicted"})
                    else:
                        suggested.append({"Set": i+1, "Weight (kg)": float(sets[i].get("weight",0)), "Target Reps": (t3 if i==2 else "-"), "Action": "Maintain"})
                return {
                    "recommendation": "predicted_last_set",
                    "message": "Model predicted a weight; used for last set",
                    "suggested_weights": suggested,
                    "hit_targets": {"set_3": hit3, "set_4": hit4},
                    "health_warning": health_condition.lower() != "healthy"
                }
            else:
                row = np.zeros(4)

            suggested = []
            for i in range(4):
                suggested.append({"Set": i+1, "Weight (kg)": round(float(row[i]),1), "Target Reps": (t3 if i==2 else (t4 if i==3 else "-")), "Action": "Predicted"})
            return {
                "recommendation": "model_predicted",
                "message": "Model provided suggested weights",
                "suggested_weights": suggested,
                "hit_targets": {"set_3": hit3, "set_4": hit4},
                "health_warning": health_condition.lower() != "healthy"
            }
        except Exception:
            # fallback to rule if anything goes wrong
            pass

    # Rule-based fallback:
    # If both targets (set3 & set4) provided and both hit and health is Healthy => increase.
    # Chest/Back/Legs get larger increment (5kg), others smaller (2.5kg).
    if hit3 and hit4 and health_condition.lower() == "healthy":
        inc = 5.0 if body_part.lower() in ("chest", "back", "legs") else 2.5
        suggested = []
        for i, s in enumerate(sets):
            if i >= 2:
                new_w = round(float(s.get("weight", 0)) + inc, 1)
                action = f"Increase by {inc}kg"
            else:
                new_w = float(s.get("weight", 0))
                action = "Completed"
            suggested.append({"Set": i+1, "Weight (kg)": new_w, "Target Reps": (t3 if i==2 else (t4 if i==3 else "-")), "Action": action})
        return {
            "recommendation": "increase",
            "message": "You hit your targets — suggested increased weights.",
            "suggested_weights": suggested,
            "hit_targets": {"set_3": hit3, "set_4": hit4},
            "health_warning": False
        }
    else:
        # maintain or rest advice if unhealthy
        suggested = []
        for i, s in enumerate(sets):
            suggested.append({"Set": i+1, "Weight (kg)": float(s.get("weight", 0)), "Target Reps": (t3 if i==2 else (t4 if i==3 else "-")), "Action": "Maintain" if health_condition.lower()=="healthy" else "Consider Rest"})
        return {
            "recommendation": "maintain",
            "message": "Maintain current weights (or consider rest if not healthy).",
            "suggested_weights": suggested,
            "hit_targets": {"set_3": hit3, "set_4": hit4},
            "health_warning": health_condition.lower() != "healthy"
        }
