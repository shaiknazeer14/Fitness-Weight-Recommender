import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import hashlib
import pickle
from datetime import datetime
from typing import List, Dict, Any

# ====================== INITIALIZATION ======================
if 'bp_exercises' not in st.session_state:
    st.session_state.bp_exercises = {
        'current_body_part': None,
        'current_exercises': None,
        'selected_exercise': None
    }

# ====================================================================
# 1. CONFIGURATION AND CONSTANTS
# ====================================================================
MODEL_FILE = "model_data.pkl"
USERS_FILE = "users.json"
WORKOUTS_FILE = "workouts.json"

# Column list from your model
all_columns = ['Set3_Hit_Target', 'Set4_Hit_Target', 'month', 'day',
               'Avg_reps_workout', 'progress_rate', 'Health_by_setweight',
               'Body_Part_Back', 'Body_Part_Chest', 'Body_Part_Legs',
               'Body_Part_Shoulders', 'Exercise_Name_Bent-over Row',
               'Exercise_Name_Bicep Curl', 'Exercise_Name_Deadlift',
               'Exercise_Name_Dumbbell Fly', 'Exercise_Name_Front Raise',
               'Exercise_Name_Hammer Curl', 'Exercise_Name_Incline Press',
               'Exercise_Name_Lat Pulldown', 'Exercise_Name_Lateral Raise',
               'Exercise_Name_Leg Press', 'Exercise_Name_Lunges',
               'Exercise_Name_Shoulder Press', 'Exercise_Name_Squat',
               'Exercise_Name_Tricep Extension', 'Equipment_Type_Weight Plate',
               'Health condition_Fever', 'Health condition_Healthy',
               'Health condition_Injury', 'Health condition_Periods',
               'Max_weight_per_reps']


# ====================================================================
# DYNAMIC EXERCISE-BODY PART MAPPING
# ====================================================================
def create_exercise_body_part_mapping(columns):
    """
    Automatically create exercise-body part mapping from model columns
    """
    # Extract exercise names from columns
    exercise_cols = [col for col in columns if col.startswith("Exercise_Name_")]
    exercises = [col.replace("Exercise_Name_", "").strip() for col in exercise_cols]

    # Extract body parts from columns
    body_part_cols = [col for col in columns if col.startswith("Body_Part_")]
    body_parts = [col.replace("Body_Part_", "").strip() for col in body_part_cols]

    # Create mapping based on exercise keywords
    exercise_mapping = {
        "Back": {
            "keywords": ["row", "deadlift", "lat", "pulldown", "pull-up", "chin-up"],
            "exercises": []
        },
        "Chest": {
            "keywords": ["press", "fly", "bench", "push-up", "incline", "decline"],
            "exercises": []
        },
        "Legs": {
            "keywords": ["leg", "squat", "lunges", "deadlift", "calf", "quad", "hamstring"],
            "exercises": []
        },
        "Shoulders": {
            "keywords": ["raise", "shoulder", "press", "upright", "shrug"],
            "exercises": []
        },
        "Arms": {
            "keywords": ["curl", "tricep", "hammer", "bicep", "extension"],
            "exercises": []
        }
    }

    # Map exercises to body parts
    for exercise in exercises:
        exercise_lower = exercise.lower()
        assigned = False

        # Check each body part for keyword matches
        for body_part, info in exercise_mapping.items():
            for keyword in info["keywords"]:
                if keyword in exercise_lower:
                    exercise_mapping[body_part]["exercises"].append(exercise)
                    assigned = True
                    break
            if assigned:
                break

        # Handle special cases or unmapped exercises
        if not assigned:
            # Add to Arms as default for curl-related exercises
            if "curl" in exercise_lower:
                exercise_mapping["Arms"]["exercises"].append(exercise)
            # Add to Legs for deadlift (can work multiple muscle groups)
            elif "deadlift" in exercise_lower:
                if exercise not in exercise_mapping["Back"]["exercises"]:
                    exercise_mapping["Back"]["exercises"].append(exercise)
            else:
                # Create "Other" category if needed
                if "Other" not in exercise_mapping:
                    exercise_mapping["Other"] = {"keywords": [], "exercises": []}
                exercise_mapping["Other"]["exercises"].append(exercise)

    # Clean up - remove empty body parts and return only the exercises
    final_mapping = {}
    for body_part, info in exercise_mapping.items():
        if info["exercises"]:  # Only include body parts that have exercises
            final_mapping[body_part] = sorted(list(set(info["exercises"])))  # Remove duplicates and sort

    return final_mapping


# Generate the dynamic mapping
EXERCISES_BY_BODY_PART = create_exercise_body_part_mapping(all_columns)

# Print the generated mapping for debugging
print("Generated Exercise Mapping:")
for body_part, exercises in EXERCISES_BY_BODY_PART.items():
    print(f"{body_part}: {exercises}")

# Initialize default selection
if st.session_state.bp_exercises['current_body_part'] is None:
    default_bp = list(EXERCISES_BY_BODY_PART.keys())[0]
    st.session_state.bp_exercises = {
        'current_body_part': default_bp,
        'current_exercises': EXERCISES_BY_BODY_PART[default_bp],
        'selected_exercise': EXERCISES_BY_BODY_PART[default_bp][0] if EXERCISES_BY_BODY_PART[default_bp] else None
    }

# ====================================================================
# 2. PAGE CONFIGURATION
# ====================================================================
st.set_page_config(
    page_title="Fitness Weight Recommender",
    page_icon="🏋️",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ====================================================================
# 3. UTILITY FUNCTIONS
# ====================================================================
def load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading {path}: {str(e)}")
        return default


def save_json(path: str, obj: Any) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving to {path}: {str(e)}")


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# ====================================================================
# 4. AUTHENTICATION
# ====================================================================
def register_user(name: str, email: str, password: str) -> bool:
    users = load_json(USERS_FILE, {})
    if email in users:
        return False
    users[email] = {"name": name, "password_hash": hash_password(password)}
    save_json(USERS_FILE, users)
    return True


def verify_user(email: str, password: str) -> bool:
    users = load_json(USERS_FILE, {})
    user_info = users.get(email)
    if not user_info:
        return False
    return user_info.get("password_hash") == hash_password(password)


# ====================================================================
# 5. WORKOUT DATA FUNCTIONS
# ====================================================================
def save_workout(user_email: str, workout_data: Dict[str, Any]) -> None:
    all_workouts = load_json(WORKOUTS_FILE, {})
    key = user_email.strip() or "_anonymous"
    all_workouts.setdefault(key, []).append(workout_data)
    save_json(WORKOUTS_FILE, all_workouts)


def get_user_history(user_email: str) -> List[Dict[str, Any]]:
    all_workouts = load_json(WORKOUTS_FILE, {})
    return all_workouts.get(user_email.strip() or "_anonymous", [])


# ====================================================================
# 6. MODEL LOADING
# ====================================================================
@st.cache_resource
def load_model() -> tuple:
    model = None
    features = []
    scaler = None
    scale_cols = []

    if os.path.exists(MODEL_FILE):
        try:
            # Try pickle first
            with open(MODEL_FILE, "rb") as f:
                model_data = pickle.load(f)
            model = model_data.get("model")
            features = list(model_data.get("features", []))
            scaler = model_data.get("scaler")
            scale_cols = list(model_data.get("cols_to_scale", []))
        except Exception as e:
            # If pickle fails, try dill
            try:
                import dill
                with open(MODEL_FILE, "rb") as f:
                    model_data = dill.load(f)
                model = model_data.get("model")
                features = list(model_data.get("features", []))
                scaler = model_data.get("scaler")
                scale_cols = list(model_data.get("cols_to_scale", []))
                st.info("✅ Model loaded successfully using dill format")
            except Exception as dill_error:
                st.warning(f"Model loading failed with both pickle and dill: {str(dill_error)}")

    return model, features, scaler, scale_cols


model, model_features, model_scaler, cols_to_scale = load_model()


# ====================================================================
# 7. RECOMMENDATION (Rule-based)
# ====================================================================
def rule_based_recommendation(
        sets: List[Dict[str, Any]],
        target_reps: Dict[str, int],
        body_part: str,
        health_condition: str,
        workout_date: Dict[str, int] = None
) -> Dict[str, Any]:
    s3_reps = int(sets[2].get("reps", 0))
    s4_reps = int(sets[3].get("reps", 0))
    t3 = int(target_reps.get("set_3", 0))
    t4 = int(target_reps.get("set_4", 0))
    hit3 = (t3 > 0 and s3_reps >= t3)
    hit4 = (t4 > 0 and s4_reps >= t4)
    is_healthy = health_condition.lower() == "healthy"

    # Use date information for enhanced recommendations (if available)
    date_info = ""
    if workout_date:
        month = workout_date.get("month", 1)
        day = workout_date.get("day", 1)
        hour = workout_date.get("hour", 12)

        # Time-based adjustments (optional logic)
        time_of_day = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
        date_info = f" (Workout time: {time_of_day}, Month: {month}, Day: {day})"

    if hit3 and hit4 and is_healthy:
        inc = 5.0 if body_part.lower() in ["chest", "back", "legs"] else 2.5
        suggested = []
        for i, s in enumerate(sets):
            new_w = round(float(s["weight"]) + inc, 1) if i >= 2 else float(s["weight"])
            action = f"Increase by {inc}kg" if i >= 2 else "Maintain"
            suggested.append({"Set": i + 1, "Weight (kg)": new_w, "Reps": int(s["reps"]),
                              "Target Reps": (t3 if i == 2 else (t4 if i == 3 else "-")), "Action": action})
        return {"recommendation": "increase", "message": f"Increased by {inc}kg for {body_part}{date_info}",
                "suggested_weights": suggested, "hit_targets": {"set_3": hit3, "set_4": hit4},
                "health_warning": False, "workout_date": workout_date}
    else:
        suggested = []
        for i, s in enumerate(sets):
            suggested.append({"Set": i + 1, "Weight (kg)": float(s["weight"]), "Reps": int(s["reps"]),
                              "Target Reps": (t3 if i == 2 else (t4 if i == 3 else "-")),
                              "Action": "Maintain" if is_healthy else "Rest"})
        return {"recommendation": "maintain", "message": f"Maintain current weights{date_info}",
                "suggested_weights": suggested, "hit_targets": {"set_3": hit3, "set_4": hit4},
                "health_warning": not is_healthy, "workout_date": workout_date}


def make_ml_prediction(sets, target_reps, body_part, exercise_name, equipment_type, health_condition, workout_date):
    """
    Use the trained ML model to make weight recommendations
    """
    try:
        # Prepare input features matching your model's expected format
        input_data = prepare_model_input(sets, target_reps, body_part, exercise_name, equipment_type, health_condition,
                                         workout_date)

        # Make prediction using your trained model
        if model_scaler and cols_to_scale:
            # Scale the specified columns
            input_scaled = input_data.copy()
            input_scaled[cols_to_scale] = model_scaler.transform(input_data[cols_to_scale])
            prediction = model.predict(input_scaled)
        else:
            prediction = model.predict(input_data)

        # Convert prediction to recommendation format
        return format_ml_recommendation(prediction, sets, target_reps, body_part, exercise_name, workout_date)

    except Exception as e:
        st.error(f"ML prediction failed: {str(e)}")
        raise e


def prepare_model_input(sets, target_reps, body_part, exercise_name, equipment_type, health_condition, workout_date):
    """
    Prepare input features for the ML model based on all_columns
    """
    # Initialize all features with 0
    input_features = {col: 0 for col in all_columns}

    # Set target hit features
    s3_reps = int(sets[2].get("reps", 0))
    s4_reps = int(sets[3].get("reps", 0))
    t3 = int(target_reps.get("set_3", 0))
    t4 = int(target_reps.get("set_4", 0))

    input_features['Set3_Hit_Target'] = 1 if (t3 > 0 and s3_reps >= t3) else 0
    input_features['Set4_Hit_Target'] = 1 if (t4 > 0 and s4_reps >= t4) else 0

    # Set date features
    if workout_date:
        input_features['month'] = workout_date.get('month', 1)
        input_features['day'] = workout_date.get('day', 1)

    # Calculate workout metrics
    total_reps = sum(int(s.get("reps", 0)) for s in sets)
    input_features['Avg_reps_workout'] = total_reps / 4 if total_reps > 0 else 0

    # Calculate max weight and progress rate
    weights = [float(s.get("weight", 0)) for s in sets]
    max_weight = max(weights) if weights else 0
    input_features['Max_weight_per_reps'] = max_weight

    # Simple progress rate calculation (can be enhanced based on history)
    input_features['progress_rate'] = 0.1  # Default, could be calculated from user history

    # Health condition encoding
    health_col = f'Health condition_{health_condition.title()}'
    if health_col in input_features:
        input_features[health_col] = 1
    elif 'Health condition_Healthy' in input_features:
        input_features['Health condition_Healthy'] = 1  # Default

    # Body part encoding
    body_part_col = f'Body_Part_{body_part}'
    if body_part_col in input_features:
        input_features[body_part_col] = 1

    # Exercise encoding
    exercise_col = f'Exercise_Name_{exercise_name}'
    if exercise_col in input_features:
        input_features[exercise_col] = 1

    # Equipment encoding (if exists in model)
    equipment_col = f'Equipment_Type_{equipment_type}'
    if equipment_col in input_features:
        input_features[equipment_col] = 1

    # Calculate health by set weight (custom metric)
    avg_weight = sum(weights) / len(weights) if weights else 0
    health_multiplier = 1.0 if health_condition.lower() == "healthy" else 0.8
    input_features['Health_by_setweight'] = avg_weight * health_multiplier

    # Convert to DataFrame for model prediction
    import pandas as pd
    return pd.DataFrame([input_features])


def format_ml_recommendation(prediction, sets, target_reps, body_part, exercise_name, workout_date):
    """
    Format ML model prediction into recommendation format
    """
    # Your model's prediction interpretation logic here
    # This depends on what your model was trained to predict
    predicted_value = prediction[0] if hasattr(prediction, '__getitem__') else prediction

    # Interpret the prediction (adjust based on your model's output)
    if predicted_value > 0.6:  # Threshold for weight increase
        inc = 5.0 if body_part.lower() in ["chest", "back", "legs"] else 2.5
        recommendation_type = "increase"
        message = f"🤖 ML Model recommends: Increase by {inc}kg for {body_part}"
    else:
        inc = 0
        recommendation_type = "maintain"
        message = f"🤖 ML Model recommends: Maintain current weights for {body_part}"

    # Create suggested weights based on prediction
    suggested = []
    for i, s in enumerate(sets):
        if recommendation_type == "increase" and i >= 2:
            new_w = round(float(s["weight"]) + inc, 1)
            action = f"Increase by {inc}kg (ML)"
        else:
            new_w = float(s["weight"])
            action = "Maintain (ML)"

        t3 = int(target_reps.get("set_3", 0))
        t4 = int(target_reps.get("set_4", 0))
        target = t3 if i == 2 else (t4 if i == 3 else "-")

        suggested.append({
            "Set": i + 1,
            "Weight (kg)": new_w,
            "Reps": int(s["reps"]),
            "Target Reps": target,
            "Action": action
        })

    date_info = ""
    if workout_date:
        hour = workout_date.get("hour", 12)
        time_of_day = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
        date_info = f" (Prediction confidence: {predicted_value:.2f}, Time: {time_of_day})"

    return {
        "recommendation": recommendation_type,
        "message": message + date_info,
        "suggested_weights": suggested,
        "hit_targets": {
            "set_3": int(sets[2].get("reps", 0)) >= int(target_reps.get("set_3", 0)),
            "set_4": int(sets[3].get("reps", 0)) >= int(target_reps.get("set_4", 0))
        },
        "health_warning": False,
        "workout_date": workout_date,
        "ml_prediction": True,
        "prediction_value": predicted_value
    }


def get_recommendation(sets, target_reps, body_part, exercise_name, equipment_type, health_condition,
                       workout_date=None):
    if exercise_name not in EXERCISES_BY_BODY_PART.get(body_part, []):
        st.error(f"Invalid exercise '{exercise_name}' for selected body part '{body_part}'!")
        st.info(f"Available exercises for {body_part}: {', '.join(EXERCISES_BY_BODY_PART.get(body_part, []))}")
        return rule_based_recommendation(sets, target_reps, body_part, health_condition, workout_date)

    # Use ML model if available
    if model is not None:
        try:
            st.info("🤖 Using your trained ML model for predictions!")
            return make_ml_prediction(sets, target_reps, body_part, exercise_name, equipment_type, health_condition,
                                      workout_date)
        except Exception as e:
            st.warning(f"ML model failed, falling back to rule-based: {str(e)}")
            return rule_based_recommendation(sets, target_reps, body_part, health_condition, workout_date)
    else:
        st.info("📋 Using rule-based recommendations (model not loaded)")
        return rule_based_recommendation(sets, target_reps, body_part, health_condition, workout_date)


# ====================================================================
# 8. STREAMLIT UI
# ====================================================================
def show_auth_page():
    st.title("🏋️ Fitness Weight Recommender")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if verify_user(email, password):
                    st.session_state.logged_in = True
                    st.session_state.user_email = email
                    st.rerun()
                else:
                    st.error("Invalid credentials")

    with tab2:
        with st.form("register_form"):
            name = st.text_input("Full Name")
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            if st.form_submit_button("Register"):
                if new_password != confirm_password:
                    st.error("Passwords don't match")
                elif register_user(name, new_email, new_password):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Email already registered")


def show_recommendation_page():
    st.header("💪 Get Weight Recommendations")

    # Debug: Show the current mapping
    if st.checkbox("Show Exercise Mapping (Debug)"):
        st.write("Current Exercise-Body Part Mapping:")
        for bp, exs in EXERCISES_BY_BODY_PART.items():
            st.write(f"**{bp}**: {', '.join(exs)}")

    # Handle body part selection OUTSIDE the form to allow immediate updates
    selected_body_part = st.selectbox(
        "Body Part",
        list(EXERCISES_BY_BODY_PART.keys()),
        key='body_part_select_main'
    )

    # Get exercises for the selected body part
    available_exercises = EXERCISES_BY_BODY_PART.get(selected_body_part, [])

    if not available_exercises:
        st.error(f"No exercises found for {selected_body_part}")
        st.stop()

    # Show the selected body part's exercises
    selected_exercise = st.selectbox(
        "Exercise",
        available_exercises,
        key='exercise_select_main'
    )

    with st.form("workout_form"):
        col1, col2 = st.columns(2)
        with col1:
            # Display the selections (read-only in form)
            st.info(f"**Selected Body Part:** {selected_body_part}")
            st.info(f"**Selected Exercise:** {selected_exercise}")

        with col2:
            equipment = st.selectbox("Equipment", ["Dumbbell", "Barbell", "Machine", "Bodyweight"])
            health = st.selectbox("Health Status", ["Healthy", "Injured", "Recovering", "Other"])

        with col2:
            equipment = st.selectbox("Equipment", ["Dumbbell", "Barbell", "Machine", "Bodyweight"],
                                     key="equipment_select")
            health = st.selectbox("Health Status", ["Healthy", "Injured", "Recovering", "Other"], key="health_select")

        st.subheader("Workout Sets")
        sets = []
        for i in range(4):
            cols = st.columns(2)
            with cols[0]:
                weight = st.number_input(f"Weight (kg) - Set {i + 1}", min_value=0.0, step=0.5, key=f"weight_{i}")
            with cols[1]:
                reps = st.number_input(f"Reps - Set {i + 1}", min_value=0, step=1, key=f"reps_{i}")
            sets.append({"weight": weight, "reps": reps})

        st.subheader("Target Reps")
        target_reps = {
            "set_3": st.number_input("Target Reps - Set 3", min_value=0, step=1, value=8, key="target_reps_3"),
            "set_4": st.number_input("Target Reps - Set 4", min_value=0, step=1, value=6, key="target_reps_4")
        }

        # Add the submit button
        submit_button = st.form_submit_button("Get Recommendations", type="primary")

        if submit_button:
            with st.spinner("Analyzing..."):
                # Get current date/time information
                current_time = datetime.now()
                workout_date = {
                    "month": current_time.month,
                    "day": current_time.day,
                    "year": current_time.year,
                    "hour": current_time.hour,
                    "minute": current_time.minute
                }

                recommendation = get_recommendation(
                    sets=sets,
                    target_reps=target_reps,
                    body_part=selected_body_part,
                    exercise_name=selected_exercise,
                    equipment_type=equipment,
                    health_condition=health,
                    workout_date=workout_date  # Pass date info to recommendation
                )
                st.success(recommendation["message"])
                st.dataframe(pd.DataFrame(recommendation["suggested_weights"]), hide_index=True)
                if recommendation["health_warning"]:
                    st.warning("Consider consulting a doctor before increasing intensity")

                workout_data = {
                    "date": current_time.strftime("%Y-%m-%d"),
                    "time": current_time.strftime("%H:%M:%S"),
                    "datetime_info": workout_date,
                    "body_part": selected_body_part,
                    "exercise": selected_exercise,
                    "equipment": equipment,
                    "health_condition": health,
                    "sets": sets,
                    "target_reps": target_reps,
                    "recommendation": recommendation
                }
                save_workout(st.session_state.user_email, workout_data)
                st.toast("Workout saved!")


def show_history_page():
    st.header("📋 Workout History")
    history = get_user_history(st.session_state.user_email)
    if not history:
        st.info("No workout history yet")
        return
    history.reverse()
    for workout in history:
        with st.expander(
                f"{workout['date']} {workout.get('time', '')} - {workout['body_part']} - {workout['exercise']}"):
            st.write(f"**Equipment:** {workout['equipment']}")
            st.write(f"**Health Status:** {workout['health_condition']}")

            # Show date/time information if available
            if 'datetime_info' in workout:
                date_info = workout['datetime_info']
                st.write(
                    f"**Workout Details:** Month {date_info['month']}, Day {date_info['day']}, {date_info['year']} at {date_info['hour']:02d}:{date_info['minute']:02d}")

            st.subheader("Sets Performance")
            st.dataframe(pd.DataFrame(workout['sets']), hide_index=True)
            st.subheader("Recommendation")
            st.write(workout['recommendation']['message'])
            st.dataframe(pd.DataFrame(workout['recommendation']['suggested_weights']), hide_index=True)


# ====================================================================
# 9. APP ROUTING
# ====================================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    show_auth_page()
    st.stop()

st.sidebar.title(f"Welcome, {st.session_state.user_email}")
page = st.sidebar.radio("Navigation", ["Recommend", "History"])

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    st.rerun()

if page == "Recommend":
    show_recommendation_page()
else:
    show_history_page()