import streamlit as st
import os
import cv2
import time
import pandas as pd
import plotly.graph_objects as go
import requests
import json

# ---------------- Gemini API Settings & Helper Functions ----------------
#GEMINI_API_KEY = "AIzaSyCAxTziyQpvsKEjEBOmpiHyMTLd3wVITLc"  # Replace with your own key
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
BUFFER_WINDOW = 2  # seconds

def call_gemini_api(prompt):
    """
    Calls the Google Gemini API with a given prompt and returns the response.
    """
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(GEMINI_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
    else:
        return f"API call failed: {response.status_code}"

def compute_deflection_intervals(times, avg_values, threshold=60, buffer_sec=2.0):
    """
    Given arrays of times and average similarity values, return a list of intervals
    (start, end) where avg_values is below the threshold. Intervals that are within
    buffer_sec seconds of each other are merged.
    """
    intervals = []
    start = None
    for t, val in zip(times, avg_values):
        if val < threshold and start is None:
            start = t
        elif val >= threshold and start is not None:
            intervals.append((start, t))
            start = None
    if start is not None:
        intervals.append((start, times[-1]))
    
    merged = []
    for intv in intervals:
        if not merged:
            merged.append(intv)
        else:
            last = merged[-1]
            if intv[0] - last[1] <= buffer_sec:
                merged[-1] = (last[0], intv[1])
            else:
                merged.append(intv)
    return merged

def extract_deflections(df):
    """
    Extracts deflections only when the 'average_similarity' column is below 60%.
    Uses a threshold of a >5% drop per keypoint and groups deflections within BUFFER_WINDOW seconds.
    """
    deflection_threshold = 5  # Drop of more than 5% from the previous value
    low_similarity_threshold = 60  # Only consider rows where average similarity is below 60%
    columns_to_check = ["Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist"]

    if "average_similarity" not in df.columns:
        st.error("CSV file must contain an 'average_similarity' column.")
        return []

    df["valid_deflection"] = df["average_similarity"] < low_similarity_threshold

    for col in columns_to_check:
        df[f"{col}_deflection"] = df[col].diff().abs() > deflection_threshold
        df[f"{col}_low_similarity"] = df[col] < low_similarity_threshold

    last_deflection_time = -BUFFER_WINDOW
    grouped_deflections = []

    for index, row in df.iterrows():
        if not row["valid_deflection"]:
            continue

        deflection_detected = False
        deflection_info = []
        for col in columns_to_check:
            if row[f"{col}_deflection"] or row[f"{col}_low_similarity"]:
                deflection_detected = True
                deflection_info.append(f"{col} at {row[col]:.2f}%")
        if deflection_detected:
            current_time = row["timestamp"]
            if current_time - last_deflection_time > BUFFER_WINDOW:
                grouped_deflections.append(f"Time {current_time:.2f}s: " + ", ".join(deflection_info))
                last_deflection_time = current_time

    return grouped_deflections if grouped_deflections else ["No major deflections detected where avg < 60%."]

def generate_user_feedback(deflection_list):
    """
    Generates short, one-sentence posture correction feedback for each deflection where avg similarity < 60%.
    """
    user_feedback = []
    for deflection in deflection_list:
        prompt = f"""You are a rehabilitation therapist for breast cancer recovery patients.
Your goal is to provide one very short, simple posture correction tip based on the following detected deflection:

Detected Deflection: {deflection}

Provide a single-sentence recommendation, such as:
- 'Move your arms smoothly to improve circulation.'  
- 'Relax your shoulders and focus on deep breathing.'  
- 'Keep your arms at a comfortable height to avoid strain.'
"""
        feedback = call_gemini_api(prompt)
        user_feedback.append(f"{deflection} → {feedback}")
    return "\n".join(user_feedback)

# ---------------- Main App Code ----------------

DATABASE_ROOT = "Similarity_DB"
st.set_page_config(layout="wide")
st.title("Pose Comparison with Synchronized Playback")

# ------------------ 1) Sidebar Selections ------------------
tolf_folders = [d for d in os.listdir(DATABASE_ROOT) if os.path.isdir(os.path.join(DATABASE_ROOT, d))]
tolf_folders.sort()
selected_tolf = st.sidebar.selectbox("Select Patient Folder", tolf_folders)
tolf_path = os.path.join(DATABASE_ROOT, selected_tolf)

exercise_subfolders = [d for d in os.listdir(tolf_path) if os.path.isdir(os.path.join(tolf_path, d))]
exercise_subfolders.sort()
selected_exercise = st.sidebar.selectbox("Select Exercise", exercise_subfolders)
exercise_path = os.path.join(tolf_path, selected_exercise)

# Reset session state if patient or exercise changes
if ("prev_patient" not in st.session_state or st.session_state["prev_patient"] != selected_tolf or
    "prev_exercise" not in st.session_state or st.session_state["prev_exercise"] != selected_exercise):
    st.session_state["analysis_triggered"] = False
    st.session_state["playback_finished"] = False
    st.session_state["prev_patient"] = selected_tolf
    st.session_state["prev_exercise"] = selected_exercise

# Paths to videos and CSV
ref_video_path = os.path.join(exercise_path, "processed_ref.mp4")
patient_video_path = os.path.join(exercise_path, "processed_patient.mp4")
similarity_csv_path = os.path.join(exercise_path, "similarity_data.csv")

# ------------------ 2) Analyze Button for Synchronized Playback ------------------
if "analysis_triggered" not in st.session_state:
    st.session_state["analysis_triggered"] = False

def on_analyze_click():
    st.session_state["analysis_triggered"] = True

analyze_button = st.button("Analyze", on_click=on_analyze_click)

# Run playback only if analysis is triggered and playback is not yet finished.
if st.session_state.get("analysis_triggered", False) and not st.session_state.get("playback_finished", False):
    # --- Check files ---
    if not os.path.exists(ref_video_path):
        st.error(f"Reference video not found: {ref_video_path}")
        st.stop()
    if not os.path.exists(patient_video_path):
        st.error(f"Patient video not found: {patient_video_path}")
        st.stop()
    if not os.path.exists(similarity_csv_path):
        st.error(f"Similarity data not found: {similarity_csv_path}")
        st.stop()

    # --- Load CSV ---
    df = pd.read_csv(similarity_csv_path)
    if "timestamp" not in df.columns:
        st.error("CSV must contain a 'timestamp' column. Please adjust code or CSV.")
        st.stop()
    joint_cols = ["Left Shoulder", "Right Shoulder", "Left Elbow",
                  "Right Elbow", "Left Wrist", "Right Wrist"]
    missing_joints = [c for c in joint_cols if c not in df.columns]
    if missing_joints:
        st.error(f"Missing joint columns in CSV: {missing_joints}")
        st.stop()
    if "average_similarity" not in df.columns:
        st.error("CSV must contain an 'average_similarity' column. Please adjust code or CSV.")
        st.stop()

    times = df["timestamp"].values
    joint_data = {col: df[col].values for col in joint_cols}
    avg_data = df["average_similarity"].values

    # --- Open Videos ---
    cap_ref = cv2.VideoCapture(ref_video_path)
    cap_user = cv2.VideoCapture(patient_video_path)
    fps_ref = cap_ref.get(cv2.CAP_PROP_FPS)
    fps_user = cap_user.get(cv2.CAP_PROP_FPS)
    playback_fps = min(fps_ref, fps_user) if fps_ref > 0 and fps_user > 0 else 30.0

    # Placeholders for side-by-side videos
    col1, col2 = st.columns(2)
    video_placeholder_ref = col1.empty()
    video_placeholder_user = col2.empty()

    # Placeholders for charts (stacked vertically)
    chart_placeholder_keypoints = st.empty()
    chart_placeholder_average = st.empty()

    # --- Create Plotly Figures ---
    # Figure for 6 keypoints
    fig_keypoints = go.Figure()
    for jcol in joint_cols:
        fig_keypoints.add_trace(go.Scatter(x=[], y=[], mode="lines", name=jcol))
    fig_keypoints.update_layout(
        title="Keypoint Similarities",
        xaxis_title="Time (s)",
        yaxis_title="Similarity (%)",
        xaxis_range=[0, times[-1]],
        yaxis_range=[0, 100]
    )

    # Figure for average similarity
    fig_average = go.Figure()
    fig_average.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Average Similarity"))
    fig_average.update_layout(
        title="Average Similarity",
        xaxis_title="Time (s)",
        yaxis_title="Similarity (%)",
        xaxis_range=[0, times[-1]],
        yaxis_range=[0, 100]
    )

    frame_idx = 0
    while True:
        ret_ref, frame_ref = cap_ref.read()
        ret_user, frame_user = cap_user.read()
        if not ret_ref or not ret_user:
            break  # End playback when either video ends

        frame_ref_rgb = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2RGB)
        frame_user_rgb = cv2.cvtColor(frame_user, cv2.COLOR_BGR2RGB)
        current_time_sec = frame_idx / playback_fps

        valid_indices = [i for i, t in enumerate(times) if t <= current_time_sec]
        last_idx = valid_indices[-1] if valid_indices else 0

        for i, jcol in enumerate(joint_cols):
            fig_keypoints.data[i].x = times[:last_idx+1]
            fig_keypoints.data[i].y = joint_data[jcol][:last_idx+1]
        fig_average.data[0].x = times[:last_idx+1]
        fig_average.data[0].y = avg_data[:last_idx+1]

        # --- Add Red Gradient for Deflections in the Average Graph ---
        fig_average.layout.shapes = []  # Clear previous shapes
        current_times = times[:last_idx+1]
        current_avg = avg_data[:last_idx+1]
        deflection_intervals = compute_deflection_intervals(current_times, current_avg, threshold=60, buffer_sec=2.0)
        for interval in deflection_intervals:
            start, end = interval
            fig_average.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.3, line_width=0, layer="below")

        video_placeholder_ref.image(frame_ref_rgb, caption=f"Reference (t={current_time_sec:.2f}s)", use_column_width=True)
        video_placeholder_user.image(frame_user_rgb, caption=f"Patient (t={current_time_sec:.2f}s)", use_column_width=True)
        chart_placeholder_keypoints.plotly_chart(fig_keypoints, use_container_width=True)
        chart_placeholder_average.plotly_chart(fig_average, use_container_width=True)

        time.sleep(1.0 / playback_fps)
        frame_idx += 1

    cap_ref.release()
    cap_user.release()
    st.success("Playback finished!")
    st.session_state["playback_finished"] = True

# ------------------ 3) Automatically Generate LLM Feedback After Playback ------------------
if st.session_state.get("playback_finished", False):
    st.markdown("---")
    st.header("LLM Feedback")
    # Load the CSV again and extract deflections
    if not os.path.exists(similarity_csv_path):
        st.error("Similarity CSV not found.")
    else:
        df_llm = pd.read_csv(similarity_csv_path)
        deflection_list = extract_deflections(df_llm)

        st.subheader("💡 General Posture Suggestion")

        if len(deflection_list) > 0 and deflection_list[0] != "No major deflections detected where avg < 60%.":
            # Generate a friendly posture improvement suggestion with encouragement
            friendly_prompt = f"""Give a **100 word, friendly tip** for improving posture based on these detected deflections and include emojis:
{', '.join(deflection_list)}

The response should:
- Be **brief and supportive**, like a coach giving quick advice.
- Include **one small improvement tip**.
- End with an encouraging statement that reminds the user they're not alone.

Example: "You're doing great! Try keeping your shoulders relaxed for smoother movement. We're always here to support you!"
"""
            general_suggestion = call_gemini_api(friendly_prompt)
            st.info(f"**📋 Try This:** {general_suggestion}")

            # Allow users to expand for specific time-stamped feedback
            with st.expander("🔍 View Specific Feedback for Each Time:"):
                for deflection in deflection_list:
                    feedback = call_gemini_api(f"Give a **one-sentence posture tip** for this deflection: {deflection}")
                    timestamp = deflection.split(":")[0].strip()
                    affected_joints = deflection.split(":")[1].strip().split(" → ")[0]  # Extract affected joints

                    with st.container():
                        st.markdown(f"**⏳ {timestamp}**")
                        st.markdown(f"**🦴 Affected Joints:** {affected_joints}")
                        st.success(f"**✅ Suggestion:** {feedback}")
                        st.markdown("---")
        else:
            # If no deflections below 60%, generate a positive general suggestion
            encouragement_prompt = f"""Give a **quick, supportive tip** for a user with no major posture issues.
The response should include:
- A **small improvement tip**.
- An **encouraging statement** to remind them that we’re here for them.

Example: "You're moving well! Keep your movements steady for even better control. We're always here for you!"
"""
            encouragement_message = call_gemini_api(encouragement_prompt)
            st.success(f"**🌟 Keep It Up!** {encouragement_message}")
