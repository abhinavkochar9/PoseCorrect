import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
import requests
import json
import time
import base64

# ---------------- Gemini API Settings & Helper Functions ----------------
GEMINI_API_KEY = "AIzaSyCAxTziyQpvsKEjEBOmpiHyMTLd3wVITLc"  # Replace with your own key
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
    Returns (start, end) intervals where avg_values < threshold, merging intervals that are within buffer_sec seconds.
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
    Detects deflections only when 'average_similarity' < 60%, grouping close occurrences within BUFFER_WINDOW seconds.
    Also checks for >5% drops on each relevant joint.
    """
    deflection_threshold = 5
    low_similarity_threshold = 60
    columns_to_check = ["Left Shoulder", "Right Shoulder", 
                        "Left Elbow", "Right Elbow", 
                        "Left Wrist", "Right Wrist"]

    if "average_similarity" not in df.columns:
        st.error("CSV file must contain an 'average_similarity' column.")
        return []

    df["valid_deflection"] = df["average_similarity"] < low_similarity_threshold

    for col in columns_to_check:
        df[f"{col}_deflection"] = df[col].diff().abs() > deflection_threshold
        df[f"{col}_low_similarity"] = df[col] < low_similarity_threshold

    grouped_deflections = []
    last_deflection_time = -BUFFER_WINDOW

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

def autoplay_video_base64(video_path, width="100%", height="auto", muted=True):
    """
    Returns HTML that embeds the given MP4 video with base64 data in autoplay mode (muted by default)
    and with improved styling (border and rounded corners).
    """
    with open(video_path, "rb") as f:
        video_bytes = f.read()
        encoded = base64.b64encode(video_bytes).decode()

    mute_str = "muted" if muted else ""
    # Add a border and rounded corners for a nicer look
    style = 'style="border:2px solid #ccc; border-radius:10px;"'
    html_code = f"""
    <video {style} width="{width}" height="{height}" autoplay controls {mute_str}>
        <source src="data:video/mp4;base64,{encoded}" type="video/mp4" />
        Your browser does not support the video tag.
    </video>
    """
    return html_code

# ---------------- Main App Code ----------------

DATABASE_ROOT = "Similarity_DB"
st.set_page_config(layout="wide")
st.title("Pose Comparison")

# 1) Sidebar for Patient & Exercise
tolf_folders = [d for d in os.listdir(DATABASE_ROOT) if os.path.isdir(os.path.join(DATABASE_ROOT, d))]
tolf_folders.sort()
selected_tolf = st.sidebar.selectbox("Select Patient Folder", tolf_folders)
tolf_path = os.path.join(DATABASE_ROOT, selected_tolf)

exercise_subfolders = [d for d in os.listdir(tolf_path) if os.path.isdir(os.path.join(tolf_path, d))]
exercise_subfolders.sort()
selected_exercise = st.sidebar.selectbox("Select Exercise", exercise_subfolders)
exercise_path = os.path.join(tolf_path, selected_exercise)

# Reset session state if new selection
if ("prev_patient" not in st.session_state or st.session_state["prev_patient"] != selected_tolf
    or "prev_exercise" not in st.session_state or st.session_state["prev_exercise"] != selected_exercise):
    st.session_state["show_chart_sim"] = False
    st.session_state["chart_finished"] = False
    st.session_state["prev_patient"] = selected_tolf
    st.session_state["prev_exercise"] = selected_exercise

# Paths to videos & CSV
ref_video_path = os.path.join(exercise_path, "processed_ref.mp4")
patient_video_path = os.path.join(exercise_path, "processed_patient.mp4")
similarity_csv_path = os.path.join(exercise_path, "similarity_data.csv")

# 2) Button: Analyze
def on_analyze_click():
    # Check existence
    if not os.path.exists(ref_video_path):
        st.error(f"Reference video not found: {ref_video_path}")
        st.stop()
    if not os.path.exists(patient_video_path):
        st.error(f"Patient video not found: {patient_video_path}")
        st.stop()
    if not os.path.exists(similarity_csv_path):
        st.error(f"CSV file not found: {similarity_csv_path}")
        st.stop()

    st.session_state["show_chart_sim"] = True
    st.session_state["chart_finished"] = False

analyze_button = st.button("Analyze", on_click=on_analyze_click)

# 3) Show Videos (autoplay) & Simulate Chart if Analyze clicked
if st.session_state.get("show_chart_sim", False):

    colv1, colv2 = st.columns(2)
    with colv1:
        html_vid1 = autoplay_video_base64(ref_video_path, width="100%", muted=True)
        st.markdown(html_vid1, unsafe_allow_html=True)
    with colv2:
        html_vid2 = autoplay_video_base64(patient_video_path, width="100%", muted=True)
        st.markdown(html_vid2, unsafe_allow_html=True)

    # Load CSV for incremental chart updates
    df = pd.read_csv(similarity_csv_path)
    if "timestamp" not in df.columns or "average_similarity" not in df.columns:
        st.error("CSV must contain 'timestamp' & 'average_similarity' columns.")
        st.stop()

    joint_cols = ["Left Shoulder", "Right Shoulder", 
                  "Left Elbow", "Right Elbow", 
                  "Left Wrist", "Right Wrist"]
    for jc in joint_cols:
        if jc not in df.columns:
            st.error(f"Missing joint column in CSV: {jc}")
            st.stop()

    times = df["timestamp"].values
    avg_vals = df["average_similarity"].values
    joint_data = {col: df[col].values for col in joint_cols}

    # Set up Plotly figures
    fig_keypoints = go.Figure()
    for jcol in joint_cols:
        fig_keypoints.add_trace(go.Scatter(x=[], y=[], mode="lines", name=jcol))
    fig_keypoints.update_layout(
        title="Keypoint Similarities (Simulated)",
        xaxis_title="Time (s)",
        yaxis_title="Similarity (%)",
        xaxis_range=[0, times[-1]],
        yaxis_range=[0, 100]
    )

    fig_average = go.Figure()
    fig_average.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Average Similarity"))
    fig_average.update_layout(
        title="Average Similarity (Simulated)",
        xaxis_title="Time (s)",
        yaxis_title="Similarity (%)",
        xaxis_range=[0, times[-1]],
        yaxis_range=[0, 100]
    )

    chart_ph_keypoints = st.empty()
    chart_ph_average = st.empty()

    # Loop through each row of the CSV, "sleeping" to simulate real-time updates
    for i in range(1, len(times)):
        cur_timestamp = times[i]
        prev_timestamp = times[i-1] if i > 0 else 0
        dt = cur_timestamp - prev_timestamp  # seconds to wait

        # Update figures up to index i
        for idx, jcol in enumerate(joint_cols):
            fig_keypoints.data[idx].x = times[:i+1]
            fig_keypoints.data[idx].y = joint_data[jcol][:i+1]

        fig_average.data[0].x = times[:i+1]
        fig_average.data[0].y = avg_vals[:i+1]

        # Highlight deflections below 60 in the average chart
        fig_average.layout.shapes = []
        sub_times = times[:i+1]
        sub_avg = avg_vals[:i+1]
        deflects = compute_deflection_intervals(sub_times, sub_avg, threshold=60, buffer_sec=2.0)
        for (start_int, end_int) in deflects:
            fig_average.add_vrect(
                x0=start_int, x1=end_int,
                fillcolor="red", opacity=0.3, line_width=0, layer="below"
            )

        # Render updated figures
        chart_ph_keypoints.plotly_chart(fig_keypoints, use_container_width=True)
        chart_ph_average.plotly_chart(fig_average, use_container_width=True)

        # Sleep for the time difference
        time.sleep(dt)
    else:
        st.success("Chart simulation finished!")
        st.session_state["chart_finished"] = True

# 4) LLM Feedback if chart simulation finished
if st.session_state.get("chart_finished", False):
    st.markdown("---")
    st.header("LLM Feedback")

    df_llm = pd.read_csv(similarity_csv_path)
    deflection_list = extract_deflections(df_llm)

    st.subheader("💡 General Posture Suggestion")
    if deflection_list and deflection_list[0] != "No major deflections detected where avg < 60%.":
        prompt = f"""Give a **100 word, friendly tip** for improving posture based on these detected deflections and include emojis:
{', '.join(deflection_list)}

The response should:
- Be **brief and supportive**,
- Include **one small improvement tip**,
- End with an encouraging statement.
"""
        general_suggestion = call_gemini_api(prompt)
        st.info(f"**📋 Try This:** {general_suggestion}")

        with st.expander("🔍 View Specific Feedback for Each Time:"):
            for deflection in deflection_list:
                feedback = call_gemini_api(f"One-sentence tip for this deflection: {deflection}")
                st.success(f"{deflection} → {feedback}")
    else:
        no_deflect_prompt = """Give a quick, supportive tip for a user with no major posture issues.
Include a small improvement tip and an encouraging statement.
"""
        no_deflect_resp = call_gemini_api(no_deflect_prompt)
        st.success(f"**🌟 Keep It Up!** {no_deflect_resp}")
else:
    st.info("Click **Analyze** to autoplay the videos & simulate the chart.")
