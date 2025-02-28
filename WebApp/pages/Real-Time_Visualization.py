import streamlit as st
import cv2
import mediapipe as mp
import time
import os
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import json
from gtts import gTTS
import io
import base64

# ---------------- Gemini API Settings & Helper Functions ----------------
GEMINI_API_KEY = "AIzaSyCAxTziyQpvsKEjEBOmpiHyMTLd3wVITLc"  # Use your API key
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

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

def extract_deflections_from_df(df, buffer_sec=2):
    """
    From the aggregated similarity data, extract deflection events.
    A deflection is recorded only when the overall average similarity ('avg')
    is below 40% and is lower than the previous value (i.e. the average goes down).
    Groups are separated if more than 'buffer_sec' seconds apart.
    """
    low_similarity_threshold = 40
    joints = {"LS": "Left Shoulder", "RS": "Right Shoulder",
              "LE": "Left Elbow", "RE": "Right Elbow",
              "LW": "Left Wrist", "RW": "Right Wrist"}
    deflections = []
    prev_avg = None
    for index, row in df.iterrows():
        current_avg = row["avg"]
        # Only consider rows where avg is below threshold
        if current_avg < low_similarity_threshold:
            # Record a deflection if the average has gone down compared to the previous value
            if prev_avg is None or current_avg < prev_avg:
                current_deflections = []
                for key, name in joints.items():
                    if row[key] < low_similarity_threshold:
                        current_deflections.append(f"{name} at {row[key]:.1f}%")
                current_time = row["timestamp"]
                if current_deflections:
                    # Group events if they're more than buffer_sec apart
                    if len(deflections) == 0 or (current_time - deflections[-1]["timestamp"]) > buffer_sec:
                        deflections.append({"timestamp": current_time, "message": f"Time {current_time:.2f}s: " + ", ".join(current_deflections)})
            prev_avg = current_avg
        else:
            prev_avg = None  # Reset if the average is not below threshold
    if deflections:
        return [d["message"] for d in deflections]
    else:
        return ["No major deflections detected where avg < 60%."]

# ---------------- MediaPipe & Pose Setup ----------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Global variable to store the latest expert pose landmarks.
global_expert_landmarks = None

# --------------------------------------------------------
# Map Exercises to Their Video Files with Updated Paths
# --------------------------------------------------------
exercise_videos = {
    "Deep Breathing": "Reference_Videos/DeepBreathing10.mp4",
    "Shoulder Rolls": "Reference_Videos/ShoulderRolls10.mp4",
    "Clasp and Spread": "Reference_Videos/ClaspandSpread10.mp4",
    "Horizontal Pumping": "Reference_Videos/HorizontalPumping10.mp4",
    "Pushdown Pumping": "Reference_Videos/PushdownPumping10.mp4",
    "Overhead Pumping": "Reference_Videos/OverheadPumping10.mp4"
}

def compute_pose_error_components(expert_landmarks, user_landmarks):
    """
    Compute the Euclidean distance (in normalized coordinates) for 6 key landmarks.
    Returns a dictionary with keys "LS", "RS", "LE", "RE", "LW", "RW" and "avg".
    """
    indices = {
        "LS": 11,
        "RS": 12,
        "LE": 13,
        "RE": 14,
        "LW": 15,
        "RW": 16
    }
    errors = {}
    total = 0
    count = 0
    for label, idx in indices.items():
        expert_lm = expert_landmarks.landmark[idx]
        user_lm = user_landmarks.landmark[idx]
        dx = expert_lm.x - user_lm.x
        dy = expert_lm.y - user_lm.y
        dist = (dx**2 + dy**2)**0.5
        errors[label] = dist
        total += dist
        count += 1
    errors["avg"] = total / count if count > 0 else None
    return errors

def compute_pose_similarity_components(expert_landmarks, user_landmarks, threshold):
    """
    Using the error for 6 key landmarks, maps each error to a similarity percentage.
      - 0 error → 100% similarity.
      - Error equal to or above 'threshold' → 0% similarity.
    Returns a dictionary with keys "LS", "RS", "LE", "RE", "LW", "RW", and "avg".
    """
    errors = compute_pose_error_components(expert_landmarks, user_landmarks)
    similarities = {}
    for key, err in errors.items():
        if err is None:
            similarities[key] = None
        else:
            perc = max(0, min(100, 100 * (1 - err / threshold)))
            similarities[key] = perc
    return similarities

def crop_to_portrait(frame):
    """
    Crop the given frame to a ~9:16 portrait aspect ratio.
    """
    target_aspect = 9 / 16.0
    h, w, _ = frame.shape
    current_aspect = w / float(h)
    if abs(current_aspect - target_aspect) < 1e-3:
        return frame
    if current_aspect > target_aspect:
        new_width = int(h * target_aspect)
        x_start = (w - new_width) // 2
        frame = frame[:, x_start:x_start+new_width]
    else:
        new_height = int(w / target_aspect)
        y_start = (h - new_height) // 2
        frame = frame[y_start:y_start+new_height, :]
    return frame

def letterbox_frame(frame, target_width, target_height, color=(0, 0, 0)):
    """
    Resize and pad the frame to fit within the target dimensions while preserving aspect ratio.
    """
    h, w, _ = frame.shape
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h))
    letterboxed = np.full((target_height, target_width, 3), color, dtype=resized_frame.dtype)
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    letterboxed[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
    return letterboxed

def get_portrait_bbox_dimensions(cap, default=(400, 700), padding_ratio=0.1):
    """
    Returns bounding box dimensions in portrait mode based on the webcam feed, with extra padding.
    """
    if cap is not None and cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if h > w:
            width, height = w, h
        else:
            width = int((9/16) * h)
            height = h
        padded_width = int(width * (1 + padding_ratio))
        padded_height = int(height * (1 + padding_ratio))
        return padded_width, padded_height
    return default

def main():
    st.title("Your Personal Lymphatic Coach!")
    
    # Initialize session state for similarity data.
    if "similarity_data" not in st.session_state:
        st.session_state["similarity_data"] = []

    # ----------------------------------
    # SIDEBAR: Controls
    # ----------------------------------
    st.sidebar.title("Controls")
    selected_exercise = st.sidebar.selectbox("Select an Exercise", list(exercise_videos.keys()))
    
    # Start and Stop buttons for the reference video.
    if "start_pressed" not in st.session_state:
        st.session_state["start_pressed"] = False
    if st.sidebar.button("Start Reference Video"):
        st.session_state["start_pressed"] = True
        st.session_state["start_time"] = time.time()
        st.session_state["similarity_data"] = []  # Reset data.
    if st.sidebar.button("Stop Reference Video"):
        st.session_state["start_pressed"] = False
    
    # Checkbox for starting the webcam.
    if "start_webcam" not in st.session_state:
        st.session_state["start_webcam"] = False
    st.sidebar.checkbox("Start Webcam", key="start_webcam")

    # ----------------------------------
    # MAIN LAYOUT: Two Columns
    # ----------------------------------
    col1, col2 = st.columns(2)
    expert_placeholder = col1.empty()  # For expert video display.
    webcam_placeholder = col2.empty()    # For webcam feed display.

    # -------------------------------
    # Expert Video Setup (Left Column)
    # -------------------------------
    video_file = exercise_videos[selected_exercise]
    if not os.path.exists(video_file):
        col1.error(f"Expert video not found: {video_file}")
        return

    expert_cap = cv2.VideoCapture(video_file)
    frame_rate_expert = 25  # FPS for expert video.
    prev_time_expert = time.time()

    # -------------------------------
    # Webcam Setup (Right Column)
    # -------------------------------
    cap = None
    if st.session_state["start_webcam"]:
        cap = cv2.VideoCapture(0)             #Change 0 to 1 or other number depending on source
    frame_rate_webcam = 15
    prev_time_webcam = time.time()

    # Determine bounding box dimensions for display.
    bb_width, bb_height = get_portrait_bbox_dimensions(cap, default=(400, 700), padding_ratio=0.1)

    # Define threshold for similarity mapping.
    similarity_threshold = 0.1  # Error threshold (error of 0.1 maps to 0% similarity)

    global global_expert_landmarks
    global_expert_landmarks = None

    # Main loop: run until the expert video completes one iteration.
    iteration_done = False
    while True:
        now_expert = time.time()
        # --- Expert Video Playback ---
        if st.session_state["start_pressed"]:
            elapsed = now_expert - st.session_state["start_time"]
            if elapsed < 5:
                countdown = 5 - int(elapsed)
                expert_placeholder.write(f"Starting in {countdown} seconds...")
            else:
                if now_expert - prev_time_expert > 1.0 / frame_rate_expert:
                    prev_time_expert = now_expert
                    ret_e, frame_expert = expert_cap.read()
                    if not ret_e:
                        st.session_state["start_pressed"] = False
                        expert_placeholder.write("Reference video completed.")
                        iteration_done = True
                        break
                    else:
                        # Process expert frame for pose landmarks.
                        expert_frame_rgb = cv2.cvtColor(frame_expert, cv2.COLOR_BGR2RGB)
                        expert_results = pose.process(expert_frame_rgb)
                        global_expert_landmarks = expert_results.pose_landmarks
                        # Display the expert frame with letterboxing.
                        letterboxed_expert = letterbox_frame(frame_expert, bb_width, bb_height)
                        expert_placeholder.image(letterboxed_expert, channels="BGR")
        else:
            expert_placeholder.write("Press 'Start Reference Video' to begin.")

        # --- Webcam Feed Update (if enabled) ---
        if st.session_state["start_webcam"]:
            now_webcam = time.time()
            if now_webcam - prev_time_webcam > 1.0 / frame_rate_webcam:
                prev_time_webcam = now_webcam
                ret_w, frame_webcam = cap.read()
                if ret_w:
                    frame_webcam = cv2.flip(frame_webcam, 1)  # Mirror effect.
                    frame_webcam = crop_to_portrait(frame_webcam)
                    frame_rgb = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB)
                    user_results = pose.process(frame_rgb)
                    
                    # Compute similarity percentages if both expert and user poses are available.
                    if global_expert_landmarks is not None and user_results.pose_landmarks is not None:
                        similarities = compute_pose_similarity_components(global_expert_landmarks,
                                                                          user_results.pose_landmarks,
                                                                          similarity_threshold)
                        # Save similarity data along with a timestamp.
                        st.session_state["similarity_data"].append({"timestamp": now_webcam, **similarities})
                        # Overlay color: green if average similarity is at least 50%, else red.
                        if similarities["avg"] is not None and similarities["avg"] >= 50:
                            overlay_color = (0, 255, 0)  # green
                        else:
                            overlay_color = (0, 0, 255)  # red
                    else:
                        overlay_color = (255, 255, 255)
                    
                    # Draw user's pose landmarks with the chosen overlay color.
                    if user_results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            frame_webcam,
                            user_results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=overlay_color, thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=overlay_color, thickness=2, circle_radius=2),
                        )
                    webcam_placeholder.image(frame_webcam, channels="BGR")
        else:
            webcam_placeholder.write("Webcam not started.")

        time.sleep(0.01)

    # End of exercise session: release resources.
    expert_cap.release()
    if cap is not None:
        cap.release()

    # Clear the webcam feed placeholder.
    webcam_placeholder.empty()

    df = pd.DataFrame(st.session_state["similarity_data"])
    fig = px.line(df, x="timestamp", y=["LS", "RS", "LE", "RE", "LW", "RW", "avg"],
                  labels={"value": "Similarity (%)", "timestamp": "Time (s)"},
                  title="Pose Similarity Over Time (Upper Body)")

    # --- Generate LLM Feedback After Exercise ---
    st.markdown("---")
    st.header("LLM Voice Feedback After Exercise")
    deflections = extract_deflections_from_df(df)
    # Build exercise context to inform the LLM.
    exercise_context = f"You are a Lymphatic coach providing guidance for the exercise '{selected_exercise}'."
    if deflections and deflections[0] != "No major deflections detected where avg < 60%.":
        prompt = (
            f"{exercise_context} Provide a 100 word friendly tip for improving posture and lymphatic drainage based on the following detected deflections. (First congratulate the user on finishing the exercise.):\n"
            f"Deflections: {', '.join(deflections)}\n\n"
            "Include one small improvement tip and end with an encouraging statement."
        )
        general_feedback = call_gemini_api(prompt)
    else:
        prompt = (
            f"{exercise_context} Provide a quick, supportive tip for a user with no major posture issues. "
            "Include a small improvement tip and an encouraging statement."
        )
        general_feedback = call_gemini_api(prompt)
    
    # Check if we already have feedback audio stored in session state.
    if "feedback_audio" not in st.session_state:
        # Convert the feedback text to speech using gTTS.
        tts = gTTS(general_feedback)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
    
        # Encode the audio in base64.
        audio_bytes = audio_fp.getvalue()
        encoded_audio = base64.b64encode(audio_bytes).decode()
        st.session_state["feedback_audio"] = encoded_audio
    else:
        encoded_audio = st.session_state["feedback_audio"]
    
    # Create an HTML audio element with autoplay enabled.
    audio_html = f"""
    <audio autoplay controls style="width: 100%;">
      <source src="data:audio/mp3;base64,{encoded_audio}" type="audio/mp3">
      Your browser does not support the audio element.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)
    
    # Simulate the feedback text (typewriter effect) while audio plays.
    feedback_text_placeholder = st.empty()
    simulated_text = ""
    for char in general_feedback:
        simulated_text += char
        feedback_text_placeholder.markdown(simulated_text)
        time.sleep(0.05)
    
    # --- Display Similarity Graph ---
    st.write("### Similarity Graph (% Similarity Over Time)")
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
