import streamlit as st
import cv2
import mediapipe as mp
import time
import os
import numpy as np
import pandas as pd
import plotly.express as px

# Initialize MediaPipe Pose for both expert and user processing.
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
    "Shoulder Rolls": "Reference_Videos/ShoulderRoll10.mp4",
    "Clasp and Spread": "Reference_Videos/ClaspandSpread10.mp4",
    "Horizontal Pumping": "Reference_Videos/HorizontalPumping10.mp4",
    "Pushdown Pumping": "Reference_Videos/PushdownPumping10.mp4",
    "Overhead Pumping": "Reference_Videos/OverheadPumping10.mp4"
}

def compute_pose_error_components(expert_landmarks, user_landmarks):
    """
    Compute the Euclidean distance (in normalized coordinates) for the 6 key landmarks:
    Left Shoulder (11), Right Shoulder (12), Left Elbow (13), Right Elbow (14),
    Left Wrist (15), and Right Wrist (16).
    
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
    Using the error computed for the 6 key landmarks, map each error to a similarity percentage.
    We use a linear mapping such that:
      - If error is 0, similarity is 100%.
      - If error is equal to or exceeds 'threshold', similarity is 0%.
    Returns a dictionary with keys "LS", "RS", "LE", "RE", "LW", "RW", and "avg" (all in %).
    """
    errors = compute_pose_error_components(expert_landmarks, user_landmarks)
    similarities = {}
    for key, err in errors.items():
        if err is None:
            similarities[key] = None
        else:
            # Compute percentage similarity.
            perc = max(0, min(100, 100 * (1 - err / threshold)))
            similarities[key] = perc
    return similarities

def crop_to_portrait(frame):
    """
    Crop the given frame to a ~9:16 portrait aspect ratio.
    (Used for the webcam feed.)
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
    Resize and pad the given frame to fit entirely within the target dimensions while preserving its aspect ratio.
    (Letterboxing: the entire frame is visible with padding.)
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
    Returns bounding box dimensions in portrait mode based on the webcam feed,
    with additional padding applied.
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
    st.title("Real-Time Pose Correction WebApp")
    
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
    frame_rate_expert = 15  # FPS for expert video.
    prev_time_expert = time.time()

    # -------------------------------
    # Webcam Setup (Right Colum # -------------------------------
    cap = None
    if st.session_state["start_webcam"]:
        cap = cv2.VideoCapture(0)
    frame_rate_webcam = 15
    prev_time_webcam = time.time()

    # Determine bounding box dimensions for the expert video display.
    bb_width, bb_height = get_portrait_bbox_dimensions(cap, default=(400, 700), padding_ratio=0.1)

    # Define a threshold for maximum acceptable error (for similarity mapping).
    # Here, error 0.10 corresponds to 0% similarity.
    similarity_threshold = 0.1

    global global_expert_landmarks
    global_expert_landmarks = None

    # Main loop: run until the reference video completes one iteration.
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
                        # Display the expert frame using letterboxing.
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
                        st.session_state["similarity_data"].append({"time": now_webcam, **similarities})
                        # For overlay: if average similarity is at least 50%, consider it a good match.
                        if similarities["avg"] is not None and similarities["avg"] >= 50:
                            overlay_color = (0, 255, 0)  # green
                        else:
                            overlay_color = (0, 0, 255)  # red
                    else:
                        overlay_color = (255, 255, 255)
                    
                    # Draw user's landmarks with the overlay color.
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

    # End of iteration: release resources.
    expert_cap.release()
    if cap is not None:
        cap.release()

    # --- Display Similarity Graph ---
    st.write("### Similarity Graph (% Similarity Over Time)")
    df = pd.DataFrame(st.session_state["similarity_data"])
    # Plot all 7 lines: LS, RS, LE, RE, LW, RW, avg.
    fig = px.line(df, x="timestamp", y=["Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "average_similarity"],
                  labels={"value": "Similarity (%)", "time": "Time (s)"},
                  title="Pose Similarity Over Time (Upper Body)")
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()