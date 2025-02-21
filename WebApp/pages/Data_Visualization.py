import streamlit as st
import os
import cv2
import json
import pandas as pd
import plotly.graph_objects as go
import time
import numpy as np

ROOT_DIR = "Patient_Data"
st.set_page_config(layout="wide")
st.title("LymphFit Data Visualization")

# Sidebar
st.sidebar.header("Select Patient and Exercise")
if os.path.exists(ROOT_DIR):
    patients = sorted(
    [f for f in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, f))]
)
    selected_patient = st.sidebar.selectbox("Select Patient", options=patients)

    if selected_patient:
        exercises = os.listdir(os.path.join(ROOT_DIR, selected_patient))
        selected_exercise = st.sidebar.selectbox("Select Exercise", options=exercises)

        exercise_dir = os.path.join(ROOT_DIR, selected_patient, selected_exercise)

        # Detect files
        video_file_path = next((os.path.join(exercise_dir, f) for f in os.listdir(exercise_dir) if f.endswith(".mp4")), None)
        json_file_path = next((os.path.join(exercise_dir, f) for f in os.listdir(exercise_dir) if f.endswith(".json")), None)
        csv_file_path = next((os.path.join(exercise_dir, f) for f in os.listdir(exercise_dir) if f.endswith(".csv")), None)

        if not video_file_path:
            st.sidebar.error("No video file found")
        if not json_file_path:
            st.sidebar.error("No JSON file found")
        if not csv_file_path:
            st.sidebar.error("No CSV file found")
else:
    st.sidebar.error("Root directory not found!")


def resample_data(df, target_length):
    """
    Resamples the input dataframe to match the target length using interpolation.
    """
    if len(df) == 0:
        return df
    
    df = df.reset_index(drop=True)
    new_index = np.linspace(0, len(df) - 1, target_length)
    df_resampled = pd.DataFrame()
    
    for column in df.columns:
        df_resampled[column] = np.interp(new_index, np.arange(len(df)), df[column])

    return df_resampled


def plot_emg_graph(x, y, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="blue")))
    fig.update_layout(
        title=dict(text=title, font=dict(size=10)),
        xaxis=dict(title="Time (s)", tickfont=dict(size=8)),
        yaxis=dict(title="EMG (mV)", tickfont=dict(size=8)),
        height=200,
        margin=dict(l=30, r=30, t=40, b=30),
    )
    return fig


def process_data(video_file_path, json_file_path, csv_file_path, acc_placeholder, gyro_placeholder, emg_placeholders):
     # Load JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    accelerometer_data = pd.DataFrame.from_dict(data.get("accelerometerData", {}), orient="index")
    accelerometer_data.index = accelerometer_data.index.astype(int)
    accelerometer_data.sort_index(inplace=True)

    gyroscope_data = pd.DataFrame.from_dict(data.get("gyroscopeData", {}), orient="index")
    gyroscope_data.index = gyroscope_data.index.astype(int)
    gyroscope_data.sort_index(inplace=True)

    # Load EMG data
    emg_data = pd.read_csv(csv_file_path)
    time_column = "Time_Index"
    affected_channels = [col for col in emg_data.columns if "Affected" in col and "NonAffected" not in col]
    non_affected_channels = [col for col in emg_data.columns if "NonAffected" in col]

    cap = cv2.VideoCapture(video_file_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_per_frame = 1 / video_fps
    total_duration = total_frames * duration_per_frame

    # Normalize sensor data index to video duration
    sensor_timestamps = accelerometer_data.index
    normalized_sensor_timestamps = (sensor_timestamps / sensor_timestamps.max() * total_duration)

    # Load video
    cap = cv2.VideoCapture(video_file_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / video_fps  # Total duration in seconds

    # Resample sensor data to match total frames
    accelerometer_data = resample_data(accelerometer_data, total_frames)
    gyroscope_data = resample_data(gyroscope_data, total_frames)
    emg_data = resample_data(emg_data, total_frames)  # Resampling EMG to match frames

    progress_bar = st.progress(0)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= total_frames:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a 1:1 black background square
        square_size = 500  # Define a uniform square size
        background = np.zeros((square_size, square_size, 3), dtype=np.uint8)  # Create a black square

        # Resize the frame while maintaining aspect ratio
        h, w, _ = frame.shape
        scale = min(square_size / h, square_size / w)
        resized_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # Convert the frame's color from BGR to RGB
        #resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        resized_frame_rgb = resized_frame

        # Center the resized frame on the black square
        x_offset = (square_size - resized_frame_rgb.shape[1]) // 2
        y_offset = (square_size - resized_frame_rgb.shape[0]) // 2
        background[y_offset:y_offset + resized_frame_rgb.shape[0], x_offset:x_offset + resized_frame_rgb.shape[1]] = resized_frame_rgb

        # Display the centered video frame on the black square
        video_placeholder.image(background, use_column_width=True)

        # Update Accelerometer graph
        current_time = frame_idx * duration_per_frame
        sensor_indices = (normalized_sensor_timestamps <= current_time)[:len(accelerometer_data)]
        acc_current_data = accelerometer_data.loc[sensor_indices]
        acc_fig = go.Figure()
        acc_fig.add_trace(go.Scatter(y=acc_current_data["Acc_X"], mode="lines", name="Acc_X"))
        acc_fig.add_trace(go.Scatter(y=acc_current_data["Acc_Y"], mode="lines", name="Acc_Y"))
        acc_fig.add_trace(go.Scatter(y=acc_current_data["Acc_Z"], mode="lines", name="Acc_Z"))
        acc_fig.update_layout(
        title=dict(
            text="Accerometer Data",
            font=dict(size=10),  # Reduce title font size
            x=0.5,  # Center align the title
            xanchor="center"
        ),
        xaxis=dict(
            title="Time",
            tickfont=dict(size=8),  # Reduce x-axis tick font size
        ),
        yaxis=dict(
            title="Acceleration",
            tickfont=dict(size=8),  # Reduce y-axis tick font size
        ),
        height=200,  # Compact graph height
        margin=dict(l=10, r=10, t=30, b=10)  # Compact margins
    )
        acc_placeholder.plotly_chart(acc_fig, use_container_width=True)

        # Update Gyroscope graph
        sensor_indices = (normalized_sensor_timestamps <= current_time)[:len(gyroscope_data)]
        gyro_current_data = gyroscope_data.loc[sensor_indices]
        gyro_fig = go.Figure()
        gyro_fig.add_trace(go.Scatter(y=gyro_current_data["Gyro_X"], mode="lines", name="Gyro_X"))
        gyro_fig.add_trace(go.Scatter(y=gyro_current_data["Gyro_Y"], mode="lines", name="Gyro_Y"))
        gyro_fig.add_trace(go.Scatter(y=gyro_current_data["Gyro_Z"], mode="lines", name="Gyro_Z"))
        gyro_fig.update_layout(
        title=dict(
            text="Gyroscope Data",
            font=dict(size=10),  # Reduce title font size
            x=0.5,  # Center align the title
            xanchor="center"
        ),
        xaxis=dict(
            title="Time",
            tickfont=dict(size=8),  # Reduce x-axis title font size
        ),
        yaxis=dict(
            title="Angular Velocity",
            tickfont=dict(size=8),  # Reduce y-axis title font size
        ),
        height=200,  # Compact graph height
        margin=dict(l=10, r=10, t=30, b=10)  # Compact margins
    )
        gyro_placeholder.plotly_chart(gyro_fig, use_container_width=True)


        # EMG Visualization
        current_emg = emg_data.iloc[frame_idx]
        for ph, channel in zip(emg_placeholders["left"], affected_channels):
            ph.plotly_chart(plot_emg_graph(
                list(range(frame_idx)), 
                emg_data[channel].iloc[:frame_idx], 
                f"Affected Arm: {channel}"
            ), use_container_width=True)

        for ph, channel in zip(emg_placeholders["right"], non_affected_channels):
            ph.plotly_chart(plot_emg_graph(
                list(range(frame_idx)), 
                emg_data[channel].iloc[:frame_idx], 
                f"Non-Affected Arm: {channel}"
            ), use_container_width=True)

        time.sleep(1/video_fps)
        frame_idx += 1
        progress_bar.progress(frame_idx / total_frames)

    cap.release()


# Main Visualization Layout
if video_file_path and json_file_path and csv_file_path:
    col1, col_video, col2 = st.columns([2, 3, 2])
    
    with col_video:
        video_placeholder = st.empty()

    emg_placeholders = {
        "left": [col1.empty() for _ in range(3)],
        "right": [col2.empty() for _ in range(3)],
    }

    acc_placeholder = st.empty()
    gyro_placeholder = st.empty()

    if st.button("Start Visualization"):
        process_data(video_file_path, json_file_path, csv_file_path,
                     acc_placeholder, gyro_placeholder, emg_placeholders)
else:
    st.info("Please select a patient and exercise to begin.")