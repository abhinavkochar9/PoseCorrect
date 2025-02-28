import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Open video file
cap = cv2.VideoCapture("Reference_Videos/ClaspandSpread1.mp44")

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video writer to save the output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("Reference_Videos/ClaspandSpread10.mp4", fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract face coordinates
            face_points = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                face_points.append((x, y))

            # Create a mask for the face
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            convex_hull = cv2.convexHull(np.array(face_points))
            cv2.fillConvexPoly(mask, convex_hull, 255)

            # Blur the entire image
            blurred_frame = cv2.GaussianBlur(frame, (99, 99), 30)

            # Extract only the blurred face region
            face_blurred = cv2.bitwise_and(blurred_frame, blurred_frame, mask=mask)

            # Extract the original non-face region
            face_mask_inv = cv2.bitwise_not(mask)
            background = cv2.bitwise_and(frame, frame, mask=face_mask_inv)

            # Combine blurred face and original background
            frame = cv2.add(background, face_blurred)

    # Write to output video
    out.write(frame)

    # Show the output frame (optional)
    cv2.imshow("Blurred Face", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()