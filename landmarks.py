import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils


def detect_landmarks(image_sequence):
    all_landmarks = {}
    frame_count = 0

    while True:
        ret, frame = image_sequence.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                all_landmarks[frame_count] = landmarks
        frame_count += 1

    image_sequence.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return all_landmarks
