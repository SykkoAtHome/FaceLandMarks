import cv2
import mediapipe as mp
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh


class VideoFile:
    def __init__(self, video_file, detect_conf: float = 0.5, trk_conf: float = 0.5, name: str = None):
        if not 0 <= detect_conf <= 1:
            raise ValueError("Detection confidence must be in the range from 0 to 1.")
        if not 0 <= trk_conf <= 1:
            raise ValueError("Tracking confidence must be in the range from 0 to 1.")
        self.name = name
        self.video_file = video_file
        self.image_seq = cv2.VideoCapture(video_file)
        self.total_frames = int(
            self.image_seq.get(cv2.CAP_PROP_FRAME_COUNT))  # Duration of video in frames. Starts at 0.
        self.fps = round(float(self.image_seq.get(cv2.CAP_PROP_FPS)), 3)
        self.frame_width = int(self.image_seq.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.image_seq.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=detect_conf, min_tracking_confidence=trk_conf)
        self.mp_landmarks = self.detect_landmarks()  # Raw, detected landmarks for each frame. Index is frame number.
        self.dataframe = self.landmarks_to_dataframe()  # Landmarks converted to dataframe.
        self.landmarks = self.mp_landmarks  # Placeholder for landmarks converted from dataframe.

    def detect_landmarks(self):
        if self.name:
            print(f"Detecting landmarks for {self.name}...")
        else:
            print("Detecting landmarks...")

        all_landmarks = {}
        frame_count = 0

        try:
            while True:
                ret, frame = self.image_seq.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = self.face_mesh.process(frame_rgb)

                if results.multi_face_landmarks:
                    for landmarks in results.multi_face_landmarks:
                        all_landmarks[frame_count] = landmarks
                frame_count += 1
        finally:
            self.release()

        return all_landmarks

    def landmarks_to_dataframe(self) -> pd.DataFrame:
        print("Converting landmarks to dataframe...")
        df_columns = ["frame", "landmark_id", "x", "y", "z"]
        df_data = []

        for frame_num, landmarks in self.mp_landmarks.items():
            for landmark_id, landmark in enumerate(landmarks.landmark):
                x, y, z = landmark.x, landmark.y, landmark.z
                df_data.append([frame_num, landmark_id, x, y, z])

        df_landmarks = pd.DataFrame(df_data, columns=df_columns)
        return df_landmarks

    def dataframe_to_landmarks(self):
        print("Converting dataframe to landmarks...")
        landmarks_by_frame = {}

        for frame_num, landmarks_info in self.dataframe.groupby("frame"):
            frame_landmarks = self.landmarks[frame_num]

            for _, row in landmarks_info.iterrows():
                landmark_id = row["landmark_id"]
                x, y, z = row["x"], row["y"], row["z"]

                if landmark_id < len(frame_landmarks.landmark):
                    frame_landmarks.landmark[landmark_id].x = x
                    frame_landmarks.landmark[landmark_id].y = y
                    frame_landmarks.landmark[landmark_id].z = z

            landmarks_by_frame[frame_num] = frame_landmarks
        self.landmarks = landmarks_by_frame

    def release(self):
        self.image_seq.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.image_seq.release()
