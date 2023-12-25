import time

import cv2
import mediapipe as mp
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh


class ImageProcessing:
    def __init__(self, video_file, name: str = None, auto: bool = False):
        self.name = name
        self.video_file = video_file
        self.image_seq = cv2.VideoCapture(video_file)
        self.total_frames = int(
            self.image_seq.get(cv2.CAP_PROP_FRAME_COUNT))  # Duration of video in frames. Starts at 0.
        self.fps = round(float(self.image_seq.get(cv2.CAP_PROP_FPS)), 3)
        self.frame_width = int(self.image_seq.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.image_seq.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.face_mesh = mp_face_mesh.FaceMesh()
        if auto:
            self.mp_landmarks = self.auto_detect_landmarks()  # Raw, detected landmarks for each frame. Index is frame number.
        else:
            self.mp_landmarks = self.detect_landmarks()  # Raw, detected landmarks for each frame. Index is frame number.
        self.dataframe = self.landmarks_to_dataframe()  # Landmarks converted to dataframe.
        self.landmarks = self.mp_landmarks  # Placeholder for landmarks converted from dataframe.
        self.num_faces = None  # Placeholder for number of faces detected in video.

    def detect_landmarks(self):
        start_time = time.time()
        self.face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=.5,
                                               max_num_faces=1,
                                               static_image_mode=True)
        if self.name:
            print(f"Detecting landmarks for Video File Name: {self.name}...")
        else:
            print("Detecting landmarks for Video File...")

        all_landmarks = {}

        for frame_count in range(self.total_frames):
            ret, frame = self.image_seq.read()

            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:

                for landmarks in results.multi_face_landmarks:
                    all_landmarks[frame_count] = landmarks

        self.image_seq.set(cv2.CAP_PROP_POS_FRAMES, 0)
        end_time = time.time()
        print(f"Total detection time: {int(end_time - start_time)} seconds")
        return all_landmarks

    def auto_detect_landmarks(self):
        start_time = time.time()

        if self.name:
            print(f"Auto detecting landmarks for Video File Name: {self.name}...")
        else:
            print("Auto detecting landmarks for Video File...")

        all_landmarks = {}

        for frame_count in range(self.total_frames):
            ret, frame = self.image_seq.read()

            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            for min_detection_confidence in [round(x * 0.1, 1) for x in range(10, 0, -1)]:
                self.face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=min_detection_confidence,
                                                       max_num_faces=1,
                                                       static_image_mode=True)
                results = self.face_mesh.process(frame_rgb)

                if results.multi_face_landmarks:

                    for landmarks in results.multi_face_landmarks:
                        all_landmarks[frame_count] = landmarks
                    break

        self.image_seq.set(cv2.CAP_PROP_POS_FRAMES, 0)
        end_time = time.time()
        print(f"Total detection time: {int(end_time - start_time)} seconds")
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

    def extract_face(self, frame_num):
        frame_landmarks_df = self.dataframe[self.dataframe['frame'] == frame_num]

        if frame_landmarks_df.empty:
            raise ValueError(f"No landmarks found for frame {frame_num}")

        x_coords = frame_landmarks_df['x'].tolist()
        y_coords = frame_landmarks_df['y'].tolist()
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        self.image_seq.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.image_seq.read()

        if not ret or frame is None:
            raise ValueError(f"Error reading frame {frame_num} or frame is None")

        face_image = frame[
                     int(min_y * self.frame_height):int(max_y * self.frame_height),
                     int(min_x * self.frame_width):int(max_x * self.frame_width)
                     ]

        return face_image
###############################


    def detect_landmarks(self):
        start_time = time.time()

        if self.image_input.name:
            print(f"Detecting landmarks for Video File Name: {self.image_input.name}...")
        else:
            print("Detecting landmarks for Video File...")

        all_landmarks = {}

        for frame_count, frame in enumerate(self.image_input.frame_seq):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    all_landmarks[frame_count] = landmarks
            else:
                print(f"No landmarks detected for frame {frame_count}")

        end_time = time.time()
        print(f"Total detection time: {int(end_time - start_time)} seconds")
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



    def single_frame_interpolation(self, frame_list: list, alpha: float = 0.5) -> None:
        for frame_num in frame_list:
            for landmark_id in range(468):
                prev_frame_data = self.dataframe[(self.dataframe['frame'] == frame_num - 1)
                                                 & (self.dataframe['landmark_id'] == landmark_id)][
                    ['x', 'y', 'z']].values
                next_frame_data = self.dataframe[(self.dataframe['frame'] == frame_num + 1)
                                                 & (self.dataframe['landmark_id'] == landmark_id)][
                    ['x', 'y', 'z']].values

                if not (prev_frame_data.size == 0 or next_frame_data.size == 0):
                    alpha = alpha
                    interpolated_values = alpha * prev_frame_data + (1 - alpha) * next_frame_data

                    # Dodanie interpolowanych wartości do dataframe dla danej klatki i landmark_id
                    self.dataframe = pd.concat([self.dataframe, pd.DataFrame({
                        'frame': [frame_num],
                        'src': ['interpolated'],
                        'landmark_id': [landmark_id],
                        'x': [interpolated_values[0][0]],
                        'y': [interpolated_values[0][1]],
                        'z': [interpolated_values[0][2]]
                    })], ignore_index=True)
                else:
                    print(f"Warning: No data for frame {frame_num} and landmark {landmark_id}")