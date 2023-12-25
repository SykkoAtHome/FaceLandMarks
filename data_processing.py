import time
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.interpolate import griddata, interp1d

mp_face_mesh = mp.solutions.face_mesh


class ImageProcessing:
    def __init__(self, file_input, auto: bool = False, fill: bool = False):
        self.file_input = file_input
        self.face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=.5,
                                               max_num_faces=1,
                                               static_image_mode=True)
        self.mp_landmarks = self.auto_detect_landmarks() if auto else self.detect_landmarks()  # Placeholder for landmarks detected by mediapipe
        self.dataframe = self.landmarks_to_dataframe()  # Placeholder for landmarks converted to dataframe
        self.landmarks = {}
        self.fill_blanks() if fill else None

    def detect_landmarks(self):
        start_time = time.time()

        if self.file_input.name:
            print(f"Detecting landmarks for Video File Name: {self.file_input.name}...")
        else:
            print("Detecting landmarks for Video File...")

        all_landmarks = {}

        for frame_count, frame in enumerate(self.file_input.frame_seq):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    all_landmarks[frame_count] = landmarks
            # else:
            #     print(f"No landmarks detected for frame {frame_count}")

        end_time = time.time()
        print(f"Total detection time: {int(end_time - start_time)} seconds")
        return all_landmarks

    def auto_detect_landmarks(self):
        start_time = time.time()

        if self.file_input.name:
            print(f"Auto detecting landmarks for Video File Name: {self.file_input.name}...")
        else:
            print("Auto detecting landmarks for Video File...")

        all_landmarks = {}

        for frame_count, frame in enumerate(self.file_input.frame_seq):
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

        end_time = time.time()
        print(f"Total detection time: {int(end_time - start_time)} seconds")
        return all_landmarks

    def landmarks_to_dataframe(self):
        print("Converting landmarks to dataframe...")
        df_columns = ["frame", "src", "landmark_id", "x", "y", "z"]
        df_data = []

        # Iterate over total number of frames
        for frame_num in range(self.file_input.total_frames):
            if frame_num in self.mp_landmarks:
                frame_landmarks = self.mp_landmarks[frame_num]

                for landmark_id, landmark in enumerate(frame_landmarks.landmark):
                    x, y, z = landmark.x, landmark.y, landmark.z
                    df_data.append([frame_num, "mediapipe", landmark_id, x, y, z])

        df_landmarks = pd.DataFrame(df_data, columns=df_columns)
        return df_landmarks

    def dataframe_to_landmarks(self):
        print("Converting dataframe to landmarks...")
        landmarks_by_frame = {}

        for frame_num, landmarks_info in self.dataframe.groupby("frame"):
            frame_landmarks = self.mp_landmarks[frame_num]

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
        # Sprawdzamy, czy żądana klatka jest dostępna
        if frame_num >= len(self.file_input.frame_seq):
            raise ValueError(f"Frame {frame_num} not available.")

        # Pobieramy klatkę
        frame = self.file_input.frame_seq[frame_num]

        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Frame {frame_num} is not a valid image.")

        frame_landmarks_df = self.dataframe[self.dataframe['frame'] == frame_num]

        if frame_landmarks_df.empty:
            raise ValueError(f"No landmarks found for frame {frame_num}")

        x_coords = frame_landmarks_df['x'].tolist()
        y_coords = frame_landmarks_df['y'].tolist()
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        face_image = frame[
                     int(min_y * self.file_input.frame_height):int(max_y * self.file_input.frame_height),
                     int(min_x * self.file_input.frame_width):int(max_x * self.file_input.frame_width)
                     ]

        return face_image

    """ Filling Blanks """

    def fill_blanks(self) -> None:
        all_frames_without_landmarks = self.find_all_frames_without_landmarks()
        print(f"Frames without landmarks: {len(all_frames_without_landmarks)}")
        self.dataframe_interpolation()

    def find_all_frames_without_landmarks(self) -> list:
        all_frames_without_landmarks = []
        frames_with_landmarks = self.dataframe['frame'].unique()
        for frame_num in range(self.file_input.total_frames):
            if frame_num not in frames_with_landmarks:
                all_frames_without_landmarks.append(frame_num)

        return all_frames_without_landmarks

    @staticmethod
    def filter_single_frames(frame_list: list) -> list:
        single_fill = []
        for frame_num in frame_list:
            if not frame_num - 1 in frame_list and not frame_num + 1 in frame_list:
                single_fill.append(frame_num)
        return single_fill

    @staticmethod
    def group_frames(frame_list: list) -> list:
        seq_fill = []
        visited_frames = set()

        for frame_num in frame_list:
            if frame_num not in visited_frames:
                group = [frame_num]
                visited_frames.add(frame_num)

                while frame_num + 1 in frame_list:
                    frame_num += 1
                    group.append(frame_num)
                    visited_frames.add(frame_num)

                seq_fill.append(group)

        return seq_fill

    def single_frame_interpolation(self, frame_list: list, alpha: float = 0.5) -> None:
        for frame_num in frame_list:
            for landmark_id in range(self.dataframe['landmark_id'].max() + 1):
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

    def dataframe_interpolation(self, method: str = "linear") -> None:
        frames_without_landmarks = self.find_all_frames_without_landmarks()
        if len(frames_without_landmarks) == 0:
            print("No frames to work with.")
            return None

        for frame_num in frames_without_landmarks:
            for landmark_id in range(self.dataframe['landmark_id'].max() + 1):
                # Odszukaj dane do interpolacji
                known_frames = self.dataframe[(self.dataframe['landmark_id'] == landmark_id)
                                              & (~self.dataframe['frame'].isin([frame_num]))]['frame'].values

                if known_frames.size > 0:
                    known_values = self.dataframe[(self.dataframe['landmark_id'] == landmark_id)
                                                  & (self.dataframe['frame'].isin(known_frames))][
                        ['frame', 'x', 'y', 'z']].values

                    # Interpolacja liniowa dla każdej współrzędnej
                    interp_func_x = interp1d(known_values[:, 0], known_values[:, 1], kind=method,
                                             fill_value='extrapolate')
                    interp_func_y = interp1d(known_values[:, 0], known_values[:, 2], kind=method,
                                             fill_value='extrapolate')
                    interp_func_z = interp1d(known_values[:, 0], known_values[:, 3], kind=method,
                                             fill_value='extrapolate')

                    # Interpoluj brakującą klatkę
                    interpolated_x = interp_func_x(frame_num)
                    interpolated_y = interp_func_y(frame_num)
                    interpolated_z = interp_func_z(frame_num)

                    # Dodaj interpolowane wartości do dataframe dla danej klatki i landmark_id
                    self.dataframe = pd.concat([self.dataframe, pd.DataFrame({
                        'frame': [frame_num],
                        'src': ['interpolated'],
                        'landmark_id': [landmark_id],
                        'x': [interpolated_x],
                        'y': [interpolated_y],
                        'z': [interpolated_z]
                    })], ignore_index=True)
                else:
                    print(f"Warning: No data for frame {frame_num} and landmark {landmark_id}")
        all_frames_without_landmarks = self.find_all_frames_without_landmarks()
        print(f"Frames without landmarks: {len(all_frames_without_landmarks)}")
