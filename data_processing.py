import time
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from file_input import FileInput

mp_face_mesh = mp.solutions.face_mesh


class DataProcessing:
    def __init__(self, file_input: FileInput, auto: bool = False, fill: bool = False, refine: bool = False):
        self.file_input = file_input
        self.face_mesh = None
        self.mp_landmarks = self.auto_detect_landmarks() if auto else self.detect_landmarks()  # Placeholder for landmarks detected by mediapipe
        self.dataframe = self.landmarks_to_dataframe()  # Placeholder for landmarks converted to dataframe
        self.dataframe_refined = None  # Placeholder for refined landmarks
        self.fill_blanks() if fill else None
        self.refine_landmarks() if refine else None

    def detect_landmarks(self):
        self.face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=.5,
                                               max_num_faces=1,
                                               static_image_mode=True)
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
        print(f"Landmarks detected in: {int(end_time - start_time)} seconds")
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
        df_columns = ["frame", "src", "landmark_id", "x", "y", "z", "x_px", "y_px"]
        df_data = []

        # Iterate over total number of frames
        for frame_num in range(self.file_input.total_frames):
            if frame_num in self.mp_landmarks:
                frame_landmarks = self.mp_landmarks[frame_num]

                for landmark_id, landmark in enumerate(frame_landmarks.landmark):
                    x, y, z = landmark.x, landmark.y, landmark.z
                    x_px, y_px = int(x * self.file_input.frame_width), int(y * self.file_input.frame_height)
                    df_data.append([frame_num, "mediapipe", landmark_id, x, y, z, x_px, y_px])

        df_landmarks = pd.DataFrame(df_data, columns=df_columns)
        if len(df_landmarks['frame'].unique()) < len(self.file_input.frame_seq):
            print(
                f"Warrning: Frames without landmarks: {len(self.file_input.frame_seq) - len(df_landmarks['frame'].unique())}")
        return df_landmarks

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
        if len(all_frames_without_landmarks) == 0:
            print("No frames to work with. Skipping interpolation.")
            return None
        else:
            print(f"Calculating interpolation for missing frames...")
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
        if len(all_frames_without_landmarks) == 0:
            print(f"Success. Frames without landmarks: {len(all_frames_without_landmarks)}")
        else:
            print(f"Warning. Frames without landmarks: {len(all_frames_without_landmarks)}")

        return None

    """ Refining """

    def refine_landmarks(self):
        print("Refining landmarks...")
        df_columns = ["frame", "src", "landmark_id", "x", "y", "z", "x_px", "y_px"]
        df_data = []

        for landmark_id in range(self.dataframe['landmark_id'].max() + 1):
            refined_landmarks = self.refine_landmark_with_motion(landmark_id)
            df_data.extend(refined_landmarks)

        # Stwórz DataFrame z zebranych danych
        self.dataframe_refined = pd.DataFrame(df_data, columns=df_columns)

    def refine_landmark_with_motion(self, landmark_id):
        refined_landmarks = []

        if self.file_input.mv is None:
            raise ValueError("Motion vectors not available.")

        # Filtruj ramki z danym landmark_id
        landmark_frames = self.dataframe[self.dataframe['landmark_id'] == landmark_id]

        for frame_num in landmark_frames['frame'].unique():
            frame_landmark = landmark_frames[landmark_frames['frame'] == frame_num].iloc[0]

            # Pobierz współrzędne punktu landmarka dla tej klatki
            x_px, y_px, z = frame_landmark['x_px'], frame_landmark['y_px'], frame_landmark['z']

            # Pobierz dane z poprzedniej klatki
            prev_frame_landmark = self.dataframe[
                (self.dataframe['landmark_id'] == landmark_id) &
                (self.dataframe['frame'] == frame_num - 1)
                ]

            # Sprawdź, czy istnieją dane z poprzedniej klatki
            if not prev_frame_landmark.empty:
                # Pobierz x_px i y_px z poprzedniej klatki
                prev_x_px = prev_frame_landmark['x_px'].values[0]
                prev_y_px = prev_frame_landmark['y_px'].values[0]
            else:
                # Jeśli nie, to ustaw x_px i y_px na aktualne
                prev_x_px, prev_y_px = x_px, y_px

            # Oblicz przesunięcie motion_vector
            # motion_vector = self.file_input.mv[frame_num - 1][prev_y_px, prev_x_px]
            motion_vector = self.file_input.mv[frame_num - 1]

            # Oblicz nowe x_px i y_px
            # new_x_px = int(x_px + motion_vector[0])
            # new_y_px = int(y_px + motion_vector[1])
            new_x_px = int(x_px + motion_vector[prev_y_px, prev_x_px, 0])
            new_y_px = int(y_px + motion_vector[prev_y_px, prev_x_px, 1])

            # Przelicz nowe współrzędne na piksele
            new_x = (new_x_px / self.file_input.frame_width)
            new_y = (new_y_px / self.file_input.frame_height)

            refined_landmarks.append([frame_num, "motion", landmark_id, new_x, new_y, z, new_x_px, new_y_px])

        return refined_landmarks
