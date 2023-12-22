from typing import Tuple, Any

import mediapipe as mp
import pandas as pd


def landmarks_to_dataframe(landmarks_results) -> pd.DataFrame:
    print("Converting landmarks to dataframe...")
    df_columns = ["frame", "landmark_id", "x", "y", "z"]
    df_data = []

    for frame_num, landmarks in landmarks_results.items():
        for landmark_id, landmark in enumerate(landmarks.landmark):
            x, y, z = landmark.x, landmark.y, landmark.z
            df_data.append([frame_num, landmark_id, x, y, z])

    df_landmarks = pd.DataFrame(df_data, columns=df_columns)
    print("Converting landmarks to dataframe...Done !")
    return df_landmarks


def dataframe_to_landmarks(df, existing_landmarks):
    landmarks_by_frame = {}

    for row in df.itertuples():
        frame_num = row.frame
        landmark_id = row.landmark_id
        x = row.x
        y = row.y
        z = row.z

        landmark_list = existing_landmarks[frame_num]

        if landmark_id < len(landmark_list.landmark):
            landmark_list.landmark[landmark_id].x = x
            landmark_list.landmark[landmark_id].y = y
            landmark_list.landmark[landmark_id].z = z

        landmarks_by_frame[frame_num] = landmark_list

    return landmarks_by_frame


def get_landmark_coordinates_from_df(df, frame, landmark_id):
    selected_landmarks = df[(df['frame'] == frame) & (df['landmark_id'] == landmark_id)]

    if not selected_landmarks.empty:
        l_id = selected_landmarks['landmark_id'].values[0]
        x = selected_landmarks['x'].values[0]
        y = selected_landmarks['y'].values[0]
        z = selected_landmarks['z'].values[0]
        return {"id": l_id, "x": x, "y": y, "z": z}
    else:
        return None
