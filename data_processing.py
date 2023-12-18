import mediapipe as mp
import pandas as pd

from scipy.signal import savgol_filter


def landmarks_to_dataframe(landmarks_results):
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


from scipy.signal import savgol_filter


def savgol_df(df, window_length, polyorder):
    # Function to apply savgol_filter to a single column
    def apply_savgol(column):
        return savgol_filter(column, window_length, polyorder)

    # Apply savgol_filter to x, y, z columns for each group
    df_filtered = df.groupby("frame").apply(lambda group: group.assign(
        x=apply_savgol(group["x"]),
        y=apply_savgol(group["y"]),
        z=apply_savgol(group["z"])
    ))

    return df_filtered.droplevel(0)  # Drop the outer level of the index
