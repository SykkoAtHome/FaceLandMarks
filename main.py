import cv2

from data_processing import landmarks_to_dataframe, dataframe_to_landmarks, savgol_df
from image_input import get_image_sequence
from landmarks import detect_landmarks, mp_face_mesh, mp_draw

file_name = "video/lb.mp4"
image_seq, total_frames, original_fps = get_image_sequence(file_name)

current_frame = 0

# detect landmarks
all_frames_landmarks = detect_landmarks(image_seq)

# convert landmarks to dataframe
df_landmarks = landmarks_to_dataframe(all_frames_landmarks)

# apply savgol filter to dataframe
df_landmarks_filtered = savgol_df(df_landmarks, window_length=10, polyorder=0)

# convert dataframe back to landmarks
landmarks_from_df = dataframe_to_landmarks(df_landmarks, all_frames_landmarks)

# draw landmarks on image
while image_seq.isOpened():

    ret, frame = image_seq.read()
    if not ret:
        break

    frame_landmarks = landmarks_from_df[current_frame]

    if frame_landmarks:
        mp_draw.draw_landmarks(frame, frame_landmarks)

    cv2.imshow(file_name, frame)
    delay = int(1000 / original_fps)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

    current_frame += 1

image_seq.release()
cv2.destroyAllWindows()
