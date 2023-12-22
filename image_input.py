import cv2


def get_image_sequence(file_name):
    image_seq = cv2.VideoCapture(file_name)
    total_frames = int(image_seq.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = round(float(image_seq.get(cv2.CAP_PROP_FPS)), 3)

    return image_seq, total_frames, original_fps
