import data_processing
import image_draw

from file_input import FileInput

file_input = FileInput("video/al2.mp4", name="Anna", motion_vectors=False)

processing = data_processing.ImageProcessing(file_input, auto=True, fill=False)

show = image_draw.ImageDraw(processing)
show.source_image(landmarks=True, resize=800).export("output/anna_landmarks.mp4")


