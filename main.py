import data_processing
import image_draw

from file_input import FileInput

file_input = FileInput("video/al2-short.mp4", name="Anna", motion_vectors=True)

processing = data_processing.DataProcessing(file_input, auto=False, fill=False, refine=True)

show = image_draw.ImageDraw(processing)
# show.source_image(landmarks=True).export("output/anna_landmarks.mp4")
show.source_image(landmarks=True, refined=True).show(loop=True)




