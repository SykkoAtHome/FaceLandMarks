

import data_processing
import image_draw

from file_input import FileInput

file_input = FileInput("video/Head Talking.mp4", name="Anna", motion_vectors=True)

processing = data_processing.ImageProcessing(file_input, auto=False, fill=False)

show = image_draw.ImageDraw(processing)
show.show_motion_vectors()
