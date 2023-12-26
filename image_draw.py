import cv2
import numpy as np

from data_processing import ImageProcessing


class ImageDraw:
    def __init__(self, image_processing: ImageProcessing):
        self.file_input = image_processing.file_input
        self.landmarks = image_processing.dataframe
        self.landmarks_color = {'mediapipe': (0, 255, 0), 'interpolated': (0, 165, 255)}  # Green and Orange colors
        self.seq_buffer = []

    def source_image(self, start_frame: int = None, end_frame: int = None,
                     resize: int = None, landmarks: bool = False) -> "ImageDraw":
        if self.file_input.frame_seq is None:
            print("Image sequence not found. Did you provide a video file or images?")
            return self
        if start_frame is None:
            start_frame = 0

        if end_frame is None or end_frame >= self.file_input.total_frames:
            end_frame = self.file_input.total_frames - 1

        for frame_num in range(start_frame, end_frame + 1):
            frame = self.file_input.frame_seq[frame_num]
            # print(type(frame))
            # Process all frames, regardless of whether they have landmarks
            frame = self.draw_landmarks_on_frame(frame, self.landmarks[
                self.landmarks['frame'] == frame_num]) if landmarks else None

            # print(type(frame))
            # Resize the frame if requested
            if resize:
                frame = cv2.resize(frame, (resize, int(frame.shape[0] * (resize / frame.shape[1]))))

            # Buffer the image to seq_buffer
            self.seq_buffer.append(frame)
            # print(f"Frame dimensions after resizing: {frame.shape}")
        return self

    def draw_landmarks_on_frame(self, frame, landmarks_df):
        if landmarks_df.empty:
            return frame

        modified_frame = frame.copy()  # Create a copy of the frame

        for _, row in landmarks_df.iterrows():
            landmark_src = row['src']
            color = self.landmarks_color.get(landmark_src, (0, 0, 255))
            x, y = int(row['x'] * self.file_input.frame_width), int(row['y'] * self.file_input.frame_height)
            cv2.circle(modified_frame, (x, y), 3, color, -1)

        return modified_frame  # Return the modified frame

    def motion_vectors(self, start_frame: int = None, end_frame: int = None,
                       resize: int = None, landmarks: bool = False) -> "ImageDraw":
        if self.file_input.mv is None:
            print("Motion vectors not found. Set motion_vectors=True when initializing FileInput.")
            return self

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.file_input.total_frames

        for frame_num in range(start_frame, min(end_frame, self.file_input.total_frames - 1)):
            flow = self.file_input.mv[frame_num].get()  # Convert UMat to NumPy array

            # Calculate vector magnitudes and angles
            magnitudes, angles = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Handle invalid values in angles
            angles = np.where(np.isnan(angles) | np.isinf(angles), 0, angles)

            # Create an RGB image to visualize the motion vectors
            hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
            hsv[..., 1] = 255
            hsv[..., 0] = (angles * 180 / np.pi / 2).astype(np.uint8)

            # Handle invalid values in magnitudes
            magnitudes = np.where(np.isnan(magnitudes) | np.isinf(magnitudes), 0, magnitudes)

            hsv[..., 2] = cv2.normalize(magnitudes, None, 0, 255, cv2.NORM_MINMAX)

            # Convert the HSV image to BGR for visualization
            motion_vectors_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Draw landmarks on the image if requested
            motion_vectors_img = self.draw_landmarks_on_frame \
                (motion_vectors_img, self.landmarks[self.landmarks['frame'] == frame_num]) if landmarks else None

            # Resize the image if requested
            if resize is not None:
                motion_vectors_img = cv2.resize(motion_vectors_img, (resize, resize))

            # Buffer the image
            self.seq_buffer.append(motion_vectors_img)
        return self

    def show(self, loop: bool = False, fps: int = None):
        while True:
            key = 0  # Initialize key before the loop

            for frame in self.seq_buffer:
                draw_start = cv2.getTickCount()  # Start the timer

                if self.file_input.name:
                    cv2.imshow(f"File name: {self.file_input.name}", frame)
                    window_name = f"File name: {self.file_input.name}"
                else:
                    cv2.imshow("Image", frame)
                    window_name = "Image"

                if fps:
                    speed = 1000 / fps
                else:
                    speed = 1000 / self.file_input.fps if self.file_input.fps != 25 else 40
                key = cv2.waitKey(int(speed))

                # Additional control before closing the window
                if key == 27:  # 27 is the ESC key code
                    break

                # Check if the window exists before attempting to close
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                draw_end = cv2.getTickCount()  # End the timer

                # Calculate the actual FPS and display it on the frame
                actual_fps = cv2.getTickFrequency() / (draw_end - draw_start)

            if not loop or key == ord('q'):
                break

        cv2.destroyAllWindows()

    def export(self, output_path: str, fps: int = None, codec: str = None):
        print("Exporting video file...")
        fps = fps if fps else self.file_input.fps
        if codec is None:
            codec = "None"

        if not self.seq_buffer:
            print("No frames in the buffer. Use source_image() or motion_vectors() to populate the buffer.")
            return

        height, width, layers = self.seq_buffer[0].shape
        size = (width, height)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, size)

        for frame in iter(self.seq_buffer):
            if frame is not None:
                out.write(frame)

        out.release()
        print(f"Sequence saved to {output_path}")