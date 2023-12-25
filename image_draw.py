import cv2
import numpy as np

from data_processing import ImageProcessing


class ImageDraw:
    def __init__(self, image_processing: ImageProcessing):
        self.file_input = image_processing.file_input
        self.landmarks = image_processing.dataframe
        self.landmarks_color = {'mediapipe': (0, 255, 0), 'interpolated': (0, 165, 255)}  # Green and Orange colors

    def show_landmarks(self, start_frame: int = None, end_frame: int = None, resize: int = None) -> None:
        if start_frame is None:
            start_frame = 0

        if end_frame is None or end_frame >= self.file_input.total_frames:
            end_frame = self.file_input.total_frames - 1

        for frame_num in range(start_frame, end_frame + 1):
            frame = self.file_input.frame_seq[frame_num]
            frame_landmarks_df = self.landmarks[self.landmarks['frame'] == frame_num]

            if not frame_landmarks_df.empty:
                self.draw_landmarks_on_frame(frame, frame_landmarks_df)

                if resize:
                    frame = cv2.resize(frame, (resize, int(frame.shape[0] * (resize / frame.shape[1]))))

                cv2.imshow('Landmarks Visualization', frame)
                speed = 1000 / self.file_input.fps if self.file_input.fps != 25 else 40
                key = cv2.waitKey(int(speed))

                # Dodatkowa kontrola przed zamknięciem okna
                if key == 27:  # 27 to kod klawisza ESC
                    break

                # Sprawdzanie, czy okno istnieje przed próbą zamknięcia
                if cv2.getWindowProperty('Landmarks Visualization', cv2.WND_PROP_VISIBLE) < 1:
                    break

        cv2.destroyAllWindows()

    def draw_landmarks_on_frame(self, frame, landmarks_df):
        for _, row in landmarks_df.iterrows():
            landmark_src = row['src']
            color = self.landmarks_color.get(landmark_src, (0, 0, 255))  # Default to red if not found in dictionary
            x, y = int(row['x'] * self.file_input.frame_width), int(row['y'] * self.file_input.frame_height)
            cv2.circle(frame, (x, y), 3, color, -1)

    def export_frames(self, output_path: str, fps: int = 25) -> None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (self.file_input.frame_width, self.file_input.frame_height))

        for frame_num, frame in enumerate(self.file_input.frame_seq):
            frame_landmarks_df = self.landmarks[self.landmarks['frame'] == frame_num]

            if not frame_landmarks_df.empty:
                self.draw_landmarks_on_frame(frame, frame_landmarks_df)
                out.write(frame)

        out.release()

    def show_motion_vectors(self, start_frame: int = None, end_frame: int = None, resize: int = None) -> None:

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.file_input.total_frames

        for i in range(start_frame, min(end_frame, self.file_input.total_frames - 1)):
            flow = self.file_input.mv[i].get()  # Convert UMat to NumPy array

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

            # Resize the image if requested
            if resize is not None:
                motion_vectors_img = cv2.resize(motion_vectors_img, (resize, resize))

            # Display the motion vectors image
            cv2.imshow(f'Motion Vectors', motion_vectors_img)
            speed = 1000 / self.file_input.fps if self.file_input.fps != 25 else 40
            cv2.waitKey(int(speed))
        cv2.destroyAllWindows()
