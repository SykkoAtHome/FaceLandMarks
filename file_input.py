import cv2
import os
import glob


class FileInput:
    def __init__(self, input_path, name: str = None, motion_vectors: bool = False):
        self.name = name
        self.input_path = input_path

        # Initialize attributes
        self.total_frames: int = 0
        self.frame_seq = None
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.fps: float = 25.0  # Default fps for image sequence. Will be updated if input is a video file
        # Load frames and set other attributes
        self.load_frames()
        self.mv = self.motion_vectors() if motion_vectors else None

    def load_frames(self):
        self.frame_seq = []
        print("Loading frames...")
        # Check if the input path is a video file
        if os.path.isfile(self.input_path):
            print("Video file detected.")
            self.load_video_frames()
        # Check if the input path is a folder containing image sequence
        elif os.path.isdir(self.input_path):
            print("Image sequence detected.")
            self.load_image_sequence_frames()
        else:
            raise ValueError("Invalid input_path. Provide a valid video file or folder with images")

    def load_video_frames(self):
        if self.name:
            print(f"Video name: {self.name}")
        cap = cv2.VideoCapture(self.input_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames: {self.total_frames}")
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Frame: {self.frame_width} x {self.frame_height} "
              f"({round(float(self.frame_width / self.frame_height), 4)})")
        self.fps = round(float(cap.get(cv2.CAP_PROP_FPS)), 3)
        print(f"FPS: {self.fps}")

        # Read frames into frame_seq
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_seq.append(frame)

        cap.release()

    def load_image_sequence_frames(self):
        image_files = sorted(glob.glob(os.path.join(self.input_path, '*.jpg')) +
                             glob.glob(os.path.join(self.input_path, '*.png')) +
                             glob.glob(os.path.join(self.input_path, '*.dpx')) +
                             glob.glob(os.path.join(self.input_path, '*.exr')))

        if not image_files:
            raise ValueError("No images found in the specified folder.")

        # Read the first image to get dimensions
        first_image = cv2.imread(image_files[0])

        # Read all images into frame_seq
        for file_path in image_files:
            frame = cv2.imread(file_path)
            self.frame_seq.append(frame)

        if self.name:
            print(f"Image sequence name: {self.name}")
        self.total_frames = len(self.frame_seq)
        print(f"Total frames: {self.total_frames}")
        self.frame_width = first_image.shape[1]
        self.frame_height = first_image.shape[0]
        print(f"Frame: {self.frame_width} x {self.frame_height} "
              f"({round(float(self.frame_width / self.frame_height), 4)})")
        print(f"FPS: {self.fps} (default)")

    def motion_vectors(self):
        self.mv = []
        start_time = cv2.getTickCount()
        # Initialize motion vectors list
        motion_vectors_list = []
        for i in range(self.total_frames):
            print("Calculating motion vectors for frame ", i)
            prev_frame = cv2.UMat(cv2.cvtColor(self.frame_seq[i - 1], cv2.COLOR_BGR2GRAY))
            current_frame = cv2.UMat(cv2.cvtColor(self.frame_seq[i], cv2.COLOR_BGR2GRAY))

            flow = cv2.calcOpticalFlowFarneback(
                prev=prev_frame,
                next=current_frame,
                flow=None,
                pyr_scale=.8,
                levels=5,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

            flow = flow.get()  # Convert UMat to NumPy array
            motion_vectors_list.append(flow)

        end_time = cv2.getTickCount()
        print("Motion Vectors Calculated in ", round((end_time - start_time) / cv2.getTickFrequency(), 2), " seconds.")
        return motion_vectors_list
