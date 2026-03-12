import cv2


class VideoReader:
    def __init__(self, video_path, frame_skip=1):
        self.video_path = video_path
        self.frame_skip = frame_skip
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0

    def __iter__(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame = 0
        return self

    def __next__(self):
        for _ in range(self.frame_skip):
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                raise StopIteration
            self.current_frame += 1
        return frame, self.current_frame
    
    def release(self):
        self.cap.release()
    
    def get_total_frames(self):
        return self.frame_count
    

class VideoWriter:
    def __init__(self, output_path, frame_size, fps=30):
        self.output_path = output_path
        self.frame_size = frame_size
        self.fps = fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
