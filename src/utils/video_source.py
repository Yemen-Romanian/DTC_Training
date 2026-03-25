from pathlib import Path
import cv2


class VideoSource:
    def __init__(self, source_path):
        self.source = Path(source_path)
        self.is_file_source = None
        self.current_frame_index = 0

        if self.source.is_dir():
            self.frames = sorted(self.source.glob("*.jpg"))
            self.is_file_source = False
        else:
            self.video = cv2.VideoCapture(str(self.source))
            self.is_file_source = True

    def __len__(self):
        if self.is_file_source is False:
            return len(self.frames)
        else:
            return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.is_file_source is False:
            if self.current_frame_index >= len(self.frames):
                raise StopIteration
            frame_path = self.frames[self.current_frame_index]
            frame = cv2.imread(str(frame_path))
            self.current_frame_index += 1
            return frame
        else:
            ret, frame = self.video.read()
            if not ret:
                self.video.release()
                raise StopIteration
            return frame
