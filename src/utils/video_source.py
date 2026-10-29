from pathlib import Path
import cv2
from tqdm import tqdm


class VideoSource:
    def __init__(self, source_path):
        self.source = Path(source_path)
        self.is_file_source = None
        self.cache_path = None
        self.current_frame_index = 0

        if self.source.is_dir():
            self.frames = sorted(self.source.glob("*.jpg"))
            self.is_file_source = False
        else:
            self.is_file_source = True
            self.cache_path = self._extract_video()
            self.frames = sorted(self.cache_path.glob("*.jpg"))

    def __len__(self):
        return len(self.frames)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_frame_index >= len(self.frames):
            raise StopIteration
        frame = self._read_frame(self.frames[self.current_frame_index])
        self.current_frame_index += 1
        return frame

    def __getitem__(self, index):
        if index >= len(self.frames):
            raise IndexError(f"Invalid frame index {index} for video with length {len(self.frames)}")
        return self._read_frame(self.frames[index])

    @staticmethod
    def _read_frame(frame_path):
        # imread signals every failure, including running out of memory mid-decode, by
        # returning None rather than raising.
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise IOError(f"cv2.imread failed to read frame {frame_path} "
                          f"(missing, corrupt, or the process is out of memory)")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    def _extract_video(self):
        video_name = self.source.stem
        cache_dir = (self.source.parent / video_name).with_name(f"{video_name}_cache")
        cache_dir.mkdir(exist_ok=True)
        print("Cache dir: ", cache_dir)
        image_files = list(cache_dir.glob("*.jpg"))

        video = cv2.VideoCapture(str(self.source))
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        video_frames_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if len(image_files) == video_frames_num:
            print(f"Cache already exists for video {self.source}, skipping extraction")
            video.release()
            return cache_dir

        frame_idx = 0
        padding_length = len(str(video_frames_num)) + 2
        status_bar = tqdm(total=int(video.get(cv2.CAP_PROP_FRAME_COUNT)), desc=f"Extracting frames from {self.source.name}")
        while True:
            ret, frame = video.read()
            if not ret:
                break
            cv2.imwrite(cache_dir / f"{frame_idx:0{padding_length}d}.jpg", frame)
            status_bar.update(1)
            frame_idx += 1
        video.release()
        status_bar.close()
        print(f"Ended video extraction")
        return cache_dir

