from typing import List
import cv2
import numpy as np


class PlayerCache:
    """Simple playback helper that caches frames for pause/step control.

    - append_frame(vis): append processed visualization frame
    - current(): return current frame (or None)
    - step_prev(), step_next(process_next_cb): navigate
    - toggle_pause(), is_paused()
    """
    def __init__(self):
        self.frames: List[np.ndarray] = []
        self.idx = -1
        self.paused = False
        self.video_ended = False

    def append_frame(self, vis: np.ndarray):
        self.frames.append(vis)
        self.idx = len(self.frames) - 1

    def current(self):
        if 0 <= self.idx < len(self.frames):
            return self.frames[self.idx]
        return None

    def toggle_pause(self):
        self.paused = not self.paused

    def is_paused(self):
        return self.paused

    def step_prev(self):
        if self.idx > 0:
            self.idx -= 1

    def step_next(self, process_next_cb=None):
        # move forward in cache if possible
        if self.idx < len(self.frames) - 1:
            self.idx += 1
            return
        # else, if provided, process one more frame and append
        if process_next_cb is not None:
            vis = process_next_cb()
            if vis is not None:
                self.append_frame(vis)
