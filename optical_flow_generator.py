from frame_generator import FrameGenerator
from preprocess_clips import compute_optical_flow, preprocess_flow
from constants import MODEL_INPUT_IMG_SIZE

import pathlib
import numpy as np
import cv2


class OpticalFlowStream:
    def __init__(self, videos_dir: pathlib.Path, idx_2_label: dict, training: bool = False, instance_idx: list = [], n_frames: int = 74):
        self.frame_generator = FrameGenerator(
            videos_dir, idx_2_label, training, instance_idx, n_frames)

    def __call__(self):
        for video_frames, label in self.frame_generator():
            prev = video_frames[0]
            prev = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
            flow_stream = np.zeros((1, *MODEL_INPUT_IMG_SIZE, 2))
            for curr in video_frames[1:]:
                curr = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
                flow = compute_optical_flow(prev, curr)
                flow = preprocess_flow(flow)
                flow_stream = np.append(flow_stream, flow, axis=0)
                prev = curr
            flow_stream = flow_stream[1:, :, :, :]
            yield flow_stream, label
