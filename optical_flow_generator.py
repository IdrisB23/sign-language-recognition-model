from frame_generator import FrameGenerator
from preprocess_clips import compute_optical_flow, preprocess_flow

import pathlib
import numpy as np
import json
import cv2


class OpticalFlowStream:
    def __init__(self, videos_dir: pathlib.Path, idx_2_label: dict, training: bool = False, instance_idx: list = [], n_frames: int = 74):
        self.frame_generator = FrameGenerator(
            videos_dir, idx_2_label, training, instance_idx, n_frames)

    def __call__(self):
        for video_frames, label in self.frame_generator():
            prev = video_frames[0]
            prev = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
            flow_stream = np.zeros((1, 224, 224, 2))
            for curr in video_frames[1:]:
                curr = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
                flow = compute_optical_flow(prev, curr)
                flow = preprocess_flow(flow)
                flow_stream = np.append(flow_stream, flow, axis=0)
                prev = curr
            flow_stream = flow_stream[1:, :, :, :]
            yield flow_stream, label


data_dir_path = pathlib.Path('data')
output_dir_path = data_dir_path / 'output'
frames_dir_path = output_dir_path / 'frames'
videos_dir_path = data_dir_path / 'videos' / 'train'
cropped_videos_dir_path = videos_dir_path / 'cropped'
dataset_dir_path = data_dir_path / 'dataset'
cleansed_training_data_path = pathlib.Path('cleansed_train_ds.json')

with open(cleansed_training_data_path, 'r') as f_:
    training_data = json.load(f_)
    f_.close()

instance_idx = [45, 46, 59, 121, 122, 178, 179, 180, 181, 184, 199, 200, 275, 276, 277, 278, 279, 292, 316, 328, 329, 330, 334, 340, 367,
                396, 421, 470, 477, 596, 633, 634, 635, 636, 637, 644, 686, 700, 775, 885, 995, 1084, 1149, 1241, 1310, 1421, 1424, 1661, 1967, 2037]
# fg = OpticalFlowStream(cropped_videos_dir_path,
#                        training_data, True, instance_idx)
# i = 0
# for flow_stream, label in fg():
#     print(i, flow_stream.shape, label)
#     i += 1
