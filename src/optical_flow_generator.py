from frame_generator import FrameGenerator
from constants import MODEL_INPUT_IMG_SIZE

import pathlib
import numpy as np
import cv2


SMALLEST_DIM = 256


def resize(img):
    original_width = int(img.shape[1])
    original_height = int(img.shape[0])
    aspect_ratio = original_width / original_height

    if original_width < original_height:
        new_width = SMALLEST_DIM
        new_height = int(new_width / aspect_ratio)  # should be int
    else:
        new_height = SMALLEST_DIM
        new_width = int(new_height * aspect_ratio)  # should be int

    dim = (new_width, new_height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

    return resized_img


def crop_center(img, new_size):
    y, x, chnls = img.shape
    cropx, cropy = new_size

    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2

    new_img = img[starty:starty+cropy, startx:startx+cropx]
    return new_img


def rescale_pixel_values(img):
    # convert pixel values' type from uint8 to float32
    img = img.astype('float32')
    img_norm_zero_one = img / 255.0
    img_norm_minus_one_one = img_norm_zero_one * 2 - 1
    return img_norm_minus_one_one


def compute_optical_flow(prev, curr):
    # use OpenCV to compute optical flow out of frames
    optical_flow = cv2.optflow.createOptFlow_DualTVL1()
    flow_frame = optical_flow.calc(prev, curr, None)
    # clip pixel values (of computed flow) into [-20, 20]
    flow_frame_clipped = np.clip(flow_frame, -20, 20)
    # normalize the values into [-1, 1]
    flow_frame_norm = flow_frame_clipped / 20.0
    return flow_frame_norm


def preprocess_flow(flow_frame):
    frame_resized = resize(flow_frame)
    # crop the frame to the size the model expects
    frame_cropped = crop_center(
        frame_resized, *MODEL_INPUT_IMG_SIZE)
    # no rescaling here unlike preprocess_rgb -> but there is reshaping of the flow frame data
    frame_reshaped = np.reshape(
        frame_cropped, (1, *MODEL_INPUT_IMG_SIZE, 2))
    return frame_reshaped


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
