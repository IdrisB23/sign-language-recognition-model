# Imports

import os
import subprocess
import shutil
import json
import ffmpeg
import numpy as np
import cv2

# Constants

DATA_DIR = 'data'

DATASET_DIR = os.path.join(DATA_DIR, 'dataset')

DS_CLASSES_PATH = os.path.join(DATASET_DIR, 'MSASL_classes.json')
DS_SYNONYMS_DATA_PATH = os.path.join(DATASET_DIR, 'MSASL_synonyms.json')

TRAIN_DATA_PATH = os.path.join(DATASET_DIR, 'MSASL_train.json')
VAL_DATA_PATH = os.path.join(DATASET_DIR, 'MSASL_val.json')
TEST_DATA_PATH = os.path.join(DATASET_DIR, 'MSASL_test.json')

ASSETS_DIR = os.path.join(DATA_DIR, 'assets')
VIDEOS_DIR = os.path.join(DATA_DIR, 'videos')

OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
OUTPUT_FRAMES_DIR = os.path.join(OUTPUT_DIR, 'frames')
OUTPUT_NPY_DIR = os.path.join(OUTPUT_DIR, 'npy_files')

CROPPED_VIDS_DIR = os.path.join(VIDEOS_DIR, 'cropped')

FRAME_RATE = 25
NUM_FRAMES_TO_EXTRACT = 74
SMALLEST_DIM = 256
IMAGE_CROP_SIZE = 224

# Dataset

with open(DS_CLASSES_PATH, 'r') as f:
    CLASSES = json.load(f)

with open(TRAIN_DATA_PATH, 'r') as f:
    TRAINING_DATA = json.load(f)

with open(VAL_DATA_PATH, 'r') as f:
    VAL_DATA = json.load(f)

with open(TEST_DATA_PATH, 'r') as f:
    TEST_DATA = json.load(f)


# Functions

# Video processing

def get_vid_duration(vid_path):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', vid_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    duration = float(result.stdout)
    return duration


def get_vid_fps(vid_path):
    probe = ffmpeg.probe(vid_path)
    video_info = next(s for s in probe['streams']
                      if s['codec_type'] == 'video')
    nums = video_info['r_frame_rate'].split('/')
    dividend = nums[0]
    dividend = float(dividend)
    divider = nums[1]
    divider = float(divider)
    fps = dividend / divider
    return fps

# directory whose path is path_output has to contain AT LEAST one frame


def loop_frames_to_complete_num_frames(extracted_frames_paths, path_output):
    num_extracted_frames = len(extracted_frames_paths)
    if num_extracted_frames == 0:
        raise ValueError(
            'directory whose path is path_output has to contain AT LEAST one frame')
    remaining_tb_created = NUM_FRAMES_TO_EXTRACT - num_extracted_frames
    i = 1
    while num_extracted_frames <= NUM_FRAMES_TO_EXTRACT:
        if i > num_extracted_frames:
            i = 1
        path_frame_tb_copied = os.path.join(
            path_output, 'frame_{:02d}.png'.format(i))
        path_new_frame = os.path.join(
            path_output, 'frame_{:02d}.png'.format(num_extracted_frames))
        shutil.copy(path_frame_tb_copied, path_new_frame)
        i += 1
        num_extracted_frames += 1
    pass

# extract a specific number of frames from video


def sample_video(video_path, path_output):
    if video_path.endswith(('.mp4', '.avi')):

        vid_duration = get_vid_duration(video_path)
        vid_fps = get_vid_fps(video_path)

        total_num_frames_in_vid = round(vid_duration * vid_fps)
        for_every_group_of_frames_extract_one = int(
            round(total_num_frames_in_vid / NUM_FRAMES_TO_EXTRACT))
        output_full_path = os.path.join(path_output, 'frame_%02d.png')

        if for_every_group_of_frames_extract_one > 1:
            ffmpeg_command = 'ffmpeg -i {0} -vf thumbnail={1},setpts=N/TB -r 1 -vframes {2} {3}'.format(
                video_path, for_every_group_of_frames_extract_one, NUM_FRAMES_TO_EXTRACT, output_full_path
            )
        else:
            ffmpeg_command = 'ffmpeg -i {0} -vf setpts=N/TB -r 1 -vframes {1} {2}'.format(
                video_path, NUM_FRAMES_TO_EXTRACT, output_full_path
            )

        os.system(ffmpeg_command)

        # if there aren't sufficient frames in the video to extract then make the video a loop
        extracted_frames_paths = os.listdir(path_output)
        if len(extracted_frames_paths) < NUM_FRAMES_TO_EXTRACT:
            extracted_frames_paths
            loop_frames_to_complete_num_frames(
                extracted_frames_paths, path_output)
    else:
        raise ValueError(
            'Video path is not that of video file (it should end with .mp4 or .avi)')
    pass

# /Video processing

# Image processing


def read_frames(frames_dir):
    frames_list = []
    for file in os.listdir(frames_dir):
        if file.endswith('.png'):
            full_file_path = os.path.join(frames_dir, file)
            frames_list.append(full_file_path)

    return sorted(frames_list)

# resize image to have the smallest dimension equal to some constant while keeping its original aspect ratio


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

# /Image processing

# Processing RGB Channel


def preprocess_rgb(rgb_frame):
    frame_resized = resize(rgb_frame)
    # crop the frame to the size the model expects
    frame_cropped = crop_center(
        frame_resized, (IMAGE_CROP_SIZE, IMAGE_CROP_SIZE))
    frame_w_rescaled_pix = rescale_pixel_values(frame_cropped)
    return frame_w_rescaled_pix


def run_rgb(sorted_frames_list):
    result = np.zeros((1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3))
    for full_file_path in sorted_frames_list:
        img = cv2.imread(full_file_path, cv2.IMREAD_UNCHANGED)
        img = preprocess_rgb(img)
        # reshape to image to the shape (1, size, size, color_channels)
        img_reshaped = np.reshape(
            img, (1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3))
        result = np.append(result, img_reshaped, axis=0)

    # discard the first element of the array containing zeors (dummy value initialized)
    result = result[1:, :, :, :]
    # insert a leading/wrapper axis -> shape becomes (1, frames, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3)
    result = np.expand_dims(result, axis=0)
    return result

# /Processing RGB Channel

# Processing Flow Channel


def preprocess_flow(flow_frame):
    frame_resized = resize(flow_frame)
    # crop the frame to the size the model expects
    frame_cropped = crop_center(
        frame_resized, (IMAGE_CROP_SIZE, IMAGE_CROP_SIZE))
    # no rescaling here unlike preprocess_rgb -> but there is reshaping of the flow frame data
    frame_reshaped = np.reshape(
        frame_cropped, (1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 2))
    return frame_reshaped

# iteratively invoked in run_flow


def compute_optical_flow(prev, curr):
    # use OpenCV to compute optical flow out of frames
    optical_flow = cv2.optflow.createOptFlow_DualTVL1()
    flow_frame = optical_flow.calc(prev, curr, None)
    # clip pixel values (of computed flow) into [-20, 20]
    flow_frame_clipped = np.clip(flow_frame, -20, 20)
    # normalize the values into [-1, 1]
    flow_frame_norm = flow_frame_clipped / 20.0
    return flow_frame_norm


def run_flow(sorted_frames_list):
    # read RGB frames in BGR format, convert them to GRAY format and save them into the list in the same order
    sorted_img_list = []
    for frame in sorted_frames_list:
        img = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sorted_img_list.append(img_gray)

    # are there exactly 2 channels in optical flow output?
    result = np.zeros((1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 2))
    # we cannot run optical flow without a predecessor frame -> initial previous will be the original gray image
    prev = sorted_img_list[0]
    for curr in sorted_img_list[1:]:
        flow = compute_optical_flow(prev, curr)
        flow = preprocess_flow(flow)
        result = np.append(result, flow, axis=0)
        prev = curr

    # discard the first element of the array containing zeors (dummy value initialized)
    result = result[1:, :, :, :]
    # insert a leading/wrapper axis -> shape becomes (1, frames, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 2)
    result = np.expand_dims(result, axis=0)
    return result

# /Processing Flow Channel

# Committing processed data to disk


def save_data_as_npy(sorted_frames_list, video_name, output_path):
    rgb = run_rgb(sorted_frames_list)
    npy_rgb_output = os.path.join(output_path, video_name + '_rgb.npy')
    np.save(npy_rgb_output, rgb)

    flow = run_flow(sorted_frames_list)
    npy_flow_output = os.path.join(output_path, video_name + '_flow.npy')
    np.save(npy_flow_output, flow)

# /Committing processed data to disk


def preprocess_vids_of_indices_instances(data, indices):
    beforehand_processed_indices = []
    if not os.path.exists(OUTPUT_NPY_DIR):
        os.mkdir(OUTPUT_NPY_DIR)
    for i in indices:
        instance = data[i]
        instance_class = instance['clean_text']
        instance_class_as_dir = instance_class.replace(' ', '_')
        instance_class_dir = os.path.join(
            OUTPUT_FRAMES_DIR, instance_class_as_dir)
        cropped_input_vid_path = os.path.join(
            CROPPED_VIDS_DIR, '{}.mp4'.format(i))
        if not os.path.exists(cropped_input_vid_path):
            print(i, ': [Clip doesn\'t exist, skipping...]')
            continue
        # if this is the first instance of that class being processed
        if not os.path.exists(instance_class_dir):
            os.mkdir(instance_class_dir)
        extracted_frames_dir = os.path.join(
            instance_class_dir, 'instance_{}'.format(i))
        if not os.path.exists(extracted_frames_dir):
            os.mkdir(extracted_frames_dir)
        # !!!makes assumption that the program will not be interrupted in the middle of preprocessing an instance (made up of: extract_frames -> save_NPY)
        # if frames are extracted it means this instance has already been preprocessed 
        else:
            print(i, ': [Already been preprocessed (check local storage!), skipping...]')
            beforehand_processed_indices.append(i)
            continue
        # extract frames from downloaded and cropped video
        sample_video(video_path=cropped_input_vid_path,
                     path_output=extracted_frames_dir)
        NPY_class_path = os.path.join(OUTPUT_NPY_DIR, instance_class_as_dir)
        if not os.path.exists(NPY_class_path):
            os.mkdir(NPY_class_path)
        sorted_frames_list = read_frames(extracted_frames_dir)
        print(i, ': [Saving it as NPY...]')
        save_data_as_npy(sorted_frames_list, video_name=str(i),
                         output_path=NPY_class_path)
        print(i, ': [Finished saving...]')
    return beforehand_processed_indices

# /Functions
