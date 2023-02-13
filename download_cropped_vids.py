# idris.babay@cgi.com
# 23.01.2023

'''
This file allows the downloading and cropping of the dataset videos to their corresponding Region Of Interest (ROI) 
'''

# Imports

import os
import json
import re
import pafy
import ffmpeg
import numpy as np
import pathlib

# Constants
VID_SIZE_THRESHOLD = 10000000

DATA_DIR = 'data'

ASSETS_DIR = os.path.join(DATA_DIR, 'assets')

TRAIN_IDX_2_INIT_DWNLD_IDX_W_SAME_URL_PATH = os.path.join(
    ASSETS_DIR, 'train_idx_2_init_dwnld_idx_w_same_url.json')
TRAIN_INSTANCE_2_TRIMMED_PATH = os.path.join(
    ASSETS_DIR, 'train_instance_2_trimmed.json')
TRAIN_INSTANCE_2_CROPPED_PATH = os.path.join(
    ASSETS_DIR, 'train_instance_2_cropped.json')

VAL_IDX_2_INIT_DWNLD_IDX_W_SAME_URL_PATH = os.path.join(
    ASSETS_DIR, 'val_idx_2_init_dwnld_idx_w_same_url.json')
VAL_INSTANCE_2_TRIMMED_PATH = os.path.join(
    ASSETS_DIR, 'val_instance_2_trimmed.json')
VAL_INSTANCE_2_CROPPED_PATH = os.path.join(
    ASSETS_DIR, 'val_instance_2_cropped.json')

TEST_IDX_2_INIT_DWNLD_IDX_W_SAME_URL_PATH = os.path.join(
    ASSETS_DIR, 'test_idx_2_init_dwnld_idx_w_same_url.json')
TEST_INSTANCE_2_TRIMMED_PATH = os.path.join(
    ASSETS_DIR, 'test_instance_2_trimmed.json')
TEST_INSTANCE_2_CROPPED_PATH = os.path.join(
    ASSETS_DIR, 'test_instance_2_cropped.json')

DATASET_DIR = os.path.join(ASSETS_DIR, 'dataset')

TRAIN_DATA_PATH = os.path.join(DATASET_DIR, 'MSASL_train.json')
VAL_DATA_PATH = os.path.join(DATASET_DIR, 'MSASL_val.json')
TEST_DATA_PATH = os.path.join(DATASET_DIR, 'MSASL_test.json')

VIDEOS_DIR = os.path.join(DATA_DIR, 'videos')

TRAIN_VID_DIR = os.path.join(VIDEOS_DIR, 'train')
DOWNLOADED_TRAIN_DIR = os.path.join(TRAIN_VID_DIR, 'downloaded')
TRIMMED_TRAIN_DIR = os.path.join(TRAIN_VID_DIR, 'trimmed')
CROPPED_TRAIN_DIR = os.path.join(TRAIN_VID_DIR, 'cropped')

VAL_VID_DIR = os.path.join(VIDEOS_DIR, 'val')
DOWNLOADED_VAL_DIR = os.path.join(VAL_VID_DIR, 'downloaded')
TRIMMED_VAL_DIR = os.path.join(VAL_VID_DIR, 'trimmed')
CROPPED_VAL_DIR = os.path.join(VAL_VID_DIR, 'cropped')

TEST_VID_DIR = os.path.join(VIDEOS_DIR, 'test')
DOWNLOADED_TEST_DIR = os.path.join(TEST_VID_DIR, 'downloaded')
TRIMMED_TEST_DIR = os.path.join(TEST_VID_DIR, 'trimmed')
CROPPED_TEST_DIR = os.path.join(TEST_VID_DIR, 'cropped')

DATA_ATTR = ['org_text', 'clean_text', 'start_time', 'signer_id', 'signer', 'start',
             'end', 'file', 'label', 'height', 'fps', 'end_time', 'url', 'text', 'box', 'width']

# Utility functions


def save_dict_as_JSON_file(path_, data_):
    with open(path_, 'w') as f:
        json.dump(data_, f, indent=2)
        print('Saved dictionary to the JSON file.')
        f.close()


def read_dict_from_JSON(path_):
    with open(path_, 'r') as f:
        result = json.load(f)
        f.close()
    return result


# /Utility functions

# Change these constants to TRAIN/VAL/TEST
DOWNLOADED_DIR = DOWNLOADED_VAL_DIR
TRIMMED_DIR = TRIMMED_VAL_DIR
CROPPED_DIR = CROPPED_VAL_DIR
IDX_2_INIT_DWNLD_IDX_W_SAME_URL_PATH = VAL_IDX_2_INIT_DWNLD_IDX_W_SAME_URL_PATH
INSTANCE_2_TRIMMED_PATH = VAL_INSTANCE_2_TRIMMED_PATH
INSTANCE_2_CROPPED_PATH = VAL_INSTANCE_2_CROPPED_PATH
DATA_PATH = VAL_DATA_PATH


# Create directories and files if not existant
if not os.path.exists(DOWNLOADED_DIR):
    os.makedirs(DOWNLOADED_DIR)
if not os.path.exists(TRIMMED_DIR):
    os.makedirs(TRIMMED_DIR)
if not os.path.exists(CROPPED_DIR):
    os.makedirs(CROPPED_DIR)


# Create JSON files if not already done that will serve as permanent storage of dicts over the potentially multiple runs
# and write an empty dict to them to avoid errors when attempting to parse their content
placeholder_dict = dict()
if not os.path.exists(IDX_2_INIT_DWNLD_IDX_W_SAME_URL_PATH):
    save_dict_as_JSON_file(
        IDX_2_INIT_DWNLD_IDX_W_SAME_URL_PATH, placeholder_dict)
if not os.path.exists(INSTANCE_2_TRIMMED_PATH):
    save_dict_as_JSON_file(INSTANCE_2_TRIMMED_PATH, placeholder_dict)
if not os.path.exists(INSTANCE_2_CROPPED_PATH):
    save_dict_as_JSON_file(INSTANCE_2_CROPPED_PATH, placeholder_dict)

# read JSON files as python dicts
DATA = read_dict_from_JSON(DATA_PATH)
IDX_2_INIT_DWNLD_IDX_W_SAME_URL = read_dict_from_JSON(
    IDX_2_INIT_DWNLD_IDX_W_SAME_URL_PATH)
INSTANCE_2_TRIMMED = read_dict_from_JSON(INSTANCE_2_TRIMMED_PATH)
INSTANCE_2_CROPPED = read_dict_from_JSON(INSTANCE_2_CROPPED_PATH)


# Functions


def mycb(total, recvd, ratio, rate, eta):
    '''
    Callback function, this callback simply prints the total number of bytes to download, the number of bytes received thusfar, 
    the ratio of downloaded bytes, the rate of transfer, and estimated time of arrival (eta)
    '''
    print('total:', total, 'received:', recvd,
          'ratio:', ratio, 'rate:', rate, 'eta:', eta)


def remove_http_or_https_from_url(url):
    return re.sub(r'(http|https)://', '', url)


def clean_up_files_in_dir(DIR_):
    file_path_list = os.listdir(DIR_)
    for path_ in file_path_list:
        os.remove(os.path.join(DIR_, path_))


def trim_in_file_and_save_to_out_file(in_file, out_file, start, end):
    # to avoid overriding confirmation prompt of the ffmpeg-cli
    if os.path.exists(out_file):
        os.remove(out_file)

    # probe function delivers useful information about the signal file
    probe_result = ffmpeg.probe(in_file)
    in_file_duration = probe_result.get('format', {}).get('duration', {})
    print(in_file_duration)

    input_stream = ffmpeg.input(in_file)

    # video and audio channels need to be handled (filtered/trimmed) separately
    pts = 'PTS-STARTPTS'
    video = input_stream.trim(start=start, end=end).setpts(pts)
    audio = (input_stream
             .filter('atrim', start=start, end=end)
             .filter('asetpts', pts))
    # then we concatenate both channels
    video_and_audio = ffmpeg.concat(video, audio, v=1, a=1)

    output = ffmpeg.output(video_and_audio, out_file, format='mp4')
    output.run()


def remove_normalization_coordinates(x_norm, y_norm, width, height):
    '''
    Since the coordinates are normalized to the width/height of the video, we simply multiply by them appropriately
    '''
    x = round(x_norm * width)
    y = round(y_norm * height)
    return x, y


def crop_video(in_vid, out_vid, x, y, width, height):
    '''
    (x, y) correspond to the coordinates of the top-left corner to start cropping from
    width determines how far to travel horizontally from the vertical line defined by the x parameter
    height determines how far to travel vertically from the horizontal line defined by the y parameter
    '''
    input_stream = ffmpeg.input(in_vid)
    pts = 'PTS-STARTPTS'
    cropped = ffmpeg.crop(input_stream, x, y, width, height).setpts(pts)

    output = ffmpeg.output(cropped, out_vid, format='mp4')
    output.run()


def update_instances_dwnld_state_w_same_vid_url(j, instance):
    instance_url = instance['url']
    for i, instance_tmp in enumerate(DATA):
        if instance_tmp['url'] == instance_url:
            # all instances w same url point to the index that initiated download
            IDX_2_INIT_DWNLD_IDX_W_SAME_URL[str(i)] = str(j)
    IDX_2_INIT_DWNLD_IDX_W_SAME_URL[str(j)] = str(-2)  # means downloaded


def download_unique_vid_of_instance_and_update_map(i, instance):
    vid_url = remove_http_or_https_from_url(instance['url'])
    print(i)
    dwnlnd_idx = IDX_2_INIT_DWNLD_IDX_W_SAME_URL.get(str(i), -1)
    # if it has not been downloaded before
    if dwnlnd_idx == -1:
        try:
            YT_vid = pafy.new(vid_url)
            s = YT_vid.getbest(preftype='mp4')
            s_filesize = s.get_filesize()
            print('stream file size:', s_filesize)
            if s_filesize < VID_SIZE_THRESHOLD:  # because of pafy"s slow download rate, only download videos of size less than a certain threshold, remove this later on
                downloaded_vid_path = os.path.join(
                    DOWNLOADED_DIR, str(i) + '.mp4')
                s.download(filepath=downloaded_vid_path, quiet=True,
                           callback=mycb)  # starts download
                print(downloaded_vid_path)
                update_instances_dwnld_state_w_same_vid_url(
                    i, instance)
            else:
                print(i, '[Video stream excedes size threshold, skipping...]')
                return i
        except Exception as ex:
            template = str(
                i) + ' [An exception of type {0} occurred. Arguments:\n{1!r}]'
            message = template.format(type(ex).__name__, ex.args)
            print(message)
        finally:
            print('-------------------------------------------------')
    else:
        print(i, '[Duplicate video, skipping...]', vid_url)
        print('-------------------------------------------------')


def download_unique_vids_of_indices_instances_and_return_url2path_map(indices):
    '''
    This function downloads UNIQUE videos in instances in the data parameter given by the other parameter indices
    data: list of instances that have the same dataset attributes
    indices: list of integer indices
    returns: dictionary that maps each (unique) video url in the dataset to the downloaded video path (may vary depending on the order in which data instances are being reviewed)
    '''

    vids_w_size_over_threshold = []

    # go through data points one by one w.r.t. indices
    for i in indices:
        instance = DATA[i]
        result = download_unique_vid_of_instance_and_update_map(i, instance)
        if not result is None:
            vids_w_size_over_threshold.append(i)

    return vids_w_size_over_threshold


def download_unique_vids_of_instances_in_range_and_return_url2path_map(start, end):
    '''
    The same functionality as download_unique_vids_of_indices_instances_and_return_url2path_map
    But this one takes in start and end as a range for indices instead of an explicit list
    '''

    vids_w_size_over_threshold = []

    for i in range(start, end):
        instance = DATA[i]
        result = download_unique_vid_of_instance_and_update_map(i, instance)
        if not result is None:
            vids_w_size_over_threshold.append(i)

    return vids_w_size_over_threshold


def trim_instance_vid_and_update_map(i, instance):

    vid_url = remove_http_or_https_from_url(instance['url'])
    idx_dwnld_status = IDX_2_INIT_DWNLD_IDX_W_SAME_URL.get(str(i), str(-1))
    # if no hit in url2downloaded_vid, then the video was not downloaded as this function is run after downloading the vids in the pipeline
    if idx_dwnld_status == str(-1):
        print(
            i, ': [This instance\'s video is private or inaccessible, skipping...]')
        return
    if idx_dwnld_status == str(-2): # means instance itself initiated downloaded
        dwnld_init_idx = str(i)
    else:
        dwnld_init_idx = idx_dwnld_status # refer to the instance that initiated it instead as in the construction of IDX_2_INIT_DWNLD_IDX_W_SAME_URL
    vid_path = os.path.join(DOWNLOADED_DIR, f'{dwnld_init_idx}.mp4')
    if INSTANCE_2_TRIMMED.get(str(i), None) is None:
        print(i, ':', vid_url, '=>', vid_path)
        class_path = os.path.join(
            TRIMMED_DIR, instance['clean_text'].replace(' ', '_'))
        if not os.path.exists(class_path):
            os.mkdir(class_path)
        clip_start = instance['start_time']
        clip_end = instance['end_time']
        trimmed_vid_path = os.path.join(class_path, f'{i}.mp4')
        trim_in_file_and_save_to_out_file(
            vid_path, trimmed_vid_path, clip_start, clip_end)
        INSTANCE_2_TRIMMED[str(i)] = str(trimmed_vid_path)

    # if it exists in instance2trimmed, then it's already been trimmed before therefore skip it
    else:
        print(i, ': [Already trimmed, skipping...]')


def trim_vids_for_indices_instances_and_return_updated_map(indices):
    '''
    Looks up the url attribute of a given data_instance:=data[index as one of indices parameter] and then queries the url2downloaded_vid_path parameter. 
    If there's a key in the map then look up the path for that downloaded video and trim it according to the other data instance attributes 
    for start and end time of desired clip inside the whole video (one video may contain multiple clips for different ASL signs).
    data: list of instances that have the same dataset attributes
    indices: list of integer indices
    returns: dictionary that maps each instance to the trimmed video path
    '''
    # maps each instance index (integer) to it's corresponding trimmed video path
    for i in indices:
        instance = DATA[i]
        trim_instance_vid_and_update_map(i, instance)
    pass


def trim_vids_for_instances_in_range_and_return_updated_map(start, end):
    '''
    The same functionality as trim_vids_for_indices_instances_and_return_updated_map
    But this one takes in start and end as a range for indices instead of an explicit list
    '''
    # maps each instance index (integer) to it's corresponding trimmed video path
    for i in range(start, end):
        instance = DATA[i]
        trim_instance_vid_and_update_map(i, instance)
    pass


def crop_vid_for_instance_and_update_map(i, instance):
    '''
    Crops each instance in data provided by indices according to the instance's relevant ROI attributes
    data: list of instances that have the same dataset attributes
    indices: list of integer indices
    returns: dictionary that maps each instance to the cropped video path
    '''
    # relevant ROI attributes
    y_norm, x_norm, _, _ = instance['box']
    height, width = instance['height'], instance['width']

    x_topleft, y_topleft = remove_normalization_coordinates(
        x_norm, y_norm, width, height)

    # again according to the pipeline's order, if it's successful, it should not be None
    in_vid_path = INSTANCE_2_TRIMMED.get(str(i), None)
    if in_vid_path is None:
        print(
            i, ': [This instance\'s video is private or inaccessible, skipping...]')
        return

    if INSTANCE_2_CROPPED.get(str(i), None) is None:
        class_path = os.path.join(
            CROPPED_DIR, instance['clean_text'].replace(' ', '_'))
        if not os.path.exists(class_path):
            os.mkdir(class_path)
        out_vid_path = os.path.join(class_path, str(i) + '.mp4')
        print(i, ', cropping from source =>', in_vid_path)
        crop_video(in_vid_path, out_vid_path,
                   x_topleft, y_topleft, width, height)
        # update map/result
        INSTANCE_2_CROPPED[str(i)] = str(out_vid_path)
    else:
        print(i, ': [Already cropped, skipping...]')


def crop_vids_for_indices_instances_and_return_updated_map(indices):
    for i in indices:
        instance = DATA[i]
        crop_vid_for_instance_and_update_map(i, instance)
    pass


def crop_vids_for_instances_in_range_and_return_updated_map(start, end):
    '''
    The same functionality as crop_vids_for_indices_instances_and_return_updated_map
    But this one takes in start and end as a range for indices instead of an explicit list
    '''
    for i in range(start, end):
        instance = DATA[i]
        crop_vid_for_instance_and_update_map(i, instance)
    pass


def dwnld_trim_crop_pipeline_indices(indices=[]):
    '''
    Downloading, trimming, and then cropping will be done in batches due to the limitations in storage and computation power of the local machine.
    The batches are defined/delimited by the indices parameter
    '''

    # Download the unique instance videos for preprocessing
    vids_w_size_over_threshold = download_unique_vids_of_indices_instances_and_return_url2path_map(
        indices)

    print(IDX_2_INIT_DWNLD_IDX_W_SAME_URL_PATH)
    print(IDX_2_INIT_DWNLD_IDX_W_SAME_URL)
    # Save url_2_downloaded MAP as JSON file
    save_dict_as_JSON_file(IDX_2_INIT_DWNLD_IDX_W_SAME_URL_PATH,
                           IDX_2_INIT_DWNLD_IDX_W_SAME_URL)

    # Trim videos for further preprocessing
    trim_vids_for_indices_instances_and_return_updated_map(indices)

    # Save instance_2_trimmed MAP as JSON file
    save_dict_as_JSON_file(INSTANCE_2_TRIMMED_PATH, INSTANCE_2_TRIMMED)
    # Clean up downloaded videos
    # clean_up_files_in_dir(DOWNLOADED_DIR)

    # Cropping Videos to the desired region of interest and saving them
    crop_vids_for_indices_instances_and_return_updated_map(indices)

    # Save instance_2_cropped MAP as JSON file
    save_dict_as_JSON_file(INSTANCE_2_CROPPED_PATH, INSTANCE_2_CROPPED)
    # Clean up trimmed videos
    # clean_up_files_in_dir(TRIMMED_DIR)
    return vids_w_size_over_threshold


def dwnld_trim_crop_pipeline_range(range_start=0, range_end=0):
    '''
    Downloading, trimming, and then cropping will be done in batches due to the limitations in storage and computation power of the local machine.
    The batches are defined/delimited by the range_start and range_end parameters, which define a range of indices
    '''

    # Download the unique instance videos for preprocessing
    vids_w_size_over_threshold = download_unique_vids_of_instances_in_range_and_return_url2path_map(
        range_start, range_end)
    # Save url_2_downloaded MAP as JSON file
    save_dict_as_JSON_file(IDX_2_INIT_DWNLD_IDX_W_SAME_URL_PATH,
                           IDX_2_INIT_DWNLD_IDX_W_SAME_URL)

    # Trim videos for further preprocessing
    trim_vids_for_instances_in_range_and_return_updated_map(
        range_start, range_end)

    # Save instance_2_trimmed MAP as JSON file
    save_dict_as_JSON_file(INSTANCE_2_TRIMMED_PATH, INSTANCE_2_TRIMMED)
    # Clean up downloaded videos
    # clean_up_files_in_dir(DOWNLOADED_DIR)

    # Cropping Videos to the desired region of interest and saving them
    crop_vids_for_instances_in_range_and_return_updated_map(
        range_start, range_end)

    # Save instance_2_cropped MAP as JSON file
    save_dict_as_JSON_file(INSTANCE_2_CROPPED_PATH, INSTANCE_2_CROPPED)
    # Clean up trimmed videos
    # clean_up_files_in_dir(TRIMMED_DIR)
    return vids_w_size_over_threshold


# /Functions

# dummy_indices=[0, 1, 3, 8, 56, 60, 72, 110, 151, 499, 858, 1049]
# dwnld_trim_crop_pipeline_indices(training_data, indices=dummy_indices)
# dwnld_trim_crop_pipeline_range(training_data, 0, 150)
