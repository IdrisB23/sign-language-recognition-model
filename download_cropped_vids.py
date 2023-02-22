# idris.babay@cgi.com
# 23.01.2023

'''
This file allows the downloading and cropping of the dataset videos to their corresponding Region Of Interest (ROI) 
'''

# Imports
import pathlib
import re
import pafy
import ffmpeg
# import numpy as np
# import collections

import constants
from utilities import read_dict_from_JSON, save_dict_as_JSON_file

# Constants
VID_SIZE_THRESHOLD = 25000000

DATA_ATTR = ['org_text', 'clean_text', 'start_time', 'signer_id', 'signer', 'start',
             'end', 'file', 'label', 'height', 'fps', 'end_time', 'url', 'text', 'box', 'width']

# Change these constants to TRAIN/VAL/TEST
DATA = constants.VAL_DATA
DOWNLOADED_DIR = constants.VAL_VIDS_DWNLD_DIR
TRIMMED_DIR = constants.VAL_VIDS_TRIMMED_DIR
CROPPED_DIR = constants.VAL_VIDS_CROPPED_DIR
IDX_2_INIT_DWNLD_IDX_W_SAME_URL_PATH = constants.VAL_IDX_2_INIT_DWNLD_PATH
INSTANCE_2_TRIMMED_PATH = constants.VAL_INSTANCE_2_TRIMMED_PATH
INSTANCE_2_CROPPED_PATH = constants.VAL_INSTANCE_2_CROPPED_PATH


# Create directories and files if not existant
if not DOWNLOADED_DIR.exists():
    DOWNLOADED_DIR.mkdir(parents=True)
if not TRIMMED_DIR.exists():
    TRIMMED_DIR.mkdir(parents=True)
if not CROPPED_DIR.exists():
    CROPPED_DIR.mkdir(parents=True)


# Create JSON files if not already done that will serve as permanent storage of dicts over the potentially multiple runs
# and write an empty dict to them to avoid errors when attempting to parse their content
placeholder_dict = dict()
if not IDX_2_INIT_DWNLD_IDX_W_SAME_URL_PATH.exists():
    save_dict_as_JSON_file(
        IDX_2_INIT_DWNLD_IDX_W_SAME_URL_PATH, placeholder_dict)
if not INSTANCE_2_TRIMMED_PATH.exists():
    save_dict_as_JSON_file(INSTANCE_2_TRIMMED_PATH, placeholder_dict)
if not INSTANCE_2_CROPPED_PATH.exists():
    save_dict_as_JSON_file(INSTANCE_2_CROPPED_PATH, placeholder_dict)

# read JSON files as python dicts
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


def remove_http_or_https_from_url(url: str):
    return re.sub(r'(http|https)://', '', url)


def clean_up_files_in_dir(DIR_: pathlib.Path):
    for path_ in DIR_.iterdir():
        if path_.is_file():
            path_.unlink()


def trim_in_file_and_save_to_out_file(in_file: pathlib.Path, out_file: pathlib.Path, start: int, end: int):
    # to avoid overriding confirmation prompt of the ffmpeg-cli
    if out_file.is_file():
        out_file.unlink()

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


def remove_normalization_coordinates(x_norm: float, y_norm: float, width: float, height: float):
    '''
    Since the coordinates are normalized to the width/height of the video, we simply multiply by them appropriately
    '''
    x = round(x_norm * width)
    y = round(y_norm * height)
    return x, y


def crop_video(in_vid: pathlib.Path, out_vid: pathlib.Path, x: float, y: float, width: float, height: float):
    '''
    (x, y) correspond to the coordinates of the top-left corner to start cropping from
    width determines how far to travel horizontally from the vertical line defined by the x parameter
    height determines how far to travel vertically from the horizontal line defined by the y parameter
    '''
    if not out_vid.exists():
        input_stream = ffmpeg.input(in_vid)
        pts = 'PTS-STARTPTS'
        cropped = ffmpeg.crop(input_stream, x, y, width, height).setpts(pts)

        output = ffmpeg.output(cropped, out_vid, format='mp4')
        output.run()


def update_instances_dwnld_state_w_same_vid_url(j: int, instance):
    instance_url = instance['url']
    for i, instance_tmp in enumerate(DATA):
        if instance_tmp['url'] == instance_url:
            # all instances w same url point to the index that initiated download
            IDX_2_INIT_DWNLD_IDX_W_SAME_URL[str(i)] = str(j)
    IDX_2_INIT_DWNLD_IDX_W_SAME_URL[str(j)] = str(-2)  # means downloaded


def download_unique_vid_of_instance_and_update_map(i: int, instance):
    vid_url = remove_http_or_https_from_url(instance['url'])
    print(i)
    dwnlnd_idx = IDX_2_INIT_DWNLD_IDX_W_SAME_URL.get(str(i), -1)
    # if it has not been downloaded before
    if dwnlnd_idx == -1:
        try:
            print(vid_url)
            YT_vid = pafy.new(vid_url)
            s = YT_vid.getbest(preftype='mp4')
            s_filesize = s.get_filesize()
            print('stream file size:', s_filesize)
            if s_filesize < VID_SIZE_THRESHOLD:  # because of pafy"s slow download rate, only download videos of size less than a certain threshold, remove this later on
                downloaded_vid_path = DOWNLOADED_DIR / (str(i) + '.mp4')
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


def download_unique_vids_of_instances_in_range_and_return_url2path_map(start: int, end: int):
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


def trim_instance_vid_and_update_map(i: int, instance):

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
    vid_path = DOWNLOADED_DIR / f'{dwnld_init_idx}.mp4'
    if INSTANCE_2_TRIMMED.get(str(i), None) is None:
        print(i, ':', vid_url, '=>', vid_path)
        class_path: pathlib.Path = TRIMMED_DIR / str(instance['clean_text'].replace(' ', '_'))
        if not class_path.exists():
            class_path.mkdir()
        clip_start = instance['start_time']
        clip_end = instance['end_time']
        trimmed_vid_path = class_path / f'{i}.mp4'
        trim_in_file_and_save_to_out_file(
            vid_path, trimmed_vid_path, clip_start, clip_end)
        INSTANCE_2_TRIMMED[str(i)] = trimmed_vid_path.resolve().__str__()

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


def trim_vids_for_instances_in_range_and_return_updated_map(start: int, end: int):
    '''
    The same functionality as trim_vids_for_indices_instances_and_return_updated_map
    But this one takes in start and end as a range for indices instead of an explicit list
    '''
    # maps each instance index (integer) to it's corresponding trimmed video path
    for i in range(start, end):
        instance = DATA[i]
        trim_instance_vid_and_update_map(i, instance)
    pass


def crop_vid_for_instance_and_update_map(i: int, instance):
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
    in_vid_path = pathlib.Path(in_vid_path)

    if INSTANCE_2_CROPPED.get(str(i), None) is None:
        class_path = CROPPED_DIR / str(instance['clean_text'].replace(' ', '_'))
        if not class_path.exists():
            class_path.mkdir(parents=True)
        out_vid_path: pathlib.Path = class_path / str(i) + '.mp4'
        print(i, ', cropping from source =>', in_vid_path)
        crop_video(in_vid_path, out_vid_path,
                   x_topleft, y_topleft, width, height)
        # update map/result
        INSTANCE_2_CROPPED[str(i)] = out_vid_path.resolve().__str__()
    else:
        print(i, ': [Already cropped, skipping...]')


def crop_vids_for_indices_instances_and_return_updated_map(indices):
    for i in indices:
        instance = DATA[i]
        crop_vid_for_instance_and_update_map(i, instance)
    pass


def crop_vids_for_instances_in_range_and_return_updated_map(start: int, end: int):
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


def dwnld_trim_crop_pipeline_range(range_start: int=0, range_end: int=0):
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
