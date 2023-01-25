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

# Constants

DOWNLOADED_VIDS_LIST_FILE_SEPARATOR = ' '
VID_SIZE_THRESHOLD = 12000000

DATA_DIR = 'data'

ASSETS_DIR = os.path.join(DATA_DIR, 'assets')

DOWNLOADED_VIDS_LIST_FILE = os.path.join(ASSETS_DIR, 'downloaded_vids.txt')

URL_2_DOWNLOADED_PATH = os.path.join(ASSETS_DIR, 'url_2_downloaded.json')
INSTANCE_2_TRIMMED_PATH = os.path.join(ASSETS_DIR, 'instance_2_trimmed.json')
INSTANCE_2_CROPPED_PATH = os.path.join(ASSETS_DIR, 'instance_2_cropped.json')

DATASET_DIR = os.path.join(ASSETS_DIR, 'dataset')

TRAIN_DATA_PATH = os.path.join(DATASET_DIR, 'MSASL_train.json')
VALID_DATA_PATH = os.path.join(DATASET_DIR, 'MSASL_val.json')
TEST_DATA_PATH = os.path.join(DATASET_DIR, 'MSASL_test.json')

VIDEOS_DIR = os.path.join(DATA_DIR, 'videos')

DOWNLOADED_DIR = os.path.join(VIDEOS_DIR, 'downloaded')
TRIMMED_DIR = os.path.join(VIDEOS_DIR, 'trimmed')
CROPPED_DIR = os.path.join(VIDEOS_DIR, 'cropped')

DATA_ATTR = ['org_text', 'clean_text', 'start_time', 'signer_id', 'signer', 'start',
             'end', 'file', 'label', 'height', 'fps', 'end_time', 'url', 'text', 'box', 'width']

# Load JSON datasets and parse them as Python dictionaries

with open(TRAIN_DATA_PATH, 'r') as f:
    training_data = json.load(f)
    f.close()
with open(VALID_DATA_PATH, 'r') as f:
    valid_data = json.load(f)
    f.close()
with open(TEST_DATA_PATH, 'r') as f:
    test_data = json.load(f)
    f.close()

# Create directories and files if not existant
if not os.path.exists(DOWNLOADED_DIR):
    os.mkdir(DOWNLOADED_DIR)
if not os.path.exists(TRIMMED_DIR):
    os.makedirs(TRIMMED_DIR)
if not os.path.exists(CROPPED_DIR):
    os.makedirs(CROPPED_DIR)

if not os.path.exists(DOWNLOADED_VIDS_LIST_FILE):
    with open(DOWNLOADED_VIDS_LIST_FILE, 'x') as f:
        f.close()

# Functions


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


# Create JSON files if not already done that will serve as permanent storage of dicts over the potentially multiple runs
# and write an empty dict to them to avoid errors when attempting to parse their content
placeholder_dict = dict()
if not os.path.exists(URL_2_DOWNLOADED_PATH):
    save_dict_as_JSON_file(URL_2_DOWNLOADED_PATH, placeholder_dict)
if not os.path.exists(INSTANCE_2_TRIMMED_PATH):
    save_dict_as_JSON_file(INSTANCE_2_TRIMMED_PATH, placeholder_dict)
if not os.path.exists(INSTANCE_2_CROPPED_PATH):
    save_dict_as_JSON_file(INSTANCE_2_CROPPED_PATH, placeholder_dict)

'''
Callback function, this callback simply prints the total number of bytes to download, the number of bytes received thusfar, 
the ratio of downloaded bytes, the rate of transfer, and estimated time of arrival (eta)
'''


def mycb(total, recvd, ratio, rate, eta):
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


'''
Since the coordinates are normalized to the width/height of the video, we simply multiply by them appropriately
'''


def remove_normalization_coordinates(x_norm, y_norm, width, height):
    x = round(x_norm * width)
    y = round(y_norm * height)
    return x, y


'''
(x, y) correspond to the coordinates of the top-left corner to start cropping from
width determines how far to travel horizontally from the vertical line defined by the x parameter
height determines how far to travel vertically from the horizontal line defined by the y parameter
'''


def crop_video(in_vid, out_vid, x, y, width, height):
    input_stream = ffmpeg.input(in_vid)
    pts = 'PTS-STARTPTS'
    cropped = ffmpeg.crop(input_stream, x, y, width, height).setpts(pts)

    output = ffmpeg.output(cropped, out_vid, format='mp4')
    output.run()


def download_unique_vid_of_instance_and_update_map(i, instance, url2downloaded_vid):
    vid_url = remove_http_or_https_from_url(instance['url'])
    # if it has not been downloaded before
    if url2downloaded_vid.get(vid_url, None) is None:
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
                url2downloaded_vid[vid_url] = downloaded_vid_path
            else:
                print(i, '[Video stream excedes size threshold, skipping...]')
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
        finally:
            print('-------------------------------------------------')
    else:
        print(i, '[Duplicate video, skipping...]', vid_url)
        print('-------------------------------------------------')


'''
This function downloads UNIQUE videos in instances in the data parameter given by the other parameter indices
data: list of instances that have the same dataset attributes
indices: list of integer indices
returns: dictionary that maps each (unique) video url in the dataset to the downloaded video path (may vary depending on the order in which data instances are being reviewed)
'''


def download_unique_vids_of_indices_instances_and_return_url2path_map(data, indices):
    # It loads url2path map from a potential previous run and this is also updated after each run
    # (see function preprocessing_pipeline... after THIS function's invocation)
    url2downloaded_vid = read_dict_from_JSON(URL_2_DOWNLOADED_PATH)

    # go through data points one by one w.r.t. indices
    for i in indices:
        instance = data[i]
        download_unique_vid_of_instance_and_update_map(
            i, instance, url2downloaded_vid)

    return url2downloaded_vid


'''
The same functionality as download_unique_vids_of_indices_instances_and_return_url2path_map
But this one takes in start and end as a range for indices instead of an explicit list
'''


def download_unique_vids_of_instances_in_range_and_return_url2path_map(data, start, end):

    url2downloaded_vid = read_dict_from_JSON(URL_2_DOWNLOADED_PATH)
    print('map_:', url2downloaded_vid)

    for i in range(start, end):
        instance = data[i]
        download_unique_vid_of_instance_and_update_map(
            i, instance, url2downloaded_vid)

    return url2downloaded_vid


def trim_instance_vid_and_update_map(i, instance, url2downloaded_vid, instance2trimmed):

    vid_url = remove_http_or_https_from_url(instance['url'])
    vid_path = url2downloaded_vid.get(vid_url, None)
    # if no hit in url2downloaded_vid, then the video was not downloaded as this function is run after downloading the vids in the pipeline
    if vid_path is None:
        print(
            i, ': [This instance\'s video is private or inaccessible, skipping...]')
        return

    if instance2trimmed.get(str(i), None) is None:
        print(i, ':', vid_url, '=>', vid_path)
        clip_start = instance['start_time']
        clip_end = instance['end_time']
        trimmed_path = os.path.join(TRIMMED_DIR, str(i) + '.mp4')
        trim_in_file_and_save_to_out_file(
            vid_path, trimmed_path, clip_start, clip_end)
        instance2trimmed[str(i)] = trimmed_path

    # if it exists in instance2trimmed, then it's already been trimmed before therefore skip it
    else:
        print(i, ': [Already trimmed, skipping...]')


'''
Looks up the url attribute of a given data_instance:=data[index as one of indices parameter] and then queries the url2downloaded_vid_path parameter. 
If there's a key in the map then look up the path for that downloaded video and trim it according to the other data instance attributes 
for start and end time of desired clip inside the whole video (one video may contain multiple clips for different ASL signs).
data: list of instances that have the same dataset attributes
indices: list of integer indices
returns: dictionary that maps each instance to the trimmed video path
'''


def trim_vids_for_indices_instances_and_return_updated_map(data, indices, url2downloaded_vid):
    # maps each instance index (integer) to it's corresponding trimmed video path
    instance2trimmed = read_dict_from_JSON(INSTANCE_2_TRIMMED_PATH)
    for i in indices:
        instance = data[i]
        trim_instance_vid_and_update_map(
            i, instance, url2downloaded_vid, instance2trimmed)
    return instance2trimmed


'''
The same functionality as trim_vids_for_indices_instances_and_return_updated_map
But this one takes in start and end as a range for indices instead of an explicit list
'''


def trim_vids_for_instances_in_range_and_return_updated_map(data, start, end, url2downloaded_vid):
    # maps each instance index (integer) to it's corresponding trimmed video path
    instance2trimmed = read_dict_from_JSON(INSTANCE_2_TRIMMED_PATH)
    for i in range(start, end):
        instance = data[i]
        trim_instance_vid_and_update_map(
            i, instance, url2downloaded_vid, instance2trimmed)
    return instance2trimmed


def crop_vid_for_instance_and_update_map(i, instance, instance2trimmed, instance2cropped):
    # relevant ROI attributes
    y_norm, x_norm, _, _ = instance['box']
    height, width = instance['height'], instance['width']

    x_topleft, y_topleft = remove_normalization_coordinates(
        x_norm, y_norm, width, height)

    # again according to the pipeline's order, if it's successful, it should not be None
    in_vid_path = instance2trimmed.get(str(i), None)
    if in_vid_path is None:
        print(
            i, ': [This instance\'s video is private or inaccessible, skipping...]')
        return

    if instance2cropped.get(str(i), None) is None:
        out_vid_path = os.path.join(CROPPED_DIR, str(i) + '.mp4')
        print(i, ', cropping from source =>', in_vid_path)
        crop_video(in_vid_path, out_vid_path,
                   x_topleft, y_topleft, width, height)
        # update map/result
        instance2cropped[str(i)] = out_vid_path
    else:
        print(i, ': [Already cropped, skipping...]')


'''
Crops each instance in data provided by indices according to the instance's relevant ROI attributes
data: list of instances that have the same dataset attributes
indices: list of integer indices
returns: dictionary that maps each instance to the cropped video path
'''


def crop_vids_for_indices_instances_and_return_updated_map(data, indices, instance2trimmed):
    instance2cropped = read_dict_from_JSON(INSTANCE_2_CROPPED_PATH)
    for i in indices:
        instance = data[i]
        crop_vid_for_instance_and_update_map(
            i, instance, instance2trimmed, instance2cropped)

    return instance2cropped


'''
The same functionality as crop_vids_for_indices_instances_and_return_updated_map
But this one takes in start and end as a range for indices instead of an explicit list
'''


def crop_vids_for_instances_in_range_and_return_updated_map(data, start, end, instance2trimmed):
    instance2cropped = read_dict_from_JSON(INSTANCE_2_CROPPED_PATH)
    for i in range(start, end):
        instance = data[i]
        crop_vid_for_instance_and_update_map(
            i, instance, instance2trimmed, instance2cropped)

    return instance2cropped


'''
Downloading, trimming, and then cropping will be done in batches due to the limitations in storage and computation power of the local machine.
The batches are defined/delimited by the indices parameter
'''


def dwnld_trim_crop_pipeline_indices(data, indices=[]):
    # Download the unique instance videos for preprocessing
    url_2_downloaded = download_unique_vids_of_indices_instances_and_return_url2path_map(
        data, indices)
    # Save url_2_downloaded MAP as JSON file
    save_dict_as_JSON_file(URL_2_DOWNLOADED_PATH, url_2_downloaded)

    # Trim videos for further preprocessing
    instance_2_trimmed = trim_vids_for_indices_instances_and_return_updated_map(
        data, indices, url_2_downloaded
    )

    # Save instance_2_trimmed MAP as JSON file
    save_dict_as_JSON_file(INSTANCE_2_TRIMMED_PATH, instance_2_trimmed)
    # Clean up downloaded videos
    # clean_up_files_in_dir(DOWNLOADED_DIR)

    # Cropping Videos to the desired region of interest and saving them
    instance_2_cropped = crop_vids_for_indices_instances_and_return_updated_map(
        data, indices, instance_2_trimmed
    )

    # Save instance_2_cropped MAP as JSON file
    save_dict_as_JSON_file(INSTANCE_2_CROPPED_PATH, instance_2_cropped)
    # Clean up trimmed videos
    # clean_up_files_in_dir(TRIMMED_DIR)


'''
Downloading, trimming, and then cropping will be done in batches due to the limitations in storage and computation power of the local machine.
The batches are defined/delimited by the range_start and range_end parameters, which define a range of indices
'''


def dwnld_trim_crop_pipeline_range(data, range_start=0, range_end=1):
    # Download the unique instance videos for preprocessing
    url_2_downloaded = download_unique_vids_of_instances_in_range_and_return_url2path_map(
        data, range_start, range_end)
    # Save url_2_downloaded MAP as JSON file
    save_dict_as_JSON_file(URL_2_DOWNLOADED_PATH, url_2_downloaded)

    # Trim videos for further preprocessing
    instance_2_trimmed = trim_vids_for_instances_in_range_and_return_updated_map(
        data, range_start, range_end, url_2_downloaded
    )

    # Save instance_2_trimmed MAP as JSON file
    save_dict_as_JSON_file(INSTANCE_2_TRIMMED_PATH, instance_2_trimmed)
    # Clean up downloaded videos
    # clean_up_files_in_dir(DOWNLOADED_DIR)

    # Cropping Videos to the desired region of interest and saving them
    instance_2_cropped = crop_vids_for_instances_in_range_and_return_updated_map(
        data, range_start, range_end, instance_2_trimmed
    )

    # Save instance_2_cropped MAP as JSON file
    save_dict_as_JSON_file(INSTANCE_2_CROPPED_PATH, instance_2_cropped)
    # Clean up trimmed videos
    # clean_up_files_in_dir(TRIMMED_DIR)
# /Functions

# dummy_indices=[0, 1, 3, 8, 56, 60, 72, 110, 151, 499, 858, 1049]
# dwnld_trim_crop_pipeline_indices(training_data, indices=dummy_indices)
# dwnld_trim_crop_pipeline_range(training_data, 0, 150)
