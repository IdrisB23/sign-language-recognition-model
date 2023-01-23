# idris.babay@cgi.com
# 23.01.2023

'''
This file allows the downloading and cropping of the dataset videos to their corresponding Region Of Interest (ROI) 
'''

## Imports
import os
import json
import re
import pafy
import ffmpeg
import numpy as np

## Constants
DOWNLOADED_VIDS_LIST_FILE_SEPARATOR = ' '

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

DATA_ATTR = ['org_text', 'clean_text', 'start_time', 'signer_id', 'signer', 'start', 'end', 'file', 'label', 'height', 'fps', 'end_time', 'url', 'text', 'box', 'width']

## Load JSON datasets and parse them as Python dictionaries
with open(TRAIN_DATA_PATH, 'r') as f:
    training_data = json.load(f)
    f.close()
with open(VALID_DATA_PATH, 'r') as f:
    valid_data = json.load(f)
    f.close()
with open(TEST_DATA_PATH, 'r') as f:
    test_data = json.load(f)
    f.close()

## Create directories and files if not existant
if not os.path.exists(DOWNLOADED_VIDS_LIST_FILE):
    with open(DOWNLOADED_VIDS_LIST_FILE, 'x') as f:
        f.close()
if not os.path.exists(DOWNLOADED_DIR):
    os.mkdir(DOWNLOADED_DIR)
if not os.path.exists(TRIMMED_DIR):
    os.makedirs(TRIMMED_DIR)
if not os.path.exists(CROPPED_DIR):
    os.makedirs(CROPPED_DIR)

## Functions
'''
Callback function, this callback simply prints the total number of bytes to download, the number of bytes received thusfar, 
the ratio of downloaded bytes, the rate of transfer, and estimated time of arrival (eta)
'''
def mycb(total, recvd, ratio, rate, eta):
    print('total:', total, 'received:', recvd, 'ratio:', ratio, 'rate:', rate, 'eta:', eta)

def remove_http_or_https_from_url(url):
    return re.sub(r'(http|https)://', '', url)

def save_dict_as_JSON_file(path_, data_):
    with open(path_, 'w') as f:
        json.dump(data_, f, indent=2)
        print('Saved dictionary to the JSON file.')
        f.close()

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

'''
This function downloads UNIQUE videos in instances in the data parameter given by the other parameter indices
data: list of instances that have the same dataset attributes
indices: list of integer indices
returns: dictionary that maps each (unique) video url in the dataset to the downloaded video path (may vary depending on the order in which data instances are being reviewed)
'''
def download_unique_vids_of_indices_instances_and_return_url2path_map(data, indices):
    # it starts by loading the unique videos that we hav already downloaded before from a txt file
    # this file is continuously expanded to newly downloaded videos (scroll to the end of THIS function)
    unique_vids_of_first_instances = []
    with open(DOWNLOADED_VIDS_LIST_FILE, 'r') as f:
        for line in f:
            unique_vids_of_first_instances += line.strip().split(DOWNLOADED_VIDS_LIST_FILE_SEPARATOR)
        f.close()
    print('unique_vids_of_first_instances:', unique_vids_of_first_instances)

    # Similarily, it loads url2path map from a potential previous run and this is also updated after each run 
    # (see function preprocessing_pipeline... after THIS function's invocation)
    with open(URL_2_DOWNLOADED_PATH, 'r') as f:
        url2downloaded_vid = json.load(f)
        f.close()
    print('map_:', url2downloaded_vid)

    # go through data points one by one w.r.t. indices
    for i in indices:
        print('Processing instance nr.', str(i))
        instance = data[i]
        vid_url = remove_http_or_https_from_url(instance['url'])
        # check if video url already exists in our list of unique urls
        if vid_url in unique_vids_of_first_instances:
            print('Duplicate video, skipping it...')
            print('-------------------------------------------------')
            continue
        try:
            YT_vid = pafy.new(vid_url)
            s = YT_vid.getbest(preftype='mp4')
            downloaded_vid_path = os.path.join(DOWNLOADED_DIR, str(i) + '.mp4')
            s.download(filepath=downloaded_vid_path, quiet=True, callback=mycb)  ### starts download
            print(downloaded_vid_path)
            unique_vids_of_first_instances.append(vid_url)
            url2downloaded_vid[vid_url] = downloaded_vid_path
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
        finally:
            print('-------------------------------------------------')

    # commit write to the local file for potential future run to avoid duplicate video downloads
    with open(DOWNLOADED_VIDS_LIST_FILE, 'w') as f:
        f.write(DOWNLOADED_VIDS_LIST_FILE_SEPARATOR.join([str(e) for e in unique_vids_of_first_instances]))
        f.close()
    return url2downloaded_vid

'''
The same functionality as download_unique_vids_of_indices_instances_and_return_url2path_map
But this one takes in start and end as a range for indices instead of an explicit list
'''
def download_unique_vids_of_instances_in_range_and_return_url2path_map(data, start, end):
    unique_vids_of_first_instances = []
    with open(DOWNLOADED_VIDS_LIST_FILE, 'r') as f:
        for line in f:
            unique_vids_of_first_instances += line.strip().split(DOWNLOADED_VIDS_LIST_FILE_SEPARATOR)
        f.close()
    print('unique_vids_of_first_instances:', unique_vids_of_first_instances)

    with open(URL_2_DOWNLOADED_PATH, 'r') as f:
        url2downloaded_vid = json.load(f)
        f.close()
    print('map_:', url2downloaded_vid)

    for i in range(start, end):
        print('Processing instance nr.', str(i))
        instance = data[i]
        vid_url = remove_http_or_https_from_url(instance['url'])
        if vid_url in unique_vids_of_first_instances:
            print('Duplicate video, skipping it...')
            print('-------------------------------------------------')
            continue
        try:
            YT_vid = pafy.new(vid_url)
            s = YT_vid.getbest(preftype='mp4')
            downloaded_vid_path = os.path.join(DOWNLOADED_DIR, str(i) + '.mp4')
            s.download(filepath=downloaded_vid_path, quiet=True, callback=mycb)  ### starts download
            print(downloaded_vid_path)
            unique_vids_of_first_instances.append(vid_url)
            url2downloaded_vid[vid_url] = downloaded_vid_path
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
        finally:
            print('-------------------------------------------------')

    with open(DOWNLOADED_VIDS_LIST_FILE, 'w') as f:
        f.write(DOWNLOADED_VIDS_LIST_FILE_SEPARATOR.join([str(e) for e in unique_vids_of_first_instances]))
        f.close()
    
    save_dict_as_JSON_file(URL_2_DOWNLOADED_PATH, url2downloaded_vid)
    return url2downloaded_vid

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
    instance2trimmed = dict()
    for i in indices:
        instance = data[i]
        vid_url = remove_http_or_https_from_url(instance['url'])
        vid_path = url2downloaded_vid.get(vid_url, None)
        # if no hit, then the video was not downloaded as this function is run after downloading the vids in the pipeline
        if vid_path is None:
            print(i, ': [This instance\'s video is private or inaccessible, skipping...]')
            continue
        print(i, ':', vid_url, '=>', vid_path)
        clip_start = instance['start_time']
        clip_end = instance['end_time']
        # trim using the queried attributes
        trimmed_path = os.path.join(TRIMMED_DIR, str(i) + '.mp4')
        trim_in_file_and_save_to_out_file(vid_path, trimmed_path, clip_start, clip_end)
        instance2trimmed[i] = trimmed_path
    return instance2trimmed

'''
The same functionality as trim_vids_for_indices_instances_and_return_updated_map
But this one takes in start and end as a range for indices instead of an explicit list
'''
def trim_vids_for_instances_in_range_and_return_updated_map(data, start, end, url2downloaded_vid):
    instance2trimmed = dict()
    for i in range(start, end):
        instance = data[i]
        vid_url = remove_http_or_https_from_url(instance['url'])
        vid_path = url2downloaded_vid.get(vid_url, None)
        if vid_path is None:
            print(i, ': [This instance\'s video is private or inaccessible, skipping...]')
            continue
        print(i, ':', vid_url, '=>', vid_path)
        clip_start = instance['start_time']
        clip_end = instance['end_time']
        trimmed_path = os.path.join(TRIMMED_DIR, str(i) + '.mp4')
        trim_in_file_and_save_to_out_file(vid_path, trimmed_path, clip_start, clip_end)
        instance2trimmed[i] = trimmed_path
    return instance2trimmed

'''
Crops each instance in data provided by indices according to the instance's relevant ROI attributes
data: list of instances that have the same dataset attributes
indices: list of integer indices
returns: dictionary that maps each instance to the cropped video path
'''
def crop_vids_for_indices_instances_and_return_updated_map(data, indices, instance2trimmed):
    instance2cropped = dict()
    for i in indices:
        instance = data[i]
        # relevant ROI attributes
        y_norm, x_norm, _, _ = instance['box']
        height, width = instance['height'], instance['width']
        
        x_topleft, y_topleft = remove_normalization_coordinates(x_norm, y_norm, width, height)
        
        # again according to the pipeline's order, if it's successful, it should not be None 
        in_vid_path = instance2trimmed.get(i, None)
        if in_vid_path is None:
            print(i, ': [This instance\'s video is private or inaccessible, skipping...]')
            continue
        print(i, '=>', in_vid_path)
        
        out_vid_path = os.path.join(CROPPED_DIR, str(i) + '.mp4')
        
        crop_video(in_vid_path, out_vid_path, x_topleft, y_topleft, width, height)
        # update map/result
        instance2cropped[i] = out_vid_path
        
    return instance2cropped

'''
The same functionality as crop_vids_for_indices_instances_and_return_updated_map
But this one takes in start and end as a range for indices instead of an explicit list
'''
def crop_vids_for_instances_in_range_and_return_updated_map(data, start, end, instance2trimmed):
    result = dict()
    for i in range(start, end):
        y_norm, x_norm, _, _ = data[i]['box']
        height, width = data[i]['height'], data[i]['width']
        
        x_topleft, y_topleft = remove_normalization_coordinates(x_norm, y_norm, width, height)
        
        in_vid_path = instance2trimmed.get(i, None)
        if in_vid_path is None:
            print(i, ': [This instance\'s video is private or inaccessible, skipping...]')
            continue
        print(i, '=>', in_vid_path)
        
        out_vid_path = os.path.join(CROPPED_DIR, str(i) + '.mp4')
        
        crop_video(in_vid_path, out_vid_path, x_topleft, y_topleft, width, height)
        
        result[i] = out_vid_path
        
    return result

'''
All preprocessing will be done in batches due to the limitations in storage and computation power of the local machine.
The batches are defined/delimited by the indices parameter
'''
def preprocessing_pipeline_indices(data, indices=[]):
    ### Download the unique instance videos for preprocessing
    url_2_downloaded = download_unique_vids_of_indices_instances_and_return_url2path_map(data, indices)
    ### Save url_2_downloaded MAP as JSON file
    save_dict_as_JSON_file(URL_2_DOWNLOADED_PATH, url_2_downloaded)
    
    ### Trim videos for further preprocessing 
    instance_2_trimmed = trim_vids_for_indices_instances_and_return_updated_map(
        data, indices, url_2_downloaded
    )
    
    ### Save instance_2_trimmed MAP as JSON file
    save_dict_as_JSON_file(INSTANCE_2_TRIMMED_PATH, instance_2_trimmed)
    ### Clean up downloaded videos
    # clean_up_files_in_dir(DOWNLOADED_DIR)
    
    ### Cropping Videos to the desired region of interest and saving them
    instance_2_cropped = crop_vids_for_indices_instances_and_return_updated_map(
        data, indices, instance_2_trimmed
    )
    
    ### Save instance_2_cropped MAP as JSON file
    save_dict_as_JSON_file(INSTANCE_2_CROPPED_PATH, instance_2_cropped)
    ### Clean up trimmed videos
    clean_up_files_in_dir(TRIMMED_DIR)

'''
All preprocessing will be done in batches due to the limitations in storage and computation power of the local machine.
The batches are defined/delimited by the range_start and range_end parameters, which define a range of indices
'''
def preprocessing_pipeline_range(data, range_start=0, range_end=1):
    ### Download the unique instance videos for preprocessing
    url_2_downloaded = download_unique_vids_of_instances_in_range_and_return_url2path_map(data, range_start, range_end)
    ### Save url_2_downloaded MAP as JSON file
    save_dict_as_JSON_file(URL_2_DOWNLOADED_PATH, url_2_downloaded)
    
    ### Trim videos for further preprocessing 
    instance_2_trimmed = trim_vids_for_instances_in_range_and_return_updated_map(
        data, range_start, range_end, url_2_downloaded
    )
    
    ### Save instance_2_trimmed MAP as JSON file
    save_dict_as_JSON_file(INSTANCE_2_TRIMMED_PATH, instance_2_trimmed)
    ### Clean up downloaded videos
    # clean_up_files_in_dir(DOWNLOADED_DIR)
    
    ### Cropping Videos to the desired region of interest and saving them
    instance_2_cropped = crop_vids_for_instances_in_range_and_return_updated_map(
        data, range_start, range_end, instance_2_trimmed
    )
    
    ### Save instance_2_cropped MAP as JSON file
    save_dict_as_JSON_file(INSTANCE_2_CROPPED_PATH, instance_2_cropped)
    ### Clean up trimmed videos
    clean_up_files_in_dir(TRIMMED_DIR)
## /Functions

# dummy_indices=[0, 1, 3, 8, 56, 60, 72, 110, 151, 499, 858, 1049]
# preprocessing_pipeline_indices(training_data, indices=dummy_indices)
# preprocessing_pipeline_range(training_data, 0, 150)